import sys
import time
import torch
import numpy as np
import cupy as cp
import opt_einsum as oe

from torch import Tensor
from cuquantum import tensornet as cutn
from cuquantum.memory import MemoryLimitExceeded
from cupy.cuda.memory import OutOfMemoryError


# Headroom (bytes) reserved on top of the output tensor when sizing the
# auto-sliced contraction workspace budget, for allocator/framework overhead.
_SLICE_SAFETY_BYTES = 512 * 1024 ** 2


def _oe_path_peak(eq, shapes, itembytes):
    """opt_einsum contraction path (in cupy/numpy ``'einsum_path'`` format) and
    its largest-intermediate size in bytes. opt_einsum's estimate is reliable;
    numpy's ``einsum_path`` under-reports the largest intermediate for dense
    networks (it returns the naive path with speedup 1.0), which would route huge
    contractions down the un-sliced cp.einsum branch and exhaust the pool."""
    path, info = oe.contract_path(
        eq, *[np.empty(s, dtype=np.int8) for s in shapes], optimize="auto")
    return ["einsum_path", *path], float(info.largest_intermediate) * itembytes


def _contract_auto(eq, operands, np_path, out_bytes, peak_bytes):
    """Contract ``eq`` over cupy ``operands``, slicing only when it won't fit.

    Fast path: an un-sliced ``cp.einsum`` with the precomputed NumPy ``np_path``,
    chosen up front whenever the largest intermediate (``peak_bytes``) plus the
    output fit in free GPU memory (the common, low-rank case — no slicing, no
    overhead). Otherwise the contraction goes through cuTensorNet with automatic
    slicing sized to the free memory (reserving ``out_bytes`` for the result).
    The decision is made from the path estimate rather than by letting cp.einsum
    OOM, because a failed cupy allocation does not reliably release its pool.
    A contraction whose output alone won't fit raises ``MemoryLimitExceeded`` so
    callers surface it as OOM-infeasible instead of crashing the run.
    """
    # Comfortably-small contraction: take the fast un-sliced path immediately,
    # without a (serializing) device sync — the common low-rank case.
    total = int(cp.cuda.Device().mem_info[1])
    if peak_bytes + out_bytes < 0.5 * total:
        return cp.einsum(eq, *operands, optimize=np_path)
    # Large: settle pending async work so freed blocks are reclaimed and the
    # free-memory reading (which sizes the slice/no-slice decision) is accurate.
    cp.cuda.Device().synchronize()
    free, _ = cp.cuda.Device().mem_info
    if peak_bytes + out_bytes + _SLICE_SAFETY_BYTES < free:
        return cp.einsum(eq, *operands, optimize=np_path)
    budget = int(free - out_bytes - _SLICE_SAFETY_BYTES)
    if budget <= 0:
        raise MemoryLimitExceeded(int(free), int(out_bytes + _SLICE_SAFETY_BYTES), 0)
    # Explicit Network + free() so the (per-equation) sliced workspace is released
    # after each call; the one-shot cutn.contract retains it and the distinct env
    # equations would otherwise accumulate workspaces and exhaust memory. The
    # returned output is a separate allocation that survives the free.
    net = cutn.Network(eq, *operands,
                       options=cutn.NetworkOptions(memory_limit=budget))
    try:
        net.contract_path()
        return net.contract()
    finally:
        net.free()


# Contraction-workspace budget handed to cuTensorNet's path planner. Expressed as
# a fraction of *total* device memory; the planner raises a clean, catchable
# ``MemoryLimitExceeded`` when a candidate's workspace would exceed this — before
# any real allocation is attempted. Kept strictly below 100% so the remaining
# headroom (~5% ≈ 1.2 GB on a 24 GB A5000) covers the core/target/output
# tensors, ensuring an over-budget structure fails as MemoryLimitExceeded rather
# than as a hard CUDA out-of-memory mid-contraction. Raised from cuTensorNet's
# 80% default to admit slightly larger (but still allocatable) networks.
_DEFAULT_MEMORY_LIMIT = "95%"


# ---------------------------------------------------------------------------
# Backend-agnostic PAM helpers (xp = torch or cupy)
# ---------------------------------------------------------------------------


def _td(a, b, axes, xp):
    """tensordot with uniform signature across backends."""
    if xp is torch:
        return torch.tensordot(a, b, dims=axes)
    return xp.tensordot(a, b, axes=axes)


def _perm(arr, perm, xp):
    """transpose/permute with uniform signature across backends."""
    if xp is torch:
        return arr.permute(perm)
    return xp.transpose(arr, perm)


def _pam_unfold(X, k, xp):
    return xp.moveaxis(X, k, 0).reshape(X.shape[k], -1)


def _pam_fold(mat, k, shape, xp):
    return xp.moveaxis(mat.reshape([shape[k], *shape[:k], *shape[k + 1 :]]), 0, k)


def _pam_gen_unfold(tensor, k, N, xp):
    m = [2 * j + 1 if j < k else 2 * j for j in range(N - 1)]
    n = [2 * j if j < k else 2 * j + 1 for j in range(N - 1)]
    dim1 = int(np.prod([tensor.shape[i] for i in m]))
    dim2 = int(np.prod([tensor.shape[i] for i in n]))
    return _perm(tensor, m + n, xp).reshape(dim1, dim2)


def _pam_fctn_comp(G, xp):
    N = len(G)
    n, m = [0], [1]
    out = G[0]
    for i in range(N - 1):
        out = _td(out, G[i + 1], (m, n), xp)
        n.append(i + 1)
        if i > 0:
            m = [m[0]] + [m[j] - j for j in range(1, i + 1)]
        m.append(1 + (i + 1) * (N - (i + 1)))
    return out


def _pam_fctn_comp_partial(G, skip, xp):
    N = len(G)
    n_list = [i for i in range(N - 1) if i != skip]
    m1 = [1 + i * N for i in range(N - 2)]
    m2 = [2 + i * N for i in range(N - 2)]
    if skip == 0:
        out = G[1]
        for i in range(1, N - 1):
            out = _td(out, G[i + 1], (m2[:i], n_list[:i]), xp)
            m2 = [m2[0]] + [m2[j] - j for j in range(1, N - 2)]
    else:
        out, j = G[0], 0
        for i in range(N - 1):
            if i + 1 < skip:
                out = _td(out, G[i + 1], (m1[: j + 1], n_list[: j + 1]), xp)
                m1 = [m1[0]] + [m1[j] - j for j in range(1, N - 2)]
                m2 = [m2[0]] + [m2[j] - j for j in range(1, N - 2)]
                j += 1
            elif i + 1 > skip:
                out = _td(out, G[i + 1], (m2[: j + 1], n_list[: j + 1]), xp)
                m2 = [m2[0]] + [m2[j] - j for j in range(1, N - 2)]
                j += 1
    return out


from scripts.utils import random_adj_matrix
from typing import Union

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def letter_range(n):
    for c in range(97, 97 + n):
        yield chr(c)


def random_tensor(shape, device, std_dev, backend, backend_dtype, rng=None):
    if backend == "torch":
        if rng is None:
            core = torch.nn.init.normal_(
                torch.nn.Parameter(torch.empty(*shape, dtype=backend_dtype, device=device)),
                mean=0.0,
                std=std_dev,
            )
        else:
            core = torch.randn(
                tuple(shape), dtype=backend_dtype, device=device, generator=rng
            ) * std_dev
    else:
        rand = cp.random if rng is None else rng
        core = rand.normal(loc=0.0, scale=std_dev, size=tuple(shape)).astype(
            backend_dtype
        )
    return core


def _random_cores_from_adj(adj_matrix, std_dev, backend, backend_dtype, rng=None):
    cores = []
    rows = adj_matrix.unbind(0) if hasattr(adj_matrix, "unbind") else adj_matrix
    device = getattr(adj_matrix, "device", None)
    for row in rows:
        shape = row.tolist()
        core = random_tensor(shape, device, std_dev, backend, backend_dtype, rng=rng)
        cores.append(core)
    return cores


def increment_mode_rank(tensor, i):
    if isinstance(tensor, Tensor):
        backend = "torch"
        new_tensor = tensor.clone()
    else:
        backend = "cupy"
        new_tensor = tensor.copy()
    backend_dtype = tensor.dtype
    new_shape = list(tensor.shape)
    new_shape[i] += 1

    padding_shape = list(tensor.shape)
    padding_shape[i] = 1
    padding = random_tensor(
        shape=padding_shape,
        device=tensor.device,
        std_dev=0.1,
        backend=backend,
        backend_dtype=backend_dtype,
    )

    if backend == "torch":
        new_tensor = torch.concat([tensor, padding], dim=i)
    elif backend == "cupy":
        new_tensor = cp.concat([tensor, padding], axis=i)

    assert list(new_tensor.shape) == new_shape

    return new_tensor


# ---------------------------------------------------------------------------
# Contraction-cost collection (cuTensorNet path planner + autotune)
# ---------------------------------------------------------------------------

# Scalar fields safe to flatten into a per-step traces.csv row. The structured
# fields (contract_path, slices, intermediate_modes, eq) are intentionally
# excluded here — they live in the full per-step contraction_traces.json.
CONTRACTION_SCALAR_KEYS = (
    "path_opt_time_s",
    "autotune_time_s",
    "autotune_iterations",
    "opt_cost_flops",
    "largest_intermediate_elements",
    "num_slices",
)


def _jsonify(obj, _depth=0):
    """Best-effort conversion of cuTensorNet info objects to JSON-serializable data.

    Never raises: anything it cannot interpret falls back to ``repr``.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # 0-d numpy/cupy scalars (numpy int64 etc. are not python ints).
    if getattr(obj, "ndim", None) == 0 and hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        if _depth > 8:
            return str(obj)
        return [_jsonify(x, _depth + 1) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonify(v, _depth + 1) for k, v in obj.items()}
    if hasattr(obj, "tolist"):
        try:
            return _jsonify(obj.tolist(), _depth + 1)
        except Exception:
            pass
    return str(obj)


def _extract_contraction_info(info):
    """Pull cost fields off a cuTensorNet path-optimizer ``info`` object.

    Defensive — a missing or unreadable attribute is recorded rather than raised,
    so cost collection can never break a decomposition run.
    """
    fields = {}
    for src, dst in (
        ("opt_cost", "opt_cost_flops"),
        ("largest_intermediate", "largest_intermediate_elements"),
        ("num_slices", "num_slices"),
        ("slices", "slices"),
        ("intermediate_modes", "intermediate_modes"),
        ("path", "contract_path"),
    ):
        try:
            fields[dst] = _jsonify(getattr(info, src, None))
        except Exception as exc:  # noqa: BLE001 — never let collection abort a run
            fields[dst] = f"<unreadable: {exc!r}>"
    return fields


def contraction_scalar_row(stats):
    """Flatten the scalar contraction-cost fields for a flat traces.csv row.

    Keys are prefixed with ``ctn_``. Pass ``None`` (e.g. for non-cuTensorNet
    decomposition methods) to get a row of ``None`` placeholders so the CSV
    schema stays stable across steps.
    """
    stats = stats or {}
    return {f"ctn_{k}": stats.get(k) for k in CONTRACTION_SCALAR_KEYS}


class cuTensorNetwork:
    _DTYPE_OPTIONS = ("float16", "float32", "float64")

    def __init__(
        self,
        adj_matrix: Union[Tensor, cp.ndarray] = None,
        cores=None,
        init_std=0.1,
        backend="cupy",
        dtype="float32",
        memory_limit: Union[int, str, None] = None,
    ) -> None:
        if adj_matrix is None:
            if cores is None:
                raise ValueError(
                    "Must provide at least one of adj_matrix or cores to initialize cuTensorNetwork."
                )
            adj_matrix = np.array([core.shape for core in cores], dtype=np.int32)

        # TODO What about ranks?
        self.backend = backend.lower()
        self.dtype_name = dtype.lower()
        # Workspace budget for the cuTensorNet path planner (GPU/cupy backend).
        self.memory_limit = _DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        if self.backend == "torch":
            self.adj_matrix = torch.maximum(adj_matrix, adj_matrix.T).to(
                dtype=torch.int
            )  # Ensures symmetric adjacency matrix
            backend_dtype = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }[self.dtype_name]
        else:
            adj_matrix = cp.asarray(adj_matrix)
            self.adj_matrix = cp.maximum(adj_matrix, adj_matrix.T).astype(dtype=cp.int8)
            backend_dtype = {
                "float16": cp.float16,
                "float32": cp.float32,
                "float64": cp.float64,
            }[self.dtype_name]

        self._bdtype = backend_dtype
        self.shape = self.adj_matrix.shape
        self.eq = einsum_expr(self.adj_matrix)

        assert self.shape[0] == self.shape[1], "adj_matrix must be a square matrix."

        self.dim = self.shape[0]

        self.cores = []
        self._qualifiers = None
        # Populated by _plan_and_autotune() on each SGD/Adam decomposition pass.
        self.contraction_stats = None
        if cores is None:
            self.cores = _random_cores_from_adj(
                self.adj_matrix, init_std, self.backend, backend_dtype
            )
        else:
            self.cores = self._to_backend_cores(cores)
        self._rebuild_network()

    def _build_cupy_qualifiers(self):
        if not hasattr(cutn, "tensor_qualifiers_dtype"):
            raise NotImplementedError(
                "This cuTensorNet build does not expose tensor_qualifiers_dtype "
                "required for CuPy gradient qualifiers."
            )
        qualifiers = np.zeros(len(self.cores), dtype=cutn.tensor_qualifiers_dtype)
        qualifiers["requires_gradient"] = 1
        return qualifiers

    def _rebuild_network(self):
        if self.backend == "cupy":
            self._qualifiers = self._build_cupy_qualifiers()
            options = cutn.NetworkOptions(memory_limit=self.memory_limit)
            self.ntwrk = cutn.Network(
                self.eq, *self.cores, qualifiers=self._qualifiers, options=options
            )
        else:
            self.ntwrk = cutn.Network(self.eq, *self.cores)
        
        # Native cuTensorNet optimizer is sufficient on a dedicated A5000.
        # Path planning can exceed the memory budget for very large (especially
        # gradient-qualified) networks. Gradient-free methods (PAM/ALS) never
        # contract self.ntwrk, so a failure here is deferred rather than fatal —
        # the SGD/Adam paths re-plan in _plan_and_autotune and surface the OOM
        # through _eval_tn's handler.
        try:
            self.ntwrk.contract_path()
        except MemoryLimitExceeded:
            pass

    # ------------------------------------------------------------------
    # Core checkpointing (FTBOSS warm-start / thaw)
    # ------------------------------------------------------------------

    def _to_backend_cores(self, cores):
        """Move core tensors (host numpy checkpoints or native cupy/torch arrays)
        onto this network's backend device and dtype."""
        if self.backend == "torch":
            return [torch.as_tensor(c).to(self.adj_matrix.device, dtype=self._bdtype)
                    for c in cores]
        return [cp.asarray(c).astype(self._bdtype, copy=False) for c in cores]

    def get_cores(self):
        """Host (numpy) copies of the current cores — checkpoint a partially
        decomposed structure off the GPU so it can be warm-started later."""
        if self.backend == "torch":
            return [c.detach().cpu().numpy() for c in self.cores]
        return [cp.asnumpy(c) for c in self.cores]

    def set_cores(self, cores):
        """Load cores (host or device) and rebuild the network so a subsequent
        :meth:`decompose` warm-continues from them. Returns ``self``."""
        self.cores = self._to_backend_cores(cores)
        self._rebuild_network()
        return self

    def _plan_and_autotune(self, autotune_iterations=5):
        """Plan the contraction path and autotune, recording cost into ``self.contraction_stats``.

        Used by the cuTensorNet-backed SGD/Adam decomposition paths. Returns
        ``(path, info)`` from ``contract_path()`` so callers can build the
        per-iteration optimize config. Cost collection is cheap (path planning
        is microseconds–milliseconds for these network sizes) and fully
        wrapped — a failure to read a field degrades to a placeholder string
        rather than aborting the decomposition.
        """
        t0 = time.perf_counter()
        path, info = self.ntwrk.contract_path()
        path_opt_time_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        autotune_result = self.ntwrk.autotune(iterations=autotune_iterations)
        autotune_time_s = time.perf_counter() - t0

        stats = {
            "backend": self.backend,
            "dtype": self.dtype_name,
            "eq": self.eq,
            "n_cores": int(self.dim),
            "path_opt_time_s": path_opt_time_s,
            "autotune_iterations": int(autotune_iterations),
            "autotune_time_s": autotune_time_s,
            "autotune_result": _jsonify(autotune_result),
        }
        try:
            stats.update(_extract_contraction_info(info))
        except Exception as exc:  # noqa: BLE001
            stats["contraction_info_error"] = repr(exc)
        self.contraction_stats = stats
        return path, info

    def contract_ntwrk(self):
        return self.ntwrk.contract()

    def contract(self) -> Union[Tensor, cp.ndarray]:
        """Alias for contract_ntwrk() to provide a more standard API."""
        return self.contract_ntwrk()

    def network_size(self, adj=None) -> int:
        """Compute TN parameter count directly from ``self.adj_matrix``.

        Each core shape is the corresponding row of the adjacency matrix, so the
        number of parameters in core ``i`` is ``prod_j A[i, j]``. Summing across
        cores gives:

            sum_i prod_j A[i, j]
        """
        A = adj if adj is not None else self.adj_matrix
        if self.backend == "torch":
            row_sizes = A.to(dtype=torch.int64).prod(dim=1)
            return int(row_sizes.sum().item())

        # CuPy path
        row_sizes = cp.prod(A.astype(cp.int64), axis=1)
        return cp.sum(row_sizes)

    def target_size(self, adj=None) -> int:
        """Compute contracted output size from the adjacency diagonal.

        In this TN parameterization, the output tensor modes are exactly the
        diagonal entries of ``self.adj_matrix``, so:

            target_numel = prod_i A[i, i]
        """
        A = adj if adj is not None else self.adj_matrix
        if self.backend == "torch":
            diag_sizes = torch.diagonal(A).to(dtype=torch.int64)
            return int(diag_sizes.prod().item())

        diag_sizes = cp.diagonal(A).astype(cp.int64)
        return cp.prod(diag_sizes)

    def compression_ratio(self, adj=None) -> float:
        """Compute compression ratio using adjacency-derived sizes by default.

        Returns
        -------
        float
            ``original_numel / tn_numel`` (larger means better compression).
        """
        original_numel = self.target_size(adj)
        tn_numel = self.network_size(adj)

        if tn_numel <= 0:
            raise ValueError(
                "TN parameter count from adjacency matrix must be positive."
            )

        return tn_numel/original_numel

    def decompose(
        self,
        target,
        tol=None,
        pct_loss_improvment=0.025,
        init_lr=None,
        loss_patience=500,
        lr_patience=50,
        max_epochs=25000,
        momentum=0.5,
        method="sgd",
        warm_start_method=None,
        warm_start_epochs=0,
        callback=None,
        **kwargs,
    ):
        # `callback`, if given, is invoked once per epoch of the MAIN pass as
        # `callback(rse) -> bool`; returning True breaks the decomposition early
        # (e.g. BOS feasibility stopping). The warm-start pass is never callbacked.
        # Automatically toggle Adam if requested via method string
        use_adam = kwargs.get("use_adam", False) or (method == "adam")
        kwargs["use_adam"] = use_adam

        if init_lr is None:
            init_lr = 0.01

        # --- Optional warm-start pass (modifies self.cores in-place) ---
        warm_losses = []
        if warm_start_method and warm_start_epochs > 0:
            warm_kw = dict(
                target=target, tol=tol, max_epochs=warm_start_epochs,
                **kwargs
            )
            if warm_start_method == "pam":
                warm_losses = self._decompose_pam(**warm_kw)
            elif warm_start_method == "als":
                warm_losses = self._decompose_als_cp(**warm_kw)
            elif warm_start_method == "agd":
                warm_losses = self._decompose_agd(**warm_kw)
            elif warm_start_method == "sgd":
                sgd_kw = dict(
                    target=target,
                    tol=tol,
                    max_epochs=warm_start_epochs,
                    init_lr=init_lr,
                    loss_patience=loss_patience,
                    lr_patience=lr_patience,
                    momentum=momentum,
                    pct_loss_improvment=pct_loss_improvment,
                    **kwargs,
                )
                if self.backend == "torch":
                    warm_losses = self._decompose_torch_sgd(**sgd_kw)
                else:
                    warm_losses = self._decompose_cupy_cutn_sgd(**sgd_kw)
            # Rebuild network after warm-start mutated cores
            self._rebuild_network()

        # --- Main decomposition pass ---
        main_kwargs = dict(
            target=target,
            tol=tol,
            pct_loss_improvment=pct_loss_improvment,
            init_lr=init_lr,
            loss_patience=loss_patience,
            lr_patience=lr_patience,
            max_epochs=max_epochs,
            momentum=momentum,
            callback=callback,
            **kwargs,
        )
        if method == "als":
            main_losses = self._decompose_als_cp(**main_kwargs)
        elif method == "pam":
            main_losses = self._decompose_pam(**main_kwargs)
        elif method == "agd":
            main_losses = self._decompose_agd(**main_kwargs)
        elif method in ("sgd", "adam"):
            if self.backend == "torch":
                main_losses = self._decompose_torch_sgd(**main_kwargs)
            else:
                main_losses = self._decompose_cupy_cutn_sgd(**main_kwargs)
        else:
            raise ValueError(f"Unsupported method '{method}' or backend '{self.backend}'.")

        return warm_losses + main_losses

    def _decompose_torch_sgd(
        self,
        target,
        tol=None,
        pct_loss_improvment=0.025,
        init_lr=0.01,
        loss_patience=500,
        lr_patience=50,
        max_epochs=25000,
        momentum=0.5,
        **kwargs,
    ):
        if self.backend != "torch":
            raise NotImplementedError(
                "decompose currently supports only backend='torch' (SGD + autograd)."
            )

        callback = kwargs.get("callback")
        if not self.cores:
            raise ValueError("Network has no tensors to optimize.")

        params = []
        for core in self.cores:
            if not isinstance(core, torch.Tensor):
                raise TypeError("All cores must be torch.Tensor for backend='torch'.")
            if not core.requires_grad:
                core.requires_grad_(True)
            params.append(core)

        optimizer = torch.optim.SGD(params, lr=init_lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=1 / torch.e, patience=lr_patience
        )

        ref_tensor = self.cores[0]
        target = target.to(dtype=ref_tensor.dtype, device=ref_tensor.device)
        target_norm = torch.norm(target).clamp_min(torch.finfo(target.dtype).eps)

        path, info = self._plan_and_autotune(autotune_iterations=5)
        slices = getattr(info, "slices", None)
        optimize_cfg = {"path": path, "slicing": slices}

        loss = torch.inf
        best_loss = loss
        wait = 0
        epoch = 0
        min_delta = 0

        while epoch < max_epochs:
            optimizer.zero_grad(set_to_none=True)

            recon = cutn.contract(self.eq, *self.cores, optimize=optimize_cfg)
            loss = torch.norm(target - recon) / target_norm

            loss.backward()
            optimizer.step()

            epoch += 1
            loss_value = float(loss.detach().item())

            if callback is not None and callback(loss_value):
                break

            if loss_value < best_loss - min_delta:
                best_loss = loss_value
                wait = 0
                min_delta = best_loss * pct_loss_improvment
            else:
                wait += 1

            if wait >= loss_patience:
                break

            if tol is not None and loss_value <= tol:
                break

            scheduler.step(loss.detach())
            if epoch % 100 == 0:
                sys.stdout.flush()
                print(
                    f"\rEpoch {epoch}, Loss: {loss_value:0.5f}, "
                    f"Learning Rate: {optimizer.param_groups[0]['lr']:0.6f}"
        )

        return [float(loss.detach().item())]

    def _decompose_cupy_cutn_sgd(
        self,
        target,
        tol=0.01,
        pct_loss_improvment=0.025,
        init_lr=0.01,
        loss_patience=500,
        lr_patience=50,
        max_epochs=25000,
        momentum=0.55,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        **kwargs,
    ):
        optimizer = "adam" if kwargs.get("use_adam", False) else "sgd"
        callback = kwargs.get("callback")
        if not self.cores:
            raise ValueError("Network has no tensors to optimize.")
        if self._qualifiers is None:
            raise RuntimeError(
                "CuPy qualifiers were not initialized for gradient computation."
            )
        if not hasattr(self.ntwrk, "gradients"):
            raise NotImplementedError(
                "This cuTensorNet build does not expose Network.gradients(). "
                "Upgrade cuQuantum to use CuPy + cuTensorNet-native gradients."
            )

        target = cp.asarray(target)
        target = target.astype(self.cores[0].dtype, copy=False)
        target_norm = cp.linalg.norm(target)

        self._plan_and_autotune(autotune_iterations=5)

        lr = float(init_lr)
        velocity = [cp.zeros_like(node) for node in self.cores]
        m = [cp.zeros_like(node) for node in self.cores]  # Adam 1st moment
        v = [cp.zeros_like(node) for node in self.cores]  # Adam 2nd moment
        epoch = 0
        wait = 0
        best_loss = float("inf")
        min_delta = 0.0
        bad_lr_steps = 0
        loss_history = []

        unfrozen = {}
        unfrozen_edge = kwargs.get("unfrozen_edge", None)
        unfrozen_rank_inc = int(kwargs.get("unfrozen_rank_inc", 1))
        if unfrozen_edge is not None:
            u, v = unfrozen_edge
            unfrozen[u] = [(v, unfrozen_rank_inc)]
            unfrozen[v] = [(u, unfrozen_rank_inc)]
        active_masks = [None] * len(self.cores)
        if unfrozen_edge is not None:
            for idx, core in enumerate(self.cores):
                if idx not in unfrozen:
                    continue
                active = cp.zeros(core.shape, dtype=cp.bool_)
                for mode_idx, n_new in unfrozen[idx]:
                    sl = [slice(None)] * core.ndim
                    sl[mode_idx] = slice(-n_new, None)
                    active[tuple(sl)] = True
                active_masks[idx] = active

        while epoch < max_epochs:
            contracted_t = self.ntwrk.contract()
            residual = contracted_t - target
            loss = cp.linalg.norm(residual) / target_norm
            loss_history.append(loss.item())
            if callback is not None and callback(loss_history[-1]):
                break

            residual_norm = cp.maximum(
                cp.linalg.norm(residual), cp.finfo(residual.dtype).eps
            )
            output_grad = residual / (target_norm * residual_norm)

            grads = self.ntwrk.gradients(output_gradient=output_grad)

            if len(grads) != len(self.cores):
                raise RuntimeError(
                    "Gradient count mismatch with number of network cores."
                )

            for idx, grad in enumerate(grads):
                if epoch < max_epochs // 2 and unfrozen_edge is not None:
                    active = active_masks[idx]
                    if active is not None:
                        # Fallback to SGD for masked updates if needed, though Adam works too
                        velocity[idx][active] = (
                            momentum * velocity[idx][active] + grad[active]
                        )
                        self.cores[idx][active] -= lr * velocity[idx][active]
                        velocity[idx][~active] = 0
                    else:
                        velocity[idx].fill(0)
                else:
                    if optimizer == "adam":
                        m[idx] = beta1 * m[idx] + (1 - beta1) * grad
                        v[idx] = beta2 * v[idx] + (1 - beta2) * (grad**2)
                        m_hat = m[idx] / (1 - beta1 ** (epoch + 1))
                        v_hat = v[idx] / (1 - beta2 ** (epoch + 1))
                        self.cores[idx] -= lr * m_hat / (cp.sqrt(v_hat) + eps)
                    else:
                        velocity[idx] = momentum * velocity[idx] + grad
                        self.cores[idx] -= lr * velocity[idx]

            epoch += 1

            if loss_history[-1] < best_loss - min_delta:
                best_loss = loss_history[-1]
                wait = 0
                min_delta = best_loss * pct_loss_improvment
                bad_lr_steps = 0
            else:
                wait += 1
                bad_lr_steps += 1

            if bad_lr_steps >= lr_patience:
                lr *= 1 / np.e
                bad_lr_steps = 0

            if wait >= loss_patience:
                break
            if tol is not None and loss_history[-1] <= tol:
                break

            if epoch % 100 == 0:
                sys.stdout.flush()
                if kwargs.get("verbose", False):
                    print(
                        f"\rEpoch {epoch}, Loss: {loss_history[-1]:0.5f}, Learning Rate: {lr:0.6f}"
                    )

        return loss_history

    def _decompose_als_cp(
        self, target, tol=0.01, pct_loss_improvment=0.025,
        **kwargs
    ):
        if self.backend != "cupy":
            raise NotImplementedError(
                "ALS decomposition is implemented only for backend='cupy'."
            )
        if not self.cores:
            raise ValueError("Network has no tensors to optimize.")

        max_iter = int(kwargs.get("max_iter", kwargs.get("max_epochs", 500)))
        cvg_threshold = float(kwargs.get("cvg_threshold", 1e-7))
        verbose = int(kwargs.get("verbose", -1))
        vertices = tuple(kwargs.get("vertices", range(len(self.cores))))
        callback = kwargs.get("callback")

        target = cp.asarray(target).astype(self.cores[0].dtype, copy=False)
        eps = cp.finfo(target.dtype).eps
        target_norm = cp.maximum(cp.linalg.norm(target), eps)

        provided_unfold = kwargs.get("target_unfold_list")
        if provided_unfold is None:
            target_unfold_list = [
                cp.moveaxis(target, v, 0).reshape(target.shape[v], -1)
                for v in range(len(self.cores))
            ]
        else:
            target_unfold_list = [cp.asarray(tu) for tu in provided_unfold]

        lhs, rhs = self.eq.split("->")
        input_terms = lhs.split(",")
        num_cores = len(self.cores)

        used_labels = set(lhs.replace(",", "") + rhs)
        label_pool = [
            c
            for c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if c not in used_labels
        ]

        env_specs = {}
        for v in vertices:
            operand_ids = tuple(i for i in range(num_cores) if i != v)
            if len(label_pool) < len(operand_ids):
                raise ValueError(
                    "Not enough einsum labels to build ALS environment contractions."
                )

            aux_labels = label_pool[: len(operand_ids)]
            env_terms = []
            for k, i in enumerate(operand_ids):
                term = list(input_terms[i])
                term[v] = aux_labels[k]
                env_terms.append("".join(term))

            env_out = "".join(rhs[i] for i in operand_ids) + "".join(aux_labels)
            env_eq = f"{','.join(env_terms)}->{env_out}"

            # Pre-compute optimal contraction path once for ALS environment
            # Use np.empty shapes to avoid host copies
            np_env_ops = [np.empty(self.cores[i].shape) for i in operand_ids]
            env_path = np.einsum_path(env_eq, *np_env_ops, optimize="optimal")[0]

            n_rows = int(
                np.prod([int(target.shape[i]) for i in operand_ids], dtype=np.int64)
            )
            n_cols = int(
                np.prod(
                    [int(self.cores[i].shape[v]) for i in operand_ids], dtype=np.int64
                )
            )
            solve_shape = list(self.cores[v].shape)
            solve_shape.insert(0, solve_shape.pop(v))
            env_specs[v] = (env_eq, env_path, n_rows, n_cols, solve_shape)

        # Pre-compute full reconstruction path for ALS loop
        np_full_ops = [np.empty(c.shape) for c in self.cores]
        full_recon_path = np.einsum_path(self.eq, *np_full_ops, optimize="optimal")[0]

        prev_loss = float("inf")
        loss_history = []

        for it in range(max_iter):
            for v in vertices:
                env_eq, env_path, n_rows, n_cols, solve_shape = env_specs[v]
                env = cp.einsum(
                    env_eq,
                    *[self.cores[i] for i in range(num_cores) if i != v],
                    optimize=env_path,
                ).reshape(n_rows, n_cols)
                sol = cp.linalg.lstsq(env, target_unfold_list[v].T, rcond=None)[0].T
                updated = cp.moveaxis(sol.reshape(solve_shape), 0, v)
                self.cores[v][...] = updated

            recon = cp.einsum(self.eq, *self.cores, optimize=full_recon_path)
            loss = float((cp.linalg.norm(recon - target) / target_norm).item())
            loss_history.append(loss)

            if verbose > 0:
                print(it, ":", loss)

            if callback is not None and callback(loss):
                break
            if tol is not None and loss <= tol:
                break

            if np.isfinite(prev_loss):
                min_delta = max(cvg_threshold, prev_loss * pct_loss_improvment)
                if abs(prev_loss - loss) < min_delta:
                    break
            prev_loss = loss

        return loss_history

    def _decompose_pam(
        self, target, tol=None, max_epochs=1000, rho=0.1,
        verbose=False, **kwargs
    ):
        """PAM — optimized CuPy port of decomp_pam.

        Same algorithm as decomp_pam / _pam_fctn_comp_partial + pinv, but faster
        and memory-robust:
          - env / reconstruction contractions go through :func:`_contract_auto`,
            which falls back to auto-sliced cuTensorNet when the intermediates
            don't fit (so high-rank structures are evaluable, not OOM-crashing);
          - the per-core ridge solve uses the push-through identity to solve in
            ``min(n_rows, n_cols)`` dimensions. For an over-parameterised core
            (``n_cols = prod(bond ranks) > n_rows = prod(other modes)``) this
            solves the bounded ``MMᵀ`` system instead of the rank-sized ``MᵀM``,
            giving the identical update at a cost set by the problem, not R.
        """
        if self.backend != "cupy":
            raise NotImplementedError(
                f"PAM decomposition requires backend='cupy', got '{self.backend}'."
            )

        xp = cp
        callback = kwargs.get("callback")
        target = xp.asarray(target).astype(self.cores[0].dtype, copy=False)
        N = len(self.cores)
        (itembytes, target_norm, env_specs, full_path, recon_peak,
         recon_bytes, big_mode) = self._alt_env_setup(target)

        loss_history = []
        for _ in range(max_epochs):
            for k in range(N):
                env_eq, other_ids, n_rows, n_cols, path_arg, env_peak = env_specs[k]
                Xk = _pam_unfold(target, k, xp)
                Gk = _pam_unfold(self.cores[k], k, xp)

                M = _contract_auto(
                    env_eq, [self.cores[i] for i in other_ids], path_arg,
                    n_rows * n_cols * itembytes, env_peak,
                ).reshape(n_rows, n_cols)

                # Ridge / proximal core update solved in the smaller dimension
                # (push-through identity — identical result):
                #   n_cols-space  (MᵀM)  when the core is determined,
                #   n_rows-space  (MMᵀ)  when it is over-parameterised (high rank).
                if n_cols <= n_rows:
                    tempA = M.T @ M + rho * xp.eye(n_cols, dtype=Gk.dtype)
                    tempC = Xk @ M + rho * Gk
                    new_Gk = xp.linalg.solve(tempA, tempC.T).T
                else:
                    S = M @ M.T + rho * xp.eye(n_rows, dtype=Gk.dtype)
                    resid = Xk - Gk @ M.T
                    new_Gk = Gk + xp.linalg.solve(S, resid.T).T @ M

                core_shape = [int(x) for x in self.adj_matrix[k].tolist()]
                self.cores[k] = _pam_fold(new_Gk, k, core_shape, xp)

                # On high-rank (sliced) structures, release the per-core env
                # matrix + solve temporaries back to the OS so they don't starve
                # the next core's sliced-contraction workspace. Skipped in the
                # ordinary fast-path regime where the pool simply reuses blocks.
                if big_mode:
                    del M, new_Gk
                    xp.cuda.Device().synchronize()
                    xp.get_default_memory_pool().free_all_blocks()

            recon = _contract_auto(self.eq, list(self.cores), full_path,
                                   recon_bytes, recon_peak)
            loss_val = xp.linalg.norm(recon - target) / target_norm
            loss_history.append(float(loss_val.item()))
            if verbose:
                d = (loss_history[-1] - loss_history[-2]) if len(loss_history) > 1 else float("nan")
                print(f"[pam {len(loss_history):>4}/{max_epochs}] "
                      f"rse={loss_history[-1]:.4e}  Δ={d:+.2e}")
            if callback is not None and callback(loss_history[-1]):
                break
            if tol is not None and loss_history[-1] <= tol:
                break

        return loss_history

    def _alt_env_setup(self, target):
        """Shared setup for the alternating decomposers (PAM / AGD).

        Builds, once: each core's environment einsum (contract all *other*
        cores) with its opt_einsum path and largest-intermediate estimate; the
        full reconstruction path; and a ``big_mode`` flag for whether any
        contraction is large enough to need cuTensorNet slicing + the per-core
        workspace cleanup. Returns ``(itembytes, target_norm, env_specs,
        full_path, recon_peak, recon_bytes, big_mode)`` where each ``env_specs``
        entry is ``(env_eq, other_ids, n_rows, n_cols, path, peak_bytes)``.
        """
        xp = cp
        itembytes = int(target.itemsize)
        eps = xp.finfo(target.dtype).eps
        target_norm = xp.maximum(xp.linalg.norm(target), eps)

        lhs, rhs = self.eq.split("->")
        input_terms = lhs.split(",")
        label_to_dim = {
            label: int(dim)
            for i in range(self.dim)
            for label, dim in zip(input_terms[i], self.cores[i].shape)
        }
        env_specs = []
        for k in range(self.dim):
            other_ids = [i for i in range(self.dim) if i != k]
            env_terms = [input_terms[i] for i in other_ids]
            shared_bonds = "".join(
                c for c in input_terms[k]
                if any(c in input_terms[j] for j in other_ids))
            other_phys = "".join(c for c in rhs if c != rhs[k])
            env_eq = ",".join(env_terms) + "->" + other_phys + shared_bonds
            n_rows = int(np.prod([label_to_dim[c] for c in other_phys]))
            n_cols = int(np.prod([label_to_dim[c] for c in shared_bonds]))
            path, peak_bytes = _oe_path_peak(
                env_eq, [self.cores[i].shape for i in other_ids], itembytes)
            env_specs.append((env_eq, other_ids, n_rows, n_cols, path, peak_bytes))

        full_path, recon_peak = _oe_path_peak(
            self.eq, [c.shape for c in self.cores], itembytes)
        recon_bytes = int(target.size) * itembytes
        total = int(cp.cuda.Device().mem_info[1])
        big_mode = (
            max(p + n_r * n_c * itembytes
                for (_, _, n_r, n_c, _, p) in env_specs) >= 0.5 * total
            or recon_peak + recon_bytes >= 0.5 * total
        )
        return (itembytes, target_norm, env_specs, full_path, recon_peak,
                recon_bytes, big_mode)

    def _decompose_agd(self, target, tol=None, max_epochs=1000,
                       verbose=False, **kwargs):
        """AGD — alternating first-order FCTN decomposition.

        Sweeps the cores using the environment matrix ``M`` (same gradient-free,
        auto-sliced contractions as PAM) and updates each core with a single
        gradient step on ``½‖X_(k) − G_k Mᵀ‖²`` — no inverse and no Gram matrix
        ``MᵀM``. The step is the exact line-search optimum for the per-core
        quadratic, ``α* = ‖g‖²/‖g Mᵀ‖²`` (no tuning).
        """
        if self.backend != "cupy":
            raise NotImplementedError(
                f"AGD decomposition requires backend='cupy', got '{self.backend}'.")
        xp = cp
        callback = kwargs.get("callback")
        target = xp.asarray(target).astype(self.cores[0].dtype, copy=False)
        N = len(self.cores)
        (itembytes, target_norm, env_specs, full_path, recon_peak,
         recon_bytes, big_mode) = self._alt_env_setup(target)

        loss_history = []
        for t in range(max_epochs):
            for k in range(N):
                env_eq, other_ids, n_rows, n_cols, path_arg, env_peak = env_specs[k]
                Xk = _pam_unfold(target, k, xp)
                M = _contract_auto(
                    env_eq, [self.cores[i] for i in other_ids], path_arg,
                    n_rows * n_cols * itembytes, env_peak,
                ).reshape(n_rows, n_cols)

                Gk = _pam_unfold(self.cores[k], k, xp)

                # Gradient of ½‖Xk − Gk Mᵀ‖²  w.r.t. Gk  (no MᵀM, no inverse).
                grad = (Gk @ M.T - Xk) @ M       # (I_k, n_cols)
                # AGD: exact line search  α* = ‖g‖²/‖g Mᵀ‖²  (no tuning).
                gMt = grad @ M.T
                alpha = float((grad * grad).sum()) / (float((gMt * gMt).sum()) + 1e-30)
                new_Gk = Gk - alpha * grad

                core_shape = [int(x) for x in self.adj_matrix[k].tolist()]
                self.cores[k] = _pam_fold(new_Gk, k, core_shape, xp)

                if big_mode:
                    del M, grad, new_Gk
                    xp.cuda.Device().synchronize()
                    xp.get_default_memory_pool().free_all_blocks()

            recon = _contract_auto(self.eq, list(self.cores), full_path,
                                   recon_bytes, recon_peak)
            loss_val = xp.linalg.norm(recon - target) / target_norm
            loss_history.append(float(loss_val.item()))
            if verbose:
                d = (loss_history[-1] - loss_history[-2]) if len(loss_history) > 1 else float("nan")
                print(f"[agd {len(loss_history):>4}/{max_epochs}] "
                      f"rse={loss_history[-1]:.4e}  Δ={d:+.2e}")
            if callback is not None and callback(loss_history[-1]):
                break
            if tol is not None and loss_history[-1] <= tol:
                break

        return loss_history


def einsum_expr(adj_matrix, keep_rank1=True):
    dim = adj_matrix.shape[0]
    labels = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    edge_labels = {}  # key: (min(i,j), max(i,j)) -> einsum char

    input_terms = []
    output_labels = []

    for i in range(dim):
        term = []
        for j in range(dim):
            if i == j:
                # Diagonal axis is the open/output mode for core i.
                out_lbl = next(labels)
                term.append(out_lbl)
                output_labels.append(out_lbl)
            else:
                rank = int(adj_matrix[i, j])
                include_edge = (rank >= 1) if keep_rank1 else (rank > 1)
                if not include_edge:
                    continue
                key = (i, j) if i < j else (j, i)
                if key not in edge_labels:
                    edge_labels[key] = next(labels)
                term.append(edge_labels[key])

        input_terms.append("".join(term))

    lhs = ",".join(input_terms)
    rhs = "".join(output_labels)
    return f"{lhs}->{rhs}"


def sim_tensor_from_adj(A, std_dev=0.1, backend="torch", dtype="float32", seed=None):
    backend_name = backend.lower()
    dtype_name = dtype.lower()

    if backend_name == "torch":
        A = torch.as_tensor(A).to(torch.int)
        adj = torch.maximum(A, A.T)
        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[dtype_name]
    elif backend_name == "cupy":
        A = cp.asarray(A).astype(cp.int16)
        adj = cp.maximum(A, A.T)
        cupy_dtype = {
            "float16": cp.float16,
            "float32": cp.float32,
            "float64": cp.float64,
        }[dtype_name]
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'torch' or 'cupy'.")

    backend_dtype = torch_dtype if backend_name == "torch" else cupy_dtype
    if seed is None:
        rng = None
    elif backend_name == "torch":
        rng = torch.Generator(device="cpu")
        rng.manual_seed(int(seed))
    else:
        rng = cp.random.RandomState(int(seed))
    cores = _random_cores_from_adj(adj, std_dev, backend_name, backend_dtype, rng=rng)

    ntwrk = cuTensorNetwork(adj, cores=cores, backend=backend_name, dtype=dtype_name)
    return ntwrk.contract_ntwrk(), cores


if __name__ == "__main__":
    torch.manual_seed(1)

    N = 7
    max_rank = 7
    A = random_adj_matrix(N, max_rank)
    tgt, cores = sim_tensor_from_adj(A, backend="cupy", dtype="float32")

    # increment_mode_rank(cores[0], 2)

    ctn = cuTensorNetwork(A, backend="cupy", dtype="float32")
    loss = ctn.decompose(
        tgt, tol=1e-8, init_lr=0.01, lr_patience=250, loss_patience=2500, max_epochs=200, method="pam", verbose=True
    )
    print(loss)
