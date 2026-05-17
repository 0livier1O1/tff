import sys
import torch
import numpy as np
import cupy as cp

from torch import Tensor
from cuquantum import tensornet as cutn


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


class cuTensorNetwork:
    _DTYPE_OPTIONS = ("float16", "float32", "float64")

    def __init__(
        self,
        adj_matrix: Union[Tensor, cp.ndarray] = None,
        cores=None,
        init_std=0.1,
        backend="cupy",
        dtype="float32",
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

        self.shape = self.adj_matrix.shape
        self.eq = einsum_expr(self.adj_matrix)

        assert self.shape[0] == self.shape[1], "adj_matrix must be a square matrix."

        self.dim = self.shape[0]

        self.cores = []
        self._qualifiers = None
        if cores is None:
            self.cores = _random_cores_from_adj(
                self.adj_matrix, init_std, self.backend, backend_dtype
            )
        else:
            if self.backend == "torch":
                self.cores = [core.clone().to(self.adj_matrix.device) for core in cores]
            else:
                self.cores = [core.copy() for core in cores]
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
            self.ntwrk = cutn.Network(self.eq, *self.cores, qualifiers=self._qualifiers)
        else:
            self.ntwrk = cutn.Network(self.eq, *self.cores)
        
        # Native cuTensorNet optimizer is sufficient on a dedicated A5000.
        self.ntwrk.contract_path()

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
        loss_patience=2500,
        lr_patience=250,
        max_epochs=25000,
        momentum=0.5,
        method="sgd",
        warm_start_method=None,
        warm_start_epochs=0,
        **kwargs,
    ):
        # Automatically toggle Adam if requested via method string
        use_adam = kwargs.get("use_adam", False) or (method == "adam")
        kwargs["use_adam"] = use_adam

        if init_lr is None:
            init_lr = 0.002 if use_adam else 0.01

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
            **kwargs,
        )
        if method == "als":
            main_losses = self._decompose_als_cp(**main_kwargs)
        elif method == "pam":
            main_losses = self._decompose_pam(**main_kwargs)
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
        init_lr=0.05,
        loss_patience=2500,
        lr_patience=250,
        max_epochs=25000,
        momentum=0.5,
        **kwargs,
    ):
        if self.backend != "torch":
            raise NotImplementedError(
                "decompose currently supports only backend='torch' (SGD + autograd)."
            )

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

        path, info = self.ntwrk.contract_path()
        self.ntwrk.autotune(iterations=5)
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
        init_lr=0.05,
        loss_patience=2500,
        lr_patience=250,
        max_epochs=25000,
        momentum=0.55,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        **kwargs,
    ):
        optimizer = "adam" if kwargs.get("use_adam", False) else "sgd"
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

        self.ntwrk.contract_path()
        self.ntwrk.autotune(iterations=5)

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
        **kwargs
    ):
        """PAM — optimized CuPy/Torch port of decomp_pam.

        Same algorithm as decomp_pam / _pam_fctn_comp_partial + pinv, but faster:
          - xp.einsum(optimize=path) replaces sequential tensordot for env contraction
          - xp.linalg.solve replaces pinv (tempA is symmetric PD; LU >> SVD)
        """
        if self.backend != "cupy":
            raise NotImplementedError(
                f"PAM decomposition requires backend='cupy', got '{self.backend}'."
            )

        xp = cp
        target = xp.asarray(target).astype(self.cores[0].dtype, copy=False)
        eps = xp.finfo(target.dtype).eps
        target_norm = xp.maximum(xp.linalg.norm(target), eps)
        N = len(self.cores)

        # Pre-compute per-core environment equations and paths once
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
                [
                    c
                    for c in input_terms[k]
                    if any(c in input_terms[j] for j in other_ids)
                ]
            )
            other_phys = "".join([c for c in rhs if c != rhs[k]])
            env_eq = ",".join(env_terms) + "->" + other_phys + shared_bonds

            n_rows = int(np.prod([label_to_dim[c] for c in other_phys]))
            n_cols = int(np.prod([label_to_dim[c] for c in shared_bonds]))

            # Find path once using numpy (CPU-side is fast for N=6)
            path_info = np.einsum_path(
                env_eq,
                *[np.empty(self.cores[i].shape) for i in other_ids],
                optimize="optimal",
            )
            path = path_info[0]
            env_specs.append((env_eq, other_ids, n_rows, n_cols, path))

        full_path_info = np.einsum_path(
            self.eq, *[np.empty(c.shape) for c in self.cores], optimize="optimal"
        )
        full_path = full_path_info[0]

        loss_history = []
        for _ in range(max_epochs):
            for k in range(N):
                env_eq, other_ids, n_rows, n_cols, path_arg = env_specs[k]
                Xk = _pam_unfold(target, k, xp)
                Gk = _pam_unfold(self.cores[k], k, xp)

                M = xp.einsum(
                    env_eq, *[self.cores[i] for i in other_ids], optimize=path_arg
                ).reshape(n_rows, n_cols)
                tempC = Xk @ M + rho * Gk
                tempA = M.T @ M + rho * xp.eye(n_cols, dtype=Gk.dtype)

                core_shape = [int(x) for x in self.adj_matrix[k].tolist()]
                self.cores[k] = _pam_fold(
                    xp.linalg.solve(tempA, tempC.T).T, k, core_shape, xp
                )

            recon = xp.einsum(self.eq, *self.cores, optimize=full_path)
            loss_val = xp.linalg.norm(recon - target) / target_norm
            loss_history.append(float(loss_val.item()))
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

    N = 5
    max_rank = 7
    A = random_adj_matrix(N, max_rank)
    tgt, cores = sim_tensor_from_adj(A, backend="cupy", dtype="float32")

    # increment_mode_rank(cores[0], 2)

    ctn = cuTensorNetwork(A, backend="cupy", dtype="float32")
    loss = ctn.decompose(
        tgt, tol=1e-8, init_lr=0.5, loss_patience=2500, max_epochs=5000, method="sgd"
    )
    print(loss)
