import sys
import numpy as np
import torch

from torch import Tensor
from cuquantum import tensornet as cutn

from scripts.utils import random_adj_matrix

import cupy as cp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def letter_range(n):
	for c in range(97, 97+n):
		yield chr(c)


def _random_cores_from_adj(adj_matrix, std_dev, backend, backend_dtype):
    cores = []
    for row in adj_matrix.unbind():
        shape = [m for m in row.tolist() if m > 1]
        if backend == "torch":
            core = torch.nn.init.normal_(
                    torch.nn.Parameter(torch.empty(*shape, dtype=torch.double, device=device)), 
                    mean=0.0, 
                    std=std_dev
                )
        else:
            core = cp.random.normal(
                loc=0.0,
                scale=std_dev,
                size=tuple(shape)
            ).astype(backend_dtype)
        cores.append(core)
    return cores


class cuTensorNetwork:
    _DTYPE_OPTIONS = ("float16", "float32", "float64")

    def __init__(self, adj_matrix: Tensor, cores=None, init_std=0.1, backend="torch", dtype="float32") -> None:
        # TODO What about ranks?
        self.adj_matrix = torch.maximum(adj_matrix, adj_matrix.T).to(dtype=torch.int)  # Ensures symmetric adjacency matrix
        self.modes = torch.diag(self.adj_matrix).tolist()  # Save ranks
        self.shape = self.adj_matrix.shape
        self.backend = backend.lower()
        self.dtype_name = dtype.lower()

        self.eq = einsum_expr(self.adj_matrix)

        assert self.shape[0] == self.shape[1], 'adj_matrix must be a square matrix.'
        if self.backend not in ("torch", "cupy"):
            raise ValueError(f"Unsupported backend '{backend}'. Use 'torch' or 'cupy'.")
        if self.dtype_name not in self._DTYPE_OPTIONS:
            raise ValueError(
                f"Unsupported dtype '{dtype}'. Options: {self._DTYPE_OPTIONS}."
            )
        if self.backend == "torch":
            backend_dtype = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }[self.dtype_name]
        else:
            backend_dtype = {
                "float16": cp.float16,
                "float32": cp.float32,
                "float64": cp.float64,
            }[self.dtype_name]
        
        self.dim = self.shape[0]

        self.nodes = []
        self._qualifiers = None
        if cores is None:
            self.nodes = _random_cores_from_adj(
                self.adj_matrix, init_std, self.backend, backend_dtype
            )
        else:
            self.nodes = list(cores)
        self._rebuild_network()

    def _build_cupy_qualifiers(self):
        if not hasattr(cutn, "tensor_qualifiers_dtype"):
            raise NotImplementedError(
                "This cuTensorNet build does not expose tensor_qualifiers_dtype "
                "required for CuPy gradient qualifiers."
            )
        qualifiers = np.zeros(len(self.nodes), dtype=cutn.tensor_qualifiers_dtype)
        qualifiers["requires_gradient"] = 1
        return qualifiers

    def _rebuild_network(self):
        if self.backend == "cupy":
            self._qualifiers = self._build_cupy_qualifiers()
            self.ntwrk = cutn.Network(self.eq, *self.nodes, qualifiers=self._qualifiers)
        else:
            self.ntwrk = cutn.Network(self.eq, *self.nodes)
    
    def contract_ntwrk(self):
        self.ntwrk.contract_path()
        return self.ntwrk.contract()

    def decompose(
        self,
        target,
        tol=None,
        pct_loss_improvment=0.025,
        init_lr=0.05,
        loss_patience=2500,
        lr_patience=250,
        max_epochs=25000,
        momentum=0.5,
    ):
        if self.backend == "torch":
            return self._decompose_torch_sgd(
                target=target,
                tol=tol,
                pct_loss_improvment=pct_loss_improvment,
                init_lr=init_lr,
                loss_patience=loss_patience,
                lr_patience=lr_patience,
                max_epochs=max_epochs,
                momentum=momentum,
            )
        if self.backend == "cupy":
            return self._decompose_cupy_cutn_sgd(
                target=target,
                tol=tol,
                pct_loss_improvment=pct_loss_improvment,
                init_lr=init_lr,
                loss_patience=loss_patience,
                lr_patience=lr_patience,
                max_epochs=max_epochs,
                momentum=momentum,
            )
        raise ValueError(f"Unsupported backend '{self.backend}'.")

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
    ):
        if self.backend != "torch":
            raise NotImplementedError(
                "decompose currently supports only backend='torch' (SGD + autograd)."
            )

        if not self.nodes:
            raise ValueError("Network has no tensors to optimize.")

        params = []
        for core in self.nodes:
            if not isinstance(core, torch.Tensor):
                raise TypeError("All cores must be torch.Tensor for backend='torch'.")
            if not core.requires_grad:
                core.requires_grad_(True)
            params.append(core)

        optimizer = torch.optim.SGD(params, lr=init_lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=1 / torch.e, patience=lr_patience
        )

        ref_tensor = self.nodes[0]
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
            
            contracted_t = cutn.contract(self.eq, *self.nodes, optimize=optimize_cfg)
            loss = torch.norm(target - contracted_t) / target_norm
            
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

        return loss, epoch

    def _decompose_cupy_cutn_sgd(
        self,
        target,
        tol=0.01,
        pct_loss_improvment=0.025,
        init_lr=0.05,
        loss_patience=2500,
        lr_patience=250,
        max_epochs=25000,
        momentum=0.5,
    ):
        if not self.nodes:
            raise ValueError("Network has no tensors to optimize.")
        if self._qualifiers is None:
            raise RuntimeError("CuPy qualifiers were not initialized for gradient computation.")
        if not hasattr(self.ntwrk, "gradients"):
            raise NotImplementedError(
                "This cuTensorNet build does not expose Network.gradients(). "
                "Upgrade cuQuantum to use CuPy + cuTensorNet-native gradients."
            )

        target = cp.asarray(target)
        target = target.astype(self.nodes[0].dtype, copy=False)
        target_norm = cp.linalg.norm(target)

        self.ntwrk.contract_path()
        self.ntwrk.autotune(iterations=5)

        lr = float(init_lr)
        velocity = [cp.zeros_like(node) for node in self.nodes]
        epoch = 0
        wait = 0
        best_loss = float("inf")
        min_delta = 0.0
        loss_value = float("inf")
        bad_lr_steps = 0

        while epoch < max_epochs:
            contracted_t = self.ntwrk.contract()
            residual = contracted_t - target
            loss = cp.linalg.norm(residual) / target_norm
            loss_value = float(loss.item())

            # d/dY ||Y-T||/||T|| = (Y-T) / (||T||*||Y-T||)
            residual_norm = cp.maximum(cp.linalg.norm(residual), cp.finfo(residual.dtype).eps)
            output_grad = residual / (target_norm * residual_norm)

            grads = self.ntwrk.gradients(output_gradient=output_grad)

            if len(grads) != len(self.nodes):
                raise RuntimeError("Gradient count mismatch with number of network cores.")

            for i, grad in enumerate(grads):
                velocity[i] = momentum * velocity[i] + grad
                self.nodes[i] -= lr * velocity[i]

            epoch += 1

            if loss_value < best_loss - min_delta:
                best_loss = loss_value
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
            if tol is not None and loss_value <= tol:
                break

            if epoch % 100 == 0:
                sys.stdout.flush()
                print(f"\rEpoch {epoch}, Loss: {loss_value:0.5f}, Learning Rate: {lr:0.6f}")

        return loss_value, epoch

def einsum_expr(adj_matrix):
    dim = adj_matrix.shape[0]
    labels = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    einsum_map = {}
    input_labels = []
    
    output_labels = []
    for i in range(dim):
        tensor_labels = []
        skip = 0 
        for j in range(dim):
            if adj_matrix[i, j] > 1:  
                if (j, i) in einsum_map:  
                    tensor_labels.append(einsum_map[(j, i)])
                else:
                    label = next(labels)  
                    einsum_map[(i, j)] = label
                    tensor_labels.append(label)
            if i == j:
                output_labels.append(label)  
        input_labels.append(tensor_labels)

    lhs = ",".join(["".join(m) for m in input_labels])
    rhs = "".join(output_labels)
    es_expr = f"{lhs}->{rhs}"
    return es_expr


def sim_tensor_from_adj(A, std_dev=0.1, backend="torch", dtype="float32"):
    A = A.to(dtype=torch.int)
    adj = torch.maximum(A, A.T)
    backend_name = backend.lower()
    dtype_name = dtype.lower()

    if backend_name == "torch":
        torch_dtype = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[dtype_name]
    elif backend_name == "cupy":
        cupy_dtype = {
            "float16": cp.float16,
            "float32": cp.float32,
            "float64": cp.float64,
        }[dtype_name]
    else:
        raise ValueError(f"Unsupported backend '{backend}'. Use 'torch' or 'cupy'.")

    backend_dtype = torch_dtype if backend_name == "torch" else cupy_dtype
    cores = _random_cores_from_adj(adj, std_dev, backend_name, backend_dtype)

    ntwrk = cuTensorNetwork(adj, cores=cores, backend=backend_name, dtype=dtype_name)
    return ntwrk.contract_ntwrk(), cores


if __name__=="__main__":
    torch.manual_seed(1)
    
    N = 5
    max_rank = 7
    A = random_adj_matrix(N, max_rank)
    tgt, cores = sim_tensor_from_adj(A, backend="cupy", dtype="float32")

    ctn = cuTensorNetwork(A, backend="cupy", dtype="float32")
    loss = ctn.decompose(
        tgt, tol=1e-8, init_lr=0.1, loss_patience=2500, max_epochs=5000
    )
    print(loss)
