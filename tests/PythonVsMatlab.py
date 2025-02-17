A = torch.tensor([
    [60, 2, 2, 2],
    [2, 60, 2, 2],
    [2, 2, 20, 2],
    [2, 2, 2, 20]
])

N = len(A)
# X, _ = sim_tensor_from_adj(A)
# X = X.to(torch.float32)
# G = [torch.rand(A[i].tolist()) for i in range(N)]
# savemat(os.path.expanduser('tensors.mat'), {'X': X, "G": G})
data = loadmat('tensors.mat')
X = torch.tensor(data["X"])
G = [torch.tensor(data['G'][0][i]) for i in range(len(data['G'][0]))]
rho = 0.1

for k in range(N):
    Xk = unfold(X, k)
    Gk = unfold(G[k], k)
    Mk = fctn_comp_partial(G, skip=k)
    Mk = gen_unfold(Mk, N, k)
    
    tempC = Xk @ Mk.t() + rho * Gk
    tempA = Mk @ Mk.t() + rho * torch.eye(Gk.shape[1])
    temp = tempC @ torch.linalg.pinv(tempA)
    G[k] = fold(temp, k, A[k])
    

X_out = fctn_comp(G)
# data_m = loadmat('tensor.mat')
# G_m = [torch.tensor(data_m['G'][0][i]) for i in range(len(data_m['G'][0]))]
X_m = loadmat("tensor.mat")["X_out"]
assert ((X_out - X_m).pow(2).mean().sqrt() < 1e-8)