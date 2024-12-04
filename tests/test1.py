## Need to start making unit tests

    # torch.manual_seed(1)
    
    # from decomp.tn import TensorNetwork, sim_tensor_from_adj
    # A = random_adj_matrix(4, 6).int()
    
    
    # ctn = cuTensorNetwork(A)
    # x, cores_ = sim_tensor_from_adj(A)
    # cores = [core.reshape(*A[i].int().tolist()).moveaxis(i, 0) for i, core in enumerate(ctn.nodes)]
    # tn = TensorNetwork(A, cores=cores)
    
    # X1 = tn.contract_network()
    # X2 = ctn.contract()
    # print(ctn.eq)
    
    # # this is a test : (X1 - X2).norm < 0