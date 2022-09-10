import numpy as np
import torch
import torch.nn.functional as F

path1 = "../experiments/debug/models/net_0_val_feats.npy"
path2 = "../experiments/debug/models/net_0_test_feats.npy"
npy1 = np.load(path1)[:100]
npy2 = np.load(path2)[:100]
tensor1 = torch.from_numpy(npy1)
tensor2 = torch.from_numpy(npy2)
tensor1 = F.normalize(tensor1, dim=1)
tensor2 = F.normalize(tensor2, dim=1)
s = (tensor1 * tensor2).sum(dim=1)
print(s)
print(tensor1.shape, tensor1[:5, :5])
print(tensor2.shape, tensor2[:5, :5])
