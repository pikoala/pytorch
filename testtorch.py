# from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)

uniform_tensor = torch.Tensor(2, 3).uniform_(-1, 1)
print(uniform_tensor)

reshape_tensor = torch.Tensor([[1, 2], [3, 4]])
print(reshape_tensor)

print(x.type())

print(x.dim())

print(x.shape)

print(x.size())

print(x.view(3, 5))
