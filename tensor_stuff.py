import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# scalar
# vector
# MATRIX
# TENSOR

def skip():

    TENSOR = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    print(TENSOR[0])
    print(TENSOR.ndim)
    print(TENSOR.shape)

    tensor1 = torch.rand(size=(3, 6, 3))
    print(tensor1)

    tensor = torch.zeros(size=(3, 6, 3))
    print(tensor)

    tensor = tensor * tensor1
    print(tensor)
    print(tensor.dtype)

    tensor = torch.ones(size=(3, 6, 3))
    print(tensor)

    tensor = torch.arange(100, 10000, 21)
    print(tensor)

    tensor = torch.zeros_like(tensor)
    print(tensor)

    tensor = torch.rand(size=(2, 6, 3), dtype=torch.float32, device=None, requires_grad=False)
    print(tensor)

    tensor = tensor.type(torch.float16)

    tensor = torch.rand(size=(2, 3, 3), dtype=torch.float32)
    tensor2 = tensor / 10
    print(tensor)

    tensor = torch.mm(tensor2, tensor)

    # mm = matmul
    # matmul: outer dimensions must match

    tensor = torch.rand(size=(2, 3, 3))

    print(tensor.mT)

    # matmul 3,2 and 2,3 do not work
    # transpose needed
    # matmul 3,2 and 2,3 do work

    tensor = torch.rand(size=(2, 3, 3))
    tensor = tensor * 10
    print(tensor.min())
    print(tensor.min())
    print(tensor.max())
    print(tensor.sum())
    print(tensor.dtype)
    print(tensor.type(torch.float32).mean())

    # argmin = index of min

    # create tensor with int64 dtype
    tensor = torch.arange(1, 10)
    # create tensor with float32 dtype

    tensor = torch.arange(1., 10.)

    # reshaped = clone with different shape
    # view = link both objects together (same object in memory)
    tensor = tensor.reshape(1,1,9)
    tensor = tensor.view(1,1,9)

    print(tensor)

    tensor.unsqueeze(dim=0)
    tensor.squeeze(dim=0)
    # dim=0

    # same as view - share memory
    tensor.permute(2, 0, 1)

    a = tensor[:, 0, 1]
    print(a)

    print(a.squeeze())

    # numpy default dtype is float64, pytorch default dtype is float32
    # reflects original datatype for both ways
    arr = np.arange(1.0, 8.0)
    tensor = torch.from_numpy(arr).type(torch.float32)

    arr = tensor.numpy()

    # manual_seed only works once
    random_seed = 42
    torch.manual_seed(random_seed)
    tensor = torch.rand(3, 4)
    torch.manual_seed(42)
    tensor2 = torch.rand(3, 4)
    print(tensor == tensor2)

    # check gpu availability 
    print(torch.cuda.is_available())

    tensor = torch.rand(3, 4, device="cuda:0")

    tensor1 = tensor.cpu()

    arr = tensor1.numpy()

    # tensor on gpu cannot be converted to numpy array

    print(torch.randn(20))



