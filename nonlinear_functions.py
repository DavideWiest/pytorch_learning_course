import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

a = torch.arange(-10, 10, 1, dtype=torch.float32)


plt.plot(a)

def relu(x: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.tensor(0), x)

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1 / (1 + torch.exp(-x))


