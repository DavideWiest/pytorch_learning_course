import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path




weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

x = torch.arange(start, end, step).unsqueeze(dim=1)

y = weight * x + bias

train_split = int(0.8 * len(x))

x_train, y_train = x[:train_split], y[:train_split]

x_test, y_test = x[train_split:], y[train_split:]




class LinearRegressionModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear_layer = nn.Linear(in_features=1, out_features=1)