import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

def plot_pred(train_data, train_labels, test_data, test_labels, predictions=None):
    train_data, train_labels = train_data.type(torch.float32), train_labels.type(torch.float32)

    plt.figure(figsize=(train_data.max(), train_labels.max()))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions != None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.show()

# plot_pred(x_train, y_train, x_test, y_test)

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias





