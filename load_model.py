import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

model_path = Path("models")
model_name = "model.pth"
model_path_final = model_path / model_name

# state dict saved only -> need to make new model and load state dict to it
model_0 = LinearRegressionModel()
model_0.load_state_dict(torch.load(model_path_final))

print(model_0.state_dict())








