import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


device = "cuda" if torch.cuda.is_available() else "cpu"



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

# device agnostic code for data
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)



class LinearRegressionModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # linear layer, probing layer, fully connected layer, dense layer
        self.linear_layer = nn.Linear(in_features=1, out_features=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

model1 = LinearRegressionModel2()
model1.to(device)



loss_fn = nn.L1Loss() # same as MAE
optimizer = torch.optim.SGD(params=model1.parameters(), lr=0.01)

torch.manual_seed(42)
epochs = 200

for epoch in range(epochs):
    model1.train()

    y_pred=model1(x_train)
    
    loss = loss_fn(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model1.eval()

    with torch.inference_mode():
        test_pred = model1(x_test)
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 10 == 0:
        print(f"Epoch {epoch} loss {loss} test loss {test_loss}")

model_path = Path("models")
model_name = "model2.pth"
model_path_final = model_path / model_name

torch.save(model1.state_dict(), model_path_final)




