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

    plt.figure(figsize=(train_data.max() * 4, train_labels.max() * 4))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions != None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.show()

def plot_pred2(train_data, train_labels, test_data, test_labels, predictions=None):
    train_data, train_labels = train_data.type(torch.float32), train_labels.type(torch.float32)

    plt.figure(figsize=(train_data.max() * 4, train_labels.max() * 4))

    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions != None:
        plt.scatter(train_data, predictions, c="r", s=4, label="Predictions")

    plt.show()

# plot_pred(x_train, y_train, x_test, y_test)

class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

torch.manual_seed(42)

model_0 = LinearRegressionModel()

# print(list(model_0.parameters()))
# print(model_0.state_dict())


# inference mode disables keeping track of data, improving performance
# similar to with torch.no_grad() but inference mode is preferred
with torch.inference_mode():
    y_preds = model_0(x_test)

# print(y_preds)

# plot_pred(x_train, y_train, x_test, y_test, predictions=y_preds)

# loss function to indicatie how accurate your model is
# loss/criterion/cost function

loss_fn = nn.L1Loss()

# lr = learning rate
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

epochs = 360

for epoch in range(epochs):
    model_0.train()

    y_pred = model_0(x_train)

    loss = loss_fn(y_pred, y_train)
    print(f"loss: {loss:.3f}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        y_preds_new = model_0(x_test)
        test_loss = loss_fn(y_preds_new, y_test)
        print(f"Epoch {epoch} loss {loss} test loss {test_loss}")

    print(model_0.state_dict())

    if epoch % 20 == 0:
        with torch.inference_mode():
            y_preds_new = model_0(x_test)
            plot_pred(x_train, y_train, x_test, y_test, predictions=y_preds_new)
            y_preds_new = model_0(x_train)
            plot_pred2(x_train, y_train, x_test, y_test, predictions=y_preds_new)




# after 360 epochs:
# OrderedDict([('weights', tensor([0.6990])), ('bias', tensor([0.3093]))])