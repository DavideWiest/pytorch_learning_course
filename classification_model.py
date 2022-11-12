import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

sample_num = 1000

x, y = make_circles(sample_num, noise=0.03, random_state=42)

circles = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": y})


# plt.scatter(x = x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

x = torch.from_numpy(x).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

# x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
test_sample = round(len(x) * 0.8)
x_train = x[:test_sample]
y_train = y[:test_sample]
x_test = x[test_sample:]
y_test = y[test_sample:]

class circleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Linear(in_features=2, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer2(self.layer1(x)) # x -> layer1 -> layer2

cm = circleModel().to(device)

# alternative model initialization
cm = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

# with torch.inference_mode():
#     untrained_preds = cm(x_train.to(device))
#     print(untrained_preds)

loss_fn = nn.BCEWithLogitsLoss() # sigmoid activation function built in
# normal BCE with loss requires inputs to have gone through sigmoid activation function prior
# bce with logits -> pass in logits
# bce -> convert logits to probabilities and pass in probabilities

optimizer = torch.optim.SGD(params=cm.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def test_model():
    cm.eval()

    with torch.inference_mode():
        y_logits = cm(x_test.to(device))[:10]

    y_pred_probs = torch.sigmoid(y_logits)
    print(y_pred_probs)
    y_preds = torch.round(y_pred_probs)
    print(y_preds)

    y_pred_labels = torch.round(torch.sigmoid(cm(x_test.to(device))[:10]))
    print(torch.eq(y_test[:10].squeeze(), y_pred_labels.squeeze()))


# raw output = logits
# logits to prediction probabilities (via activation function)
# prbablities to labels (via torch.round)



epochs = 100

for epoch in range(epochs):
    cm.train()

    y_logits = cm(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)
    acc = accuracy_fn(y_train, y_pred)

    # print(f"loss: {loss:.3f}")
    # print(f"acc: {acc:.3f}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    cm.eval()

    with torch.inference_mode():
        test_logits = cm(x_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_test, test_pred)

    if epoch % 10 == 0:
        print(f"epoch {epoch} loss {loss:.3f} acc {acc:.3f} test loss {test_loss:.3f} test acc {test_acc:.3f}")


# import requests
# from pathlib import Path 

# # Download helper functions from Learn PyTorch repo (if not already downloaded)
# if Path("helper_functions.py").is_file():
#   print("helper_functions.py already exists, skipping download")
# else:
#   print("Downloading helper_functions.py")
#   request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
#   with open("helper_functions.py", "wb") as f:
#     f.write(request.content)

from bourke_helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(cm, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(cm, x_test, y_test)


# output of model are raw logits
# need to be passed into activation function (sigmoid for binary corssentropy, softmax for multiclass classification)
# then: convert the models prediction probabilities to prediction labels by rounding them (binary) or by taking argmax (multiclass)


