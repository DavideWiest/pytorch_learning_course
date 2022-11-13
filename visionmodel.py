import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor

device = "cuda" if torch.cuda.is_available() else "cpu"

train_data = datasets.FashionMNIST(
    root="data", 
    train=True, 
    download=True, 
    transform=torchvision.transforms.ToTensor(), 
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)


class cnnModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layers = nn.Sequential(

            )

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x))))) # x -> layer1 -> layer2 -> layer3

cm = cnnModel().to(device)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc


















epochs = 100

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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






