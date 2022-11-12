import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
LEARNING_RATE = 0.1
EPOCHS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

torchmetric_accuracy = Accuracy().to(device)

x_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob,  test_size=0.2, random_state=RANDOM_SEED)

# plt.figure(figsize=(7, 7))
# plt.scatter(x_blob[:, 0].squeeze(), x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
# plt.show()


x_blob_train, x_blob_test, y_blob_train, y_blob_test = x_blob_train.to(device), x_blob_test.to(device), y_blob_train.to(device), y_blob_test.to(device)




class blobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = blobModel(NUM_FEATURES, NUM_CLASSES, 8).to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

model.eval()

def test_model():
    with torch.inference_mode():

        y_logits = model(x_blob_test)
        y_logits = torch.argmax(y_logits, dim=1)
        test_pred = torch.softmax(y_logits, dim=1)

        print(test_pred)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

for epoch in range(EPOCHS):
    model.train()

    y_logits = model(x_blob_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train)
    acc = torchmetric_accuracy(y_blob_train, y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        y_logits2 = model(x_blob_test)
        y_pred2 = torch.softmax(y_logits2, dim=1).argmax(dim=1)

        test_loss = loss_fn(y_logits2, y_blob_test)
        # test_acc = accuracy_fn(y_blob_test, y_pred2)
        test_acc = torchmetric_accuracy(y_pred2, y_blob_test)

    if epoch % 10 == 0:
        print(f"epoch {epoch} loss {loss:.3f} acc {acc:.3f} test loss {test_loss:.3f} test acc {test_acc:.3f}")


model.eval()

with torch.inference_mode():
    y_logits = model(x_blob_test)
    y_pred2 = torch.softmax(y_logits2, dim=1).argmax(dim=1)

    test_loss = loss_fn(y_logits2, y_blob_test)
    test_acc = accuracy_fn(y_blob_test, y_pred2)

from bourke_helper_functions import plot_predictions, plot_decision_boundary

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, x_blob_train, y_blob_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, x_blob_test, y_blob_test)

plt.show()




