import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs 
from sklearn.model_selection import train_test_split

NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
LEARNING_RATE = 0.1
EPOCHS = 1000

device = "cuda" if torch.cuda.is_available() else "cpu"

x_blob, y_blob = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)

x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.float)

x_blob_train, x_blob_test, y_blob_train, y_blob_test = train_test_split(x_blob, y_blob,  test_size=0.2, random_state=RANDOM_SEED)

plt.figure(figsize=(7, 7))
plt.scatter(x_blob[:, 0].squeeze(), x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()


x_blob_train, x_blob_test, y_blob_train, y_blob_test = x_blob_train.to(device), x_blob_test.to(device), y_blob_train.to(device), y_blob_test.to(device)







class blobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


model = blobModel(NUM_FEATURES, NUM_CLASSES, 8)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.paramters(), lr=LEARNING_RATE)

model.eval()

with torch.interence_mode():

    y_logits = model(x_blob_test)
    y_logits = torch.argmax(y_logits, dim=1)
    test_pred = torch.softmax(y_logits, dim=1)

    print(test_pred)





