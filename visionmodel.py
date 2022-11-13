import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



BATCH_SIZE = 32


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


class_names = train_data.classes

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

torch.manual_seed(42)

train_features_batch, train_labels_batch = next(iter(train_dataloader))

train_features_batch.to(device)
train_labels_batch.to(device)

# random_idx = torch.randint(0, len(train_features_batch))

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

    

def show_data():
    image, label = train_data[0]
    print(image.shape)
    print(label)
    print(class_names[label])

    torch.manual_seed(42)
    fig = plt.figure(figsize=(9,9))
    rows, cols = 4, 4

    for i in range(1, rows*cols+1):
        random_idx = torch.randint(0, len(train_data), size=[1]).item()
        img, label = train_data[random_idx]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img.squeeze(), modelap=plt.model.RdYlBu)
        plt.title(class_names[label])
        plt.axis(False)

    # plt.imshow(image.squeeze())

    plt.show()


class visionModel(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape):
        super().__init__()

        self.cnn_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units, out_features=output_shape)#,
            # nn.ReLU()
        )

    def forward(self, x):
        return self.cnn_layers(x.to(device))

model = visionModel(
    28 * 28,
    10,
    len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

def print_train_time(start, end, device: torch.device = None):
    total_time = end - start
    print(f"Time elapsed on {device}: {total_time:.3f}")

start_time = timer()

end_time = timer()

print_train_time(start_time, end_time, device)

train_time_start = timer()



epochs = 2

torch.manual_seed(42)
torch.cuda.manual_seed(42)

for epoch in tqdm(range(epochs)):
    model.train()

    train_loss = 0

    for batch, (x, y) in enumerate(train_dataloader):

        y_pred = model(x).squeeze()

        loss = loss_fn(y_pred, y.to(device))
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        if batch % 400 == 0:
            print(f"Looked at {batch * len(x)}/{len(train_dataloader.dataset)}")

    train_loss /= len(train_dataloader)

    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0

        for x_test, y_test in test_dataloader:
            
            test_pred = model(x_test).squeeze().to(device)
            # test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss += loss_fn(test_pred, y_test.to(device))
            test_acc += accuracy_fn(y_test.to(device), test_pred.argmax(dim=1))

        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    # if epoch % 10 == 0:
    print(f"epoch {epoch} loss {loss:.3f} test loss {test_loss:.3f} test acc {test_acc:.3f}")

train_time_end = timer()

total_train_time = print_train_time(train_time_start, train_time_end, device)

torch.manual_seed(42)

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_fn):
    loss, acc, time_elapsed = 0, 0, 0

    model.eval()

    with torch.inference_mode():
        start = timer()
        
        for x, y in data_loader:
            y_pred = model(x)

            loss += loss_fn(y_pred, y.to(device))
            acc += accuracy_fn(y.to(device), y_pred.argmax(dim=1))
        
        end = timer()
        time_elapsed += (end - start)

        loss /= len(data_loader)
        acc /= len(data_loader)
        time_elapsed /= len(data_loader)

    return {
        "model_name": model.__class__.__name__,
        "model_loss": loss.item(),
        "model_acc": acc,
        "model_ttc1": time_elapsed
    }

model_results = eval_model(model, test_dataloader, loss_fn, accuracy_fn)
print(model_results)