import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
import sys
import random
import torchmetrics
import mlxtend
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



BATCH_SIZE = 32
EPOCHS = 3
CONV_BLOCK2_OUTPUT_SHAPE = 7

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

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*CONV_BLOCK2_OUTPUT_SHAPE*CONV_BLOCK2_OUTPUT_SHAPE, out_features=output_shape)
        )


    def forward(self, x):
        x = self.conv_block1(x.to(device))
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

model = visionModel(
    1,
    10,
    len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)




def print_train_time(start, end, device: torch.device = None):
    total_time = end - start
    print(f"Time elapsed on {device}: {total_time:.3f}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

images = torch.randn(size=(32, 3, 64, 64))
test_image = images[0]

# print(test_image.shape)

conv_layer = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3,3), stride=1, padding=0)
max_pool_layer = nn.MaxPool2d(kernel_size=2)

# conv_output = conv_layer(test_image)
# print(conv_output.shape)
# maxpool_output = max_pool_layer(conv_output)
# print(maxpool_output.shape)

def test_model():
    test_image2 = torch.randn(size=[1, 28, 28]).unsqueeze(1).to(device)

    test_image2 = model.conv_block1(test_image2)
    print(test_image2.shape)
    test_image2 = model.conv_block2(test_image2)
    print(test_image2.shape)
    test_image2 = model.classifier(test_image2)
    print(test_image2.shape)



train_time_start = timer()

def train_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):
    model.train()

    train_loss, train_acc = 0,0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        y_pred = model(x)# .squeeze()

        loss = loss_fn(y_pred, y)
        train_loss += loss
        train_acc += accuracy_fn(y, y_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    
    print(f"\nLooked at {batch * len(x)}/{len(dataloader.dataset)} loss {train_loss:.3f} acc {train_acc:.3f}")


def test_step(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_fn, device: torch.device = device):

    model.eval()

    with torch.inference_mode():
        test_loss, test_acc = 0, 0

        for x_test, y_test in dataloader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            test_pred = model(x_test).squeeze().to(device)
            # test_pred = torch.round(torch.sigmoid(test_logits))

            test_loss += loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1))

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    # if epoch % 10 == 0:
    print(f"epoch {epoch} test loss {test_loss:.3f} test acc {test_acc:.3f}")


for epoch in tqdm(range(EPOCHS)):
    train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
    test_step(model, test_dataloader, loss_fn, optimizer, accuracy_fn, device)




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

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []

    model.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in tqdm(data):
            sample = torch.unsqueeze(sample, dim=0).to(device)

            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)


test_samples = []
test_labels = []

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

test_samples[0].shape

def show_first_sample():
    plt.imshow(test_samples[0].squeeze(), cmap="gray")
    plt.title(class_names[test_labels[0]])
    plt.show()

pred_probs = make_predictions(model, test_samples, device)
pred_classes = pred_probs.argmax(dim=1)

def plot_sample_preds():
    plt.figure(figsize=(9,9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(sample.squeeze(), cmap="gray")
        pred_label = class_names[pred_classes[i]]
        truth_label = class_names[test_labels[i]]
        if pred_label == truth_label:
            plt.title(f"{pred_label} [{truth_label}]", fontsize=10, c="g")
        else:
            plt.title(f"{pred_label} [{truth_label}]", fontsize=10, c="r")

    plt.axis(False)
    plt.show()

y_preds = []
model.eval()
with torch.inference_mode():
    for x, y in tqdm(test_dataloader, desc="making predictions"):
        x, y = x.to(device), y.to(device)

        y_logit = model(x)
        y_pred = torch.softmax(y_logit.squeeze(), dim=0).argmax(dim=1)
        y_preds.append(y_pred.cpu())

print(y_preds)
y_pred_tensor = torch.cat(y_preds)
y_pred_tensor[:10]

def show_conf_matrix():
    confmat = ConfusionMatrix(num_classes=len(class_names))
    confmat_tensor = confmat(preds=y_pred_tensor, target=test_data.targets)

    fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.numpy(), class_names=class_names, figsize=(10,7))
    fig.show()
    plt.show()






