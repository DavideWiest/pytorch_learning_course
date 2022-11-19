import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm
import os, sys
from pathlib import Path

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset

import torchinfo
from torchinfo import summary

from typing import Tuple, Dict, List

from PIL import Image

def main():
    BATCH_SIZE = 32
    NUM_WORKERS = os.cpu_count()
    EPOCHS = 10
    CONV_BLOCK2_OUTPUT_SHAPE = 13 # look at shape of label batch 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_train = Path("C:/Users/DavWi/OneDrive/Desktop/storage/ml_datasets/foods/train")
    img_test = Path("C:/Users/DavWi/OneDrive/Desktop/storage/ml_datasets/foods/test")

    img_train_list = list(img_train.glob("*/*.jpg"))
    img_test_list = list(img_test.glob("*/*.jpg"))

    data_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
    ])

    train_transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.TrivialAugmentWide(5),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(root=img_train, transform=train_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=img_test, transform=data_transform, target_transform=None)

    # print(train_data.class_to_idx)
    # print(train_data.classes)

    train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    img, label = next(iter(train_data))







    class_names = train_data.classes

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


    class TinyVGG(nn.Module):
        def __init__(self, input_shape, hidden_units, output_shape):
            super().__init__()

            self.conv_block1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units*CONV_BLOCK2_OUTPUT_SHAPE*CONV_BLOCK2_OUTPUT_SHAPE, out_features=output_shape)
            )

        def forward(self, x):
            # x = self.conv_block1(x.to(device))
            # x = self.conv_block2(x)
            # x = self.classifier(x)
            # return x
            # performance benefits through operator fusion
            return self.classifier(self.conv_block2(self.conv_block1(x.to(device))))

    model = TinyVGG(
        3, # color channels
        10,
        len(class_names)
    ).to(device)


    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


    def getmodel_info():
        summary(model, input_size=[1, 3, 64, 64])
        image_batch, label_batch = next(iter(train_dataloader))
        # image_batch = image_batch.permute
        pred = model(image_batch.to(device))
        print(pred.shape)


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
        
        return train_loss, train_acc

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

        return test_loss, test_acc

    def train_full_fn(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module, accuracy_fn, epochs: int, device):
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device)
            test_loss, test_acc = test_step(model, test_dataloader, loss_fn, optimizer, accuracy_fn, device)
            results["train_loss"].append(float(train_loss))
            results["train_acc"].append(float(train_acc))
            results["test_loss"].append(float(test_loss))
            results["test_acc"].append(float(test_acc))

            print(f"\n\nEpoch {epoch} | train loss: {train_loss} | train acc: {train_acc} | test loss: {test_loss} | test acc: {test_acc} \n")
        
        return results

    results = train_full_fn(model, train_dataloader, test_dataloader, optimizer, loss_fn, accuracy_fn, EPOCHS, device)


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

    def plot_loss_curves(results):
        epochs = list(range(EPOCHS))

        plt.figure(figsize=(15, 7))
        plt.subplot(1,2,1)
        plt.plot(epochs, results["test_loss"], label="test_loss")
        plt.plot(epochs, results["train_loss"], label="train_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(epochs, results["test_acc"], label="test_acc")
        plt.plot(epochs, results["train_acc"], label="train_acc")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.legend()

        plt.show()


    plot_loss_curves(results)

if __name__ == "__main__":
    main()