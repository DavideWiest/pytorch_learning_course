import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sample_num = 1000

x, y = make_circles(sample_num, noise=0.03, random_state=42)

circles = pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "label": y})


# plt.scatter(x = x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=42)






