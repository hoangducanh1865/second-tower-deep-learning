import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import random
import matplotlib.pyplot as plt
import sys
# Thiết lập môi trường
torch.manual_seed(42)

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # Cập nhật inplace
            param.grad.zero_()  # Xóa gradient

def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape))**2) /2


def load_array(data_arrays, batch_size, is_train = True): # chia minibatch
    data_set = TensorDataset(*data_arrays)
    return DataLoader(data_set, batch_size, shuffle= is_train)

def get_dataloader_workers(num_workers=4):
    if sys.platform.startswith("win"):
        return 0
    else:
        return num_workers

def load_data_fashion_mnist(batch_size, resize = None):
    trans = []

    if resize:
        trans.append(transforms.Resize(resize))
    trans.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    mnist_train = datasets.FashionMNIST(root="../data", train = True, download = True, transform = transform)
    mnist_test = datasets.FashionMNIST(root="../data", train = False, download = True, transform = transform)

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers = get_dataloader_workers())
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers = get_dataloader_workers())
    
    return train_iter, test_iter
