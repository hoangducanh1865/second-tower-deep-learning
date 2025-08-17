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
    transform = transforms.Compose(trans)

    mnist_train = datasets.FashionMNIST(root="../data", train = True, download = True, transform = transform)
    mnist_test = datasets.FashionMNIST(root="../data", train = False, download = True, transform = transform)

    train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers = get_dataloader_workers())
    test_iter = DataLoader(mnist_test, batch_size, shuffle=False, num_workers = get_dataloader_workers())
    
    return train_iter, test_iter

def accuracy (y_hat, y):
    if y_hat.shape[1] > 1:
        preds = y_hat.argmax(axis =1)
    else:
        preds = y_hat.round().int().squeeze()

    return (preds == y).sum().item()

def evaluate_accuracy (net, test_iter):
    metric = Accumulator (2)
    for X, y in test_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]

class Accumulator():
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float (b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]
    
def train_epoch_ch3(net, train_iter, loss_fn, optimizer, device=None):
    net.train()  
    metric = Accumulator(3)  # [train_loss_sum, train_acc_sum, num_examples]

    for X, y in train_iter:
        if device:
            X, y = X.to(device), y.to(device)

        y_hat = net(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric.add(
            float(loss.sum()),            
            accuracy(y_hat, y),           
            y.size(0)                     
        )
    return metric[0] / metric[2], metric[1] / metric[2]



def train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, optimizer, device = None):
    for epoch in range (num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss_fn, optimizer, device)
        test_acc = evaluate_accuracy(net, test_iter)



