import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import random
import matplotlib.pyplot as plt

# Thiết lập môi trường
torch.manual_seed(42)

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    return ((y_hat - y.reshape(y_hat.shape))**2) /2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # Cập nhật inplace
            param.grad.zero_()  # Xóa gradient

def load_array(data_arrays, batch_size, is_train = True): # chia minibatch
    data_set = TensorDataset(*data_arrays)
    return DataLoader(data_set, batch_size, shuffle= is_train)
