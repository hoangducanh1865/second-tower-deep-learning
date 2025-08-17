# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from save import function as f

num_inputs = 4096
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here
def net(X):
    X = X.reshape((-1, 64 * 64))  # từ ảnh 64x64 → vector 4096 chiều
    return softmax(torch.matmul(X, W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10
train_iter, test_iter = f.load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
f.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
f.predict_ch3(net, test_iter)
