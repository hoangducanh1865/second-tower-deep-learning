# `ToTensor` converts the image data from PIL type to 32-bit floating point
# tensors. It divides all numbers by 255 so that all pixel values are between
# 0 and 1
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from d2l import function as f

d2l.use_svg_display()
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)

X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
f.show_images(X.reshape(18, 28, 28), 2, 9, titles=f.get_fashion_mnist_labels(y))