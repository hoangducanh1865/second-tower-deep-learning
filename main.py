import torch
from torch import nn
from d2l import torch as d2l
import random

# Tạo dữ liệu giả lập tuyến tính y = 2x1 - 3.4x2 + 4.2 + noise
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

# Hàm sinh mini-batch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# Hàm mô hình tuyến tính
def linreg(X, w, b):
    return torch.matmul(X, w) + b

# Hàm tổn thất bình phương
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# Tối ưu SGD
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# Khởi tạo mô hình và tham số
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# Huấn luyện
lr = 0.03
num_epochs = 3
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)  # l shape: (batch_size, 1)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):.6f}')

print(f'Estimated w: {w.reshape(true_w.shape)}')
print(f'Estimated b: {b}')
print(f'Error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'Error in estimating b: {true_b - b}')
