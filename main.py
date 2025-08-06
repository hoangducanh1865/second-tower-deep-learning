import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from d2l import synthetic_data, linreg, squared_loss, sgd
torch.manual_seed(42)



# initialize parameters
true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.03
num_epochs = 3
batch_size = 10

features, labels = synthetic_data(true_w, true_b, 1000)




# init minibatch
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# Training
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = squared_loss(linreg(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {train_l.mean():.6f}')

# In kết quả
print(f'Estimated w: {w.reshape(-1).detach()}')
print(f'Estimated b: {b.item()}')
