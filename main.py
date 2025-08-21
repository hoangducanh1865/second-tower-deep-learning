import torch
from torch import nn
from save import function as f
num_inputs, num_hiddens, num_outputs = 28, 256, 10

net = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                    nn.ReLU(),
                    nn.Linear(num_hiddens, num_outputs))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std =0.01)

net.apply(init_weights)
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction = "none")
updater = torch.optim.SGD(net.parameters(), lr = lr)
train_iter, test_iter = f.load_data_fashion_mnist(batch_size= batch_size)
f.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)