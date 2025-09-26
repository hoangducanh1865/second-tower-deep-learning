import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8,1)
)

def block1():
    return nn.Sequential(nn.Linear(4, 8),
                          nn.ReLU(),
                          nn.Linear(8,4),
                          nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block{i}', block1())
    return net
def init_normal (m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 1)
       # nn.init.zeros_(m.bias)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >=5

if __name__ == "__main__":

    net.apply(my_init)
    print(net[2].weight.data[0])
    print(net[2].weight.data[0][1])

