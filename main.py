from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()
# maybe save in d2l package later
def synthetic_data(w, b, num_examples):
    X = np.random_normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random_normal(0, 0.001, y.shape)
    return X, y

def data_iter (batch_size, features, labels):
    num_examples = len(features)
    indices = list (range(num_examples))
    random.shuffle(indices)
    for i in range (0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)]) 
        yield features[batch_indices], labels[batch_indices] 

def linreg(X, w, b):
    return np.dot(X,w) + b

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2/2

def sgd (params, lr, batch_size):
    for param in params:
        param[:] = param - lr*param.grad / batch_size





# Init

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
w = np.random_normal(0, 0.01, (2,1))
b = np.zeros(1)
batch_size = 10

w.attach_grad()
b.attach_grad()


lr = 0.03 
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range (num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w,b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %.4f' % (epoch + 1, train_l.mean().asnumpy))  

print("Error in estimating w:", true_w - w.reshape(true_w.shape))
print("Error in estimating b:", true_b - b.reshape(true_b.shape))