import math
import time
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from IPython import display
from d2l import tensorflow as d2l


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def synthetic_data(w, b, num_examples):
    X = tf.random.normal((num_examples, len(w)))
    y = tf.matmul(X, tf.reshape(w, (-1,1))) + b
    y += tf.random.normal(y.shape, stddev=0.01)
    return X, tf.reshape(y, (-1,1))


def linreg(X, w, b):
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param.assign_sub(lr * param.gradient / batch_size)

# ==============================
# Load Fashion-MNIST
# ==============================
def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if isinstance(img, tf.Tensor):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():
    """Use 4 processes to read the data."""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    return d2l.load_data_fashion_mnist(batch_size, resize)


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

class Accumulator:  #@save
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_loss(loss, net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        y_hat = net(X)
        l = loss(y, y_hat)
        metric.add(tf.reduce_sum(l).numpy(), y.shape[0])
    return metric[0] / metric[1]

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)  # correct, total
    for X, y in data_iter:
        y_hat = net(X)
        metric.add(accuracy(y_hat, y), y.shape[0])
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)  # loss, correct, num examples
    for X, y in train_iter:
        with tf.GradientTape() as tape:
            y_hat = net(X)
            l = loss(y, y_hat)
        grads = tape.gradient(l, net.trainable_variables)
        updater.apply_gradients(zip(grads, net.trainable_variables))
        metric.add(float(tf.reduce_sum(l)), accuracy(y_hat, y), y.shape[0])
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(12, 8)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f"epoch {epoch+1}, loss {train_metrics[0]:.3f}, "
              f"train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}")


def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y.numpy())
    preds = d2l.get_fashion_mnist_labels(tf.argmax(net(X), axis=1).numpy())
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()



