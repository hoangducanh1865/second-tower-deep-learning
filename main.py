from d2l import mxnet as d2l
from mxnet import autograd, np, npx
import random
npx.set_np()

def synthetic_data(w, b, num_examples):
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.001, y.shape)
    return X, y

def iter (batch_size, features, labels):
    num_examples = len(features)
    indices = list (range(num_examples))
    random.shuffle(indices)
    for i in range (0, num_examples, batch_size):
        batch_indices = np.array(indices[i:min(i + batch_size, num_examples)]) 
        yield features[batch_indices], labels[batch_indices]