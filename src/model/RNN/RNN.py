import math
import torch
from torch import nn
import torch.nn.functional as F
import d2l.torch as d2l


class RNNFromScratch:
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size = vocab_size
        self.num_hiddens = num_hiddens
        self.device = device
        self.params = self._init_params()

    
    def _init_params(self):
        def normal(shape):
            return torch.randn(size=shape, device=self.device) * 0.01

        W_xh = normal((self.vocab_size, self.num_hiddens))
        W_hh = normal((self.num_hiddens, self.num_hiddens))
        b_h = torch.zeros(self.num_hiddens, device=self.device)

        W_hq = normal((self.num_hiddens, self.vocab_size))
        b_q = torch.zeros(self.vocab_size, device=self.device)

        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for p in params:
            p.requires_grad_(True)
        return params

    def begin_state(self, batch_size):
        return (torch.zeros((batch_size, self.num_hiddens), device=self.device),)

    
    def forward(self, X, state):
        # X shape: (batch_size, num_steps)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)  # (num_steps, batch_size, vocab_size)
        W_xh, W_hh, b_h, W_hq, b_q = self.params
        H, = state
        outputs = []
        for x in X:
            H = torch.tanh(torch.mm(x, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)

    
    def grad_clipping(self, theta):
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in self.params))
        if norm > theta:
            for p in self.params:
                p.grad[:] *= theta / norm

    
    def predict(self, prefix, num_preds, vocab):
        state = self.begin_state(batch_size=1)
        outputs = [vocab[prefix[0]]]
        get_input = lambda: torch.tensor([outputs[-1]], device=self.device).reshape((1, 1))

        # warm-up
        for y in prefix[1:]:
            _, state = self.forward(get_input(), state)
            outputs.append(vocab[y])

        # generate
        for _ in range(num_preds):
            y, state = self.forward(get_input(), state)
            outputs.append(int(y.argmax(dim=1).reshape(1)))

        return ''.join([vocab.idx_to_token[i] for i in outputs])

    
    def train_epoch(self, train_iter, loss, updater, use_random_iter):
        state, timer = None, d2l.Timer()
        metric = d2l.Accumulator(2)  # total loss, token count

        for X, Y in train_iter:
            if state is None or use_random_iter:
                state = self.begin_state(batch_size=X.shape[0])
            else:  # detach state
                for s in state:
                    s.detach_()

            y = Y.T.reshape(-1)
            X, y = X.to(self.device), y.to(self.device)
            y_hat, state = self.forward(X, state)
            l = loss(y_hat, y.long()).mean()

            updater.zero_grad()
            l.backward()
            self.grad_clipping(1)
            updater.step()

            metric.add(l * y.numel(), y.numel())

        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    
    def train(self, train_iter, vocab, lr, num_epochs, use_random_iter=False):
        loss = nn.CrossEntropyLoss()
        updater = torch.optim.SGD(self.params, lr=lr)
        

        for epoch in range(num_epochs):
            ppl, speed = self.train_epoch(train_iter, loss, updater, use_random_iter)
            if (epoch + 1) % 10 == 0:
                print(self.predict('time traveller', 50, vocab))
            

        print(f'Final perplexity {ppl:.1f}, {speed:.1f} tokens/sec')
        