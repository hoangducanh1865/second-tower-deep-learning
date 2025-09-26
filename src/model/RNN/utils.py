# utils.py
import collections
import re
import requests
import torch
from torch.utils.data import TensorDataset, DataLoader


class Vocab:
    """Vocabulary for text tokens."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        tokens = tokens or []
        reserved_tokens = reserved_tokens or []

        counter = self._count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)


        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(t) for t in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]

    @property
    def unk(self):
        return 0  

    @property
    def token_freqs(self):
        return self._token_freqs

    @staticmethod
    def _count_corpus(tokens):
        """Count token frequencies."""
        if len(tokens) == 0:
            return collections.Counter()
        if isinstance(tokens[0], list):  # flatten list of lists
            tokens = [t for line in tokens for t in line]
        return collections.Counter(tokens)



class TextDataset:
    def __init__(self, path='https://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt',
                 batch_size=32, num_steps=35, num_train=10000, num_val=5000,
                 token='char'):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val

        raw_text = self._load_text(path)
        tokens = self._tokenize(self._preprocess(raw_text), token=token)
        self.vocab = Vocab(tokens)

        corpus = [self.vocab[tok] for line in tokens for tok in line]

        array = torch.tensor([corpus[i:i+num_steps+1]
                              for i in range(len(corpus) - num_steps)])
        self.X, self.Y = array[:, :-1], array[:, 1:]

    def _load_text(self, path):
        if path.startswith('http'):
            response = requests.get(path)
            response.raise_for_status()
            return response.text
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text, token='char'):
        lines = text.splitlines()
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            raise ValueError("token must be 'word' or 'char'")

    def get_dataloader(self, train=True):
        """Return DataLoader for train/val split."""
        idx = slice(0, self.num_train) if train else slice(self.num_train, self.num_train+self.num_val)
        dataset = TensorDataset(self.X[idx], self.Y[idx])
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=train, drop_last=True)
