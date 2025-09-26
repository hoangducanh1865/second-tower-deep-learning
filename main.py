import torch
from src.model.RNN.utils import TextDataset
from src.model.RNN.RNN import RNNFromScratch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = TextDataset(batch_size=32, num_steps=35, token='char')
train_iter = data.get_dataloader(train=True)
vocab = data.vocab


print(vocab["d"])
# print(type(len(vocab)))
num_hiddens = 256
model = RNNFromScratch(vocab_size=len(vocab), num_hiddens=num_hiddens, device=device)
model.train(train_iter, vocab, lr=1, num_epochs=50)

print(model.predict('time traveller', 50, vocab))
