import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

#tokenizer
tokenizer = get_tokenizer('basic_english')

#build vocab
def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter = AG_NEWS(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])

#recreate iterators
train_iter = list(AG_NEWS(split='train'))
test_iter = list(AG_NEWS(split='test'))

#collator
def collator(batch):
    labels = torch.tensor([label - 1 for (label, text) in batch], dtype=torch.long)
    token_ids = [torch.tensor(vocab(tokenizer(text)), dtype=torch.long) for (_, text) in batch]
    padded = pad_sequence(token_ids, batch_first=True, padding_value=vocab["<pad>"])
    return padded, labels

#dataloaders
BATCH_SIZE = 128
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_iter, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

VOCAB_SIZE = len(vocab)
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_OUTPUTS = 4
NUM_EPOCHS = 3


class SWEM(nn.Module):
    def __init__(self, vocab_size, embedding_dimensions, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dimensions)
        
        self.fc1 = nn.Linear(embedding_dimensions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        embed = self.embedding(x)
        embed_mean = torch.mean(embed, dim=1)
        
        h = self.fc1(embed_mean)
        h = F.relu(h)
        h = self.fc2(h)
        return h

#training
model = SWEM(#instantiate model
    vocab_size = VOCAB_SIZE,
    embedding_dimensions = EMBED_DIM, 
    hidden_dim = HIDDEN_DIM, 
    num_outputs = NUM_OUTPUTS,
)

#binary cross-entropy (BCE) loss and Adam optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(NUM_EPOCHS):
    correct = 0
    num_examples = 0
    for inputs, labels in train_loader:
        #zero out the gradients
        optimizer.zero_grad()
        
        #forward pass
        y = model(inputs)
        loss = criterion(y, labels)
        
        #backward pass
        loss.backward()
        optimizer.step()
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_examples += len(inputs)
    
    #print training progress
    if epoch % 25 == 0:
        acc = correct/num_examples
        print("Epoch: {0} \t Train Loss: {1} \t Train Acc: {2}".format(epoch, loss, acc))

##testing
correct = 0
num_test = 0

with torch.no_grad():
    #iterate through test set minibatchs 
    for inputs, labels in test_loader:
        #forward pass
        y = model(inputs)
        
        predictions = torch.argmax(y, dim=1)
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))