import torch
import torch.nn as nn
import torch.nn.functional as F
from model_data_loader import load_model_data

class SWEMWithEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_dim, num_outputs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc1 = nn.Linear(embedding_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_outputs)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=0)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
train_loader, test_loader = load_model_data()

#training
model = SWEMWithEmbeddings(#instantiate model
    vocab_size = 5000,
    embedding_size = 300, 
    hidden_dim = 64, 
    num_outputs = 1,
)

#binary cross-entropy (BCE) loss and Adam optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Iterate through train set minibatchs 
for epoch in range(250):
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
        
        predictions = torch.round(torch.sigmoid(y))
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
        
        predictions = torch.round(torch.sigmoid(y))
        correct += torch.sum((predictions == labels).float())
        num_test += len(inputs)
    
print('Test accuracy: {}'.format(correct/num_test))