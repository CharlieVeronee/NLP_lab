import torch
import torch.nn as nn
import torch.nn.functional as F
from model_data_loader import load_model_data

class SWEM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(300, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
train_loader, test_loader = load_model_data()

#training
model = SWEM()#instantiate model

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