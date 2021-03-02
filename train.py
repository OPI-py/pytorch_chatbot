import json
import numpy as np
from nltk_func import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralModel


with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = [] #  xy-pair list

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
ignore_words = ['?', '!', '.', ',']
# stemming words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sort, remove duplicates
all_words = sorted(set(all_words))
tag = sorted(set(tags))

# create training data
x_train = []
y_train = []
for sentence, tag in xy:
    bag = bag_of_words(sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag) # index for tags
    y_train.append(label)
    
x_train = np.array(x_train)
y_train = np.array(y_train)

class BotDataset(Dataset):
    '''Pytorch Dataset class'''
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
        
    def __getitem__(self, index):
        '''Access dataset with an index'''
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        '''Return Dataset size'''
        return self.n_samples

# Parameters
batch_size = 12
hidden_size = 12
output_size = len(tags)
input_size = len(x_train[0]) # len of first bow
learning_rate = 0.001
num_epochs = 1000

dataset = BotDataset()
# train_loader to automatically iterate over BotDataset
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,
    shuffle=True, num_workers=0)

# check if GPU Compute Unified Device Architecture available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralModel(input_size, hidden_size, output_size).to(device)

# loss and optimizer
c_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long)
        # forward
        outputs = model(words)
        loss = c_loss(outputs, labels)
        # backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}') 
print(f'Final loss={loss.item():.4f}')

data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags}

data_file = 'data.pth'
torch.save(data, data_file)
print('Done')