import json 
import numpy as np 
from nltk_util import tokenization,stemm,bag_of_word
import sys
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('bibleversa.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for patter in intent['patterns']:
        w = tokenization(patter)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?','!','.',',']
all_words = [stemm(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence,tag) in xy:
    bag = bag_of_word(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)


X_train = np.array(X_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples

#hyperparameter
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size,shuffle=True,num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)

# loss and optimizer 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)


for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)


        output = model(words)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'epoch[{epoch+1}/{num_epochs}], Loss:{loss.item():.4f}')



print(f'Final Loss:{loss.item():.4f}')


data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words':all_words,
    'tags': tags
}

FILE = 'bible2.pth'
torch.save(data, FILE)

print(f'training done file is saved in {FILE}')

