import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import io
import sys
import csv
csv.field_size_limit(100000000)

import math
import random
import numpy as np
from collections import Counter, defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
Hyperparameters
"""
num_epochs = 1
batch_size = 420
learning_rate = 0.1

negative_examples = 4
layer_one_size = 300
layer_two_size = 300

def cosine_similarity(v1, v2):
    cs = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if math.isnan(cs):
        return 0.0
    return cs

def word2tri(word, letter_tri_mapping):
    w = "#" + word + "#"
    for i in range(0, len(w)-3+1):
        yield letter_tri_mapping[w[i:i+3]]

def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')

def unicode_csv_reader(unicode_csv_data):
    # csv.py doesn't do Unicode; encode temporarily as UTF-8:
    csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), dialect=csv.excel)

    for row in csv_reader:
        # decode UTF-8 back to Unicode, cell by cell:
        yield [unicode(cell, 'utf-8') for cell in row]

def csv_iter_file(inf,**kwargs):
    reader = unicode_csv_reader(inf,**kwargs)
    headers = ['u1', 'u2']
    column = {}
    for h in enumerate(headers):
        column[h[1]] =h[0]

    for item in reader:
        d=dict()
        for field in headers:
            d[field]=item[column[field]]
        yield d

class DssmDataset(Dataset):

    def __init__(self, path, transform=None):
        self.items = []
        self.transform = transform

        column = {}
        headers = ['u1', 'u2']
        for h in enumerate(headers):
            column[h[1]] = h[0]

        with io.open(path, mode='r', encoding="utf-8") as handle:
          for item in handle:
              values = item.split('\t')
              d = dict()
              for field in headers:
                  d[field]=values[column[field]].strip()
              self.items.append(d)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.items[idx])
        return self.items[idx]

class LetterNgrams(object):

    def __init__(self, vocab_size, letter_tri_mapping):
        self.vocab_size = vocab_size
        self.letter_tri_mapping = letter_tri_mapping

    def __call__(self, sample):
        u1, u2 = sample['u1'], sample['u2']
        w1 = set(u1.split(' '))
        w2 = set(u2.split(' '))

        x1 = [0.0] * self.vocab_size
        for w in w1:
            for i in word2tri(w, self.letter_tri_mapping):
                x1[i] = 1.0

        x2 = [0.0] * self.vocab_size
        for w in w2:
            for i in word2tri(w, self.letter_tri_mapping):
                x2[i] = 1.0

        return {'u1': u1, 'u2': u2, 'u1_vec': np.array(x1), 'u2_vec': np.array(x2)}

# Fully connected neural network with one hidden layer
class DSSM(nn.Module):
    def __init__(self, input_size, layer_one_size, layer_two_size):
        super(DSSM, self).__init__()
        self.layer_one = nn.Linear(input_size, layer_one_size)
        #self.layer_one.bias.data.normal_(-0.02, 0.02)
        #self.layer_one.weight.data.normal_(-0.02, 0.02)

        self.layer_two = nn.Linear(layer_one_size, layer_two_size) 
        #self.layer_two.bias.data.normal_(-0.02, 0.02)
        #self.layer_two.weight.data.normal_(-0.02, 0.02)

        self.relu_one = nn.ReLU()
        self.relu_two = nn.ReLU()
    
    def forward(self, x):
        out = self.layer_one(x)
        out = self.relu_one(out)
        out = self.layer_two(out)
        out = self.relu_two(out)
        return out

def forward_dssm(u1, u2):
    u1_norm = torch.sqrt(torch.sum(u1**2, dim=1).unsqueeze(dim=1)).repeat(negative_examples+1, 1) # tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    u2_norm = torch.sqrt(torch.sum(u2**2, dim=1).unsqueeze(dim=1))

    product = torch.sum(u1.repeat(negative_examples+1, 1) * u2, dim=1).unsqueeze(dim=1) # prod = tf.reduce_sum(tf.multiply(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_product = u1_norm * u1_norm

    cos_sim_raw = torch.div(product, norm_product)    
    cos_sim = torch.t(torch.t(cos_sim_raw).view(negative_examples + 1, batch_size)) * 20 # cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

    softmax = nn.Softmax(dim=1)
    prob = softmax(cos_sim)
    hit_prob = prob[:, 0]

    loss = -torch.sum(torch.log(hit_prob)) / batch_size
    return loss

# hovercat bdl s3://bluetrain-datasets-live/kevquinn/scratch/letter_ngram_counts/part-00000-c616b50d-c2cd-482a-841b-02d92b60c607-c000.csv > /tmp/letter_ngrams.csv

"""
Load the Letter Ngrams
"""
vocab_size = 0
letter_tri_mapping = defaultdict(lambda: 0)
with open('/tmp/letter_ngrams.csv') as letter_trigrams:
    for i, pair in enumerate(letter_trigrams):
        word_count = pair.strip().split("\t")
        if int(word_count[1]) > 5:
            letter_tri_mapping[word_count[0]] = vocab_size
            vocab_size += 1

print ("letter_ngrams: ", vocab_size)

trn_pairs = DssmDataset(path='/tmp/trn_1mill.csv', transform=transforms.Compose([LetterNgrams(vocab_size, letter_tri_mapping)]))
tst_pairs = DssmDataset(path='/tmp/tst_1thou.csv', transform=transforms.Compose([LetterNgrams(vocab_size, letter_tri_mapping)]))

trn_dataloader = DataLoader(trn_pairs, batch_size=batch_size, shuffle=True, num_workers=1)
tst_dataloader = DataLoader(tst_pairs, batch_size=batch_size, shuffle=True, num_workers=1)

model = DSSM(vocab_size, layer_one_size, layer_two_size).to(device)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)

print ("Training model...")
total_step = len(trn_dataloader)
for epoch in range(num_epochs):
    for i_batch, sample_batched in enumerate(trn_dataloader):
        u1v, u2v = sample_batched['u1_vec'], sample_batched['u2_vec']

        if len(u1v) != batch_size:
            print ("Truncating. ", len(u1v))
            continue

        # Forward pass
        outputs1 = model(u1v.float().to(device)) # 420 x 300
        outputs2 = model(u2v.float().to(device)) # 420 x 300

        # Add negative instances
        tmp = outputs2.clone() 
        for i in range(negative_examples):
            rand = int((random.random() + i) * batch_size / negative_examples)
            if rand == 0:
                rand += 1

            negative_sample1 = tmp.narrow(0, rand, batch_size - rand)  
            negative_sample2 = tmp.narrow(0, 0, rand)
            outputs2 = torch.cat((outputs2, negative_sample1, negative_sample2), 0)

        # Calculate the loss
        loss = forward_dssm(outputs1, outputs2)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i_batch+1) % 100 == 0:
            gradients = []
            for param in model.parameters():
                gradients.append(param.grad.data.sum())

            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                 .format(epoch+1, num_epochs, i_batch+1, total_step, loss.item()))

# Test the model
with torch.no_grad():
    for i, sample_batched in enumerate(tst_dataloader): 
        u1, u2 = sample_batched['u1'], sample_batched['u2']
        u1v, u2v = sample_batched['u1_vec'], sample_batched['u2_vec']

        # Forward pass
        outputs1 = model(u1v.float().to(device)) # 420 x 300
        outputs2 = model(u2v.float().to(device)) # 420 x 300

        out_mat1 = outputs1.data.cpu().numpy()
        out_mat2 = outputs2.data.cpu().numpy()

        for j in range(len(u1)):
            u1_data = u1[j]
            u2_data = u2[j]
            cos = cosine_similarity(out_mat1[j], out_mat2[j])
            print (i, ",", j, u1_data, " - ", u2_data, ": ", cos)

# Save the model checkpoint
torch.save(model.state_dict(), 'dssm_model_v0.ckpt')
