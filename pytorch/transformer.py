
import io
import math
import mmap
import time
import random
import argparse
import numpy as np

from operator import itemgetter
from itertools import starmap, repeat

from sklearn.metrics import roc_auc_score, f1_score, fbeta_score

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from torchtext.vocab import build_vocab_from_iterator
from transformers.optimization import get_linear_schedule_with_warmup


def select_threshold(thresholds, labels, preds, metric):
    def _value(threshold):
        return metric(labels, list(map(lambda x: 1 if x > threshold else 0, preds)))

    return thresholds[np.argmax(np.array(list(map(_value, thresholds))))]


def num_prefix_match(word1, word2):
    idx = 0
    letters1 = list(word1)
    letters2 = list(word2)
    while idx < min(len(letters1), len(letters2)):
        if letters1[idx] != letters2[idx]:
            break
        idx += 1
    return idx


def text_pair_tokenizer(max_seq_len):
    def _tokenize(a, b):
        """ Check if the request and annotation differ by a single word. """
        beg = num_prefix_match(a, b)
        end = num_prefix_match(list(reversed(a)), list(reversed(b)))

        end_a = len(a) - end
        end_b = len(b) - end

        prefix = a[:beg].strip().split()
        suffix = a[end_a:].strip().split()

        middle_a = a[beg:end_a].strip().split()
        middle_b = b[beg:end_b].strip().split()

        tokens_a = ['<sos>'] + prefix + middle_a + suffix + ['<eos>']
        tokens_b = ['<sos>'] + prefix + middle_b + suffix + ['<eos>']

        pairwise = list(filter(None, tokens_a)) + ['<sep>'] + list(filter(None, tokens_b))
        padding = list(repeat('<pad>', max(0, max_seq_len - len(pairwise))))
        return pairwise[:max_seq_len] + padding
    return _tokenize


def phoneme_pair_tokenizer(max_seq_len):
    def _tokenize(a, b):
        """ Check if the request and annotation differ by a single word. """
        tokens_a = ['<sos>'] + a.split() + ['<eos>']
        tokens_b = ['<sos>'] + b.split() + ['<eos>']

        pairwise = list(filter(None, tokens_a)) + ['<sep>'] + list(filter(None, tokens_b))
        padding = list(repeat('<pad>', max(0, max_seq_len - len(pairwise))))
        return pairwise[:max_seq_len] + padding
    return _tokenize


def read_examples(input_file, g2p=None):
    """Read a list of `InputExample`s from an input file."""
    def t_or_p(text):
        if g2p is not None:
            if text not in g2p:
                return None
            return g2p[text]
        return text
    def to_example(idx, bytes):
        line = bytes.decode("utf-8")
        if not line:
            return None
        args = line.strip().split('\t')
        if len(args) == 3:
            return idx, t_or_p(args[0]), t_or_p(args[1]), args[2]
        return idx, t_or_p(args[1]), t_or_p(args[2]), args[3]
    with open(input_file, mode="r", encoding="utf8") as file_obj:
        map_file = mmap.mmap(file_obj.fileno(), 0, prot=mmap.PROT_READ)
        examples = list(filter(lambda x: None not in x, starmap(to_example, enumerate(iter(map_file.readline, b"")))))
    np.random.shuffle(examples)
    return examples


def to_tensor(examples, vocab, tokenizer):
    def _to_index(tokens):
        return list(map(vocab.__getitem__, tokens))

    numeric = list(map(_to_index, starmap(tokenizer, examples)))
    return torch.tensor(numeric, dtype=torch.long)


def to_dataloader(examples, vocab, tokenizer, max_seq_len, batch_size=32):
    label = torch.tensor(list(map(float, map(itemgetter(3), examples))), dtype=torch.long)
    tokens = to_tensor(list(map(itemgetter(1, 2), examples)), vocab, tokenizer)

    src_mask = src_key_mask(tokens, len(examples), max_seq_len, vocab['<pad>'])
    dataset = TensorDataset(tokens, src_mask, label)
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, shuffle=False)


def src_key_mask(tokens, batch, max_seq_len, pad_idx):
    indices = torch.where(tokens == pad_idx)
    mask = torch.zeros((batch, max_seq_len), dtype=torch.bool)
    mask[indices] = True
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, vocab_size=5000, dropout=0.1, batch_size=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nhead=8,
        nhid=200,
        num_layers=2,
        dropout=0.1,
        classifier_dropout=0.1,
        max_len=256,
    ):

        super().__init__()

        self.d_model = d_model
        assert (
            self.d_model % nhead == 0
        ), "nheads must divide evenly into d_model"

        self.emb = nn.Embedding(vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(
            self.d_model, dropout=dropout, vocab_size=max_len
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=nhid, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 2),
        )

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        self.emb.weight.data.uniform_(-0.1, 0.1)

    def forward(self, tokens, src_mask, mask):
        x = self.emb(tokens) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x.permute(1, 0, 2), mask=src_mask, src_key_padding_mask=mask).permute(1, 0, 2)
        x = x.mean(dim=1)  # could use all layers:  x = x.reshape(x.shape[1], -1), or last layer: x[:, -1, :]
        return self.classifier(x)



if __name__ == '__main__':
    """ /apollo/env/KevquinnSandbox/bin/python transformer.py """
    seed_val = 123

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cl-config', type=str)
    parser.add_argument('--ld-config', type=str)
    parser.add_argument('--ph-config', type=str)
    args = parser.parse_args()

    t0 = time.time()
    epochs = 50
    learning_rate = 0.0001
    batch_size = 32
    max_seq_len = 40
    d_model = 320

    graph_to_phone = {}
    with open('/home/ec2-user/map.csv', 'r') as reader:
        for line in reader.readlines():
            if not line or not line.strip():
                continue
            tokens = line.strip().split('\t')
            if len(tokens) != 2:
                continue
            graph_to_phone[tokens[0]] = tokens[1]

    tokenizer = phoneme_pair_tokenizer(max_seq_len)

    trn_examples = read_examples('/home/ec2-user/trn.csv', g2p=graph_to_phone)
    val_examples = read_examples('/home/ec2-user/val.csv', g2p=graph_to_phone)
    tst_examples = read_examples('/home/ec2-user/tst.csv', g2p=graph_to_phone)

    print(len(trn_examples), len(val_examples), len(tst_examples))

    trn_tokens = list(starmap(tokenizer, map(itemgetter(1, 2), trn_examples)))

    vocab = build_vocab_from_iterator(trn_tokens)
    print('vocab size', len(vocab))
    print(vocab.unk_index)

    trn_dataloader = to_dataloader(trn_examples, vocab, tokenizer, max_seq_len, batch_size=batch_size)
    val_dataloader = to_dataloader(val_examples, vocab, tokenizer, max_seq_len, batch_size=batch_size)
    tst_dataloader = to_dataloader(tst_examples, vocab, tokenizer, max_seq_len, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(len(vocab), d_model, max_len=max_seq_len).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3.0, gamma=0.1)

    num_training_steps = len(trn_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    sig = nn.Sigmoid()

    print('    epochs: {0}, learning_rate: {1}, batch_size: {2}'.format(epochs, learning_rate, batch_size))
    model.train()

    def _to_string(tokens):
        pad_idx = vocab.stoi['<pad>']
        tokens_first_pad_idx = np.where(tokens == pad_idx)[0].tolist()
        end_a = len(tokens)-1 if len(tokens_first_pad_idx) == 0 else tokens_first_pad_idx[0]
        return ' '.join(list(map(vocab.itos.__getitem__, tokens[0:end_a])))

    best_val_loss = 30
    best_val_fscore = 0.1
    best_val_threshold = 0.5
    for epoch in range(int(epochs)):
        n_total = 0
        total_trn_loss = 0
        src_mask = model.generate_square_subsequent_mask(max_seq_len).to(device)
        for step, batch in enumerate(trn_dataloader):
            batch = tuple(t.to(device) for t in batch)
            tokens, mask, label = batch
   
            if tokens.size(0) != batch_size:
                continue
   
            n_total += tokens.size(0)
   
            optimizer.zero_grad()
            outputs = model(tokens, src_mask, mask)
            loss = criterion(outputs, label)
            total_trn_loss += loss.item() * tokens.size(0)
            avg_trn_loss = total_trn_loss / n_total
   
            if step % 2000 == 0:
                total_val_loss = 0
                p=[]
                l=[]
                for val_step, val_batch in enumerate(val_dataloader):
                    val_batch = tuple(t.to(device) for t in val_batch)
                    val_tokens, val_mask, val_label = val_batch
   
                    with torch.no_grad():
                        val_outputs = model(val_tokens, src_mask, val_mask)
                        val_loss = criterion(val_outputs, val_label)
   
                    p.extend(sig(val_outputs[:, 1]).detach().cpu().numpy())
                    l.extend(val_label.to('cpu').numpy().tolist())
   
                    total_val_loss += val_loss.item()
   
                threshold = select_threshold(np.arange(0.0, 1.0, 0.05), l, p, f1_score)
                p_binary = list(map(lambda x: 0 if x < threshold else 1, p))
   
                val_fscore = fbeta_score(l, p_binary, beta=1)
                if val_fscore > best_val_fscore:
                    best_val_threshold = threshold
                    best_val_fscore = val_fscore
                    torch.save(model.state_dict(), '/home/ec2-user/transformer/best_fscore.bin')
   
                val_auc = roc_auc_score(l, p)
                avg_val_loss = total_val_loss / len(val_dataloader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), '/home/ec2-user/transformer/best_loss.bin')
   
                print('    epoch: {0}, trn loss: {1:.5f}, val loss: {2:.5f}, val_auc: {3:.5f}, val_fscore: {4:.5f}, lr: {5:.7f}'.format(epoch, avg_trn_loss, avg_val_loss, val_auc, val_fscore, optimizer.param_groups[0]['lr']))
   
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()

    model.load_state_dict(torch.load('/home/ec2-user/transformer/best_fscore.bin'))
    model.eval()


