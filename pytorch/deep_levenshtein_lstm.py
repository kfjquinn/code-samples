
"""
https://medium.com/@adam.wearne/seq2seq-with-pytorch-46dc00ff5164
"""
# These are the standard torch imports

import json
import re
import Levenshtein
import argparse
import os
import time
import datetime
# These are the standard torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import random
import mmap

import numpy as np

from itertools import repeat, zip_longest, chain, starmap
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


def char_tokenizer(utterance):
    return list(utterance.lower())


def format_time(elapsed):
    """ Takes a time in seconds and returns a string hh:mm:ss """
    return str(datetime.timedelta(seconds=int(round(elapsed))))


def normalized_levenshtein(seq1, seq2):
    """ Normalize the levenshtein distance based on the maximum length between each sequence """
    max_len = max(len(seq1), len(seq2))
    return float(max_len - Levenshtein.distance(seq1, seq2)) / float(max_len)


class LevenshteinCosineLoss(nn.Module):
    def __init__(self):
        super(LevenshteinCosineLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, a, b, d, eps=1e-8):
        """ This uses a siamese loss, specifically the loss is zero when the euclidean distance between words is equal to its Levenshtein distance.

        Dimensions:
            a: batch x embedding
            b: batch x embedding
            t: batch x [0,1] (float)
        """
        c = (self.cos(a, b) + 1) / 2
        return torch.mean((c - d) ** 2)


class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding, num_layers=2, dropout=0.0):
        super(Encoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Embedding layer that will be shared with Decoder
        self.embedding = embedding

        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size,
                          num_layers=num_layers,
                          dropout=dropout,
                          bidirectional=True)

    def forward(self, input_sequence, input_lengths):
        # Convert input_sequence to word embeddings
        word_embeddings = self.embedding(input_sequence)

        # Pack the sequence of embeddings
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(word_embeddings, input_lengths, enforce_sorted=False)

        # Run the packed embeddings through the GRU, and then unpack the sequences
        outputs, hidden = self.gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # The ouput of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class DeepLevenshtein(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(DeepLevenshtein, self).__init__()

        # Embedding layer shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Encoder network
        self.encoder = Encoder(hidden_size,
                               embedding_size,
                               self.embedding,
                               num_layers=2,
                               dropout=0.5)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, src_sequence, trg_sequence):

        # Unpack input_sequence tuple
        src_tokens, src_lengths = src_sequence
        trg_tokens, trg_lengths = trg_sequence

        # Pass through the first half of the network
        src_encoder_outputs, src_hidden = self.encoder(src_tokens, src_lengths)
        trg_encoder_outputs, trg_hidden = self.encoder(trg_tokens, trg_lengths)

        # Pass through the shared fully connected and activate
        src_fc = self.activation(self.fc(src_hidden[-1]))
        trg_fc = self.activation(self.fc(trg_hidden[-1]))

        return src_fc, trg_fc


class InMemoryDataset(data.Dataset):
    def __init__(self, data_lines, fields, levenshtein):
        examples = []
        for line in data_lines:
            id, text_a, text_b, label = line
            nld = normalized_levenshtein(text_a, text_b)
            if levenshtein:
                examples.append(data.Example.fromlist([id, text_a, text_b, nld], fields))
            else:
                examples.append(data.Example.fromlist([id, text_a, text_b, label], fields))
        super(InMemoryDataset, self).__init__(examples, fields)
        self.sort_key = None

    @classmethod
    def all(cls, path, fields, levenshtein=True, seed=None):
        def to_example(idx, bytes):
            line = bytes.decode("utf-8")
            if not line:
                return None
            arguments = line.strip().split('\t')
            if len(arguments) == 3:
                return str(idx), arguments[0], arguments[1], int(arguments[2])
            return str(idx), arguments[1], arguments[2], int(arguments[3])
        with open(path, mode="r", encoding="utf8") as file_obj:
            map_file = mmap.mmap(file_obj.fileno(), 0, prot=mmap.PROT_READ)
            examples = list(starmap(to_example, enumerate(iter(map_file.readline, b""))))
        train_data = cls(examples, fields, levenshtein)
        return train_data


def trn_model(model, iterator, criterion, optimizer, clip=1.0):
    # Put the model in training mode!
    model.train()

    epoch_loss = 0

    for idx, batch in enumerate(iterator):
        text_a, text_b, label = batch.text_a, batch.text_b, batch.label

        # zero out the gradient for the current batch
        optimizer.zero_grad()

        # Run the batch through our model
        src_hidden, trg_hidden = model(text_a, text_b)

        # if idx == 0:
        #     c = list(map(lambda x: round(x, 4), torch_cosine(src_hidden, trg_hidden).detach().cpu().numpy().tolist()[0][:10]))
        #     d = list(map(lambda x: round(x, 4), label.detach().cpu().numpy().tolist()[:10]))
        #     print(list(zip(c, d)))

        # LevenshteinCosineLoss
        loss = criterion(src_hidden, trg_hidden, label)

        # Perform back-prop and calculate the gradient of our loss function
        loss.backward()

        # Clip the gradient if necessary.
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update model parameters
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def val_model(model, iterator, criterion, optimizer):
    # Put the model in training mode!
    model.eval()

    epoch_loss = 0

    for idx, batch in enumerate(iterator):
        text_a, text_b, label = batch.text_a, batch.text_b, batch.label

        # zero out the gradient for the current batch
        optimizer.zero_grad()

        # Run the batch through our model
        src_hidden, trg_hidden = model(text_a, text_b)

        # LevenshteinCosineLoss
        loss = criterion(src_hidden, trg_hidden, label)

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


""" Script Beginning """

if __name__ == "__main__":
    """
    /apollo/env/KevquinnSandbox/bin/python deep_levenshtein_lstm.py --emb-config /home/ec2-user/dl/1/config.json
    """
    seed_val = 123

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--clip', type=float, default=50.0)
    parser.add_argument('--emb-config', type=str)
    parser.add_argument('--gbt', action='store_true')
    parser.add_argument('--emb-retrain', action='store_true')
    parser.add_argument('--gbt-retrain', action='store_true')
    args = parser.parse_args()

    t0 = time.time()

    with open(args.emb_config, 'r+') as handle:
        emb_config = json.load(handle)

    print(emb_config)

    DATASET = '/home/ec2-user/emb_lite.csv'

    EPOCHS = 25
    CLIP = 1
    EMB_PATH = emb_config['embedding_path']
    MAX_VOCAB_SIZE = emb_config['max_vocab_size']
    MIN_COUNT = emb_config['min_count']
    MAX_SEQUENCE_LENGTH = emb_config['max_sequence_length']
    BATCH_SIZE = emb_config['batch_size']
    EMBEDDING_DIM = emb_config['embedding_dim']
    HIDDEN_DIM = emb_config['hidden_dim']
    LEARNING_RATE = emb_config['learning_rate']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ID = data.Field(use_vocab=False, sequential=False, dtype=torch.int)
    WORD = data.Field(include_lengths=True, init_token='<sos>', eos_token='<eos>', tokenize=char_tokenizer)
    LABEL = data.LabelField(use_vocab=False, sequential=False, dtype=torch.float)

    fields = [
        ('id', ID),
        ('text_a', WORD),
        ('text_b', WORD),
        ('label', LABEL)
    ]

    phonemes = InMemoryDataset.all(DATASET, fields)
    WORD.build_vocab(phonemes, max_size=MAX_VOCAB_SIZE, min_freq=MIN_COUNT)

    trn_data, tst_data = phonemes.split()
    trn_data, val_data = trn_data.split()

    trn_iterator, val_iterator, tst_iterator = data.Iterator.splits(
        (trn_data, val_data, tst_data),
        sort=False,
        batch_size=BATCH_SIZE,
        device=DEVICE)

    pad_idx = WORD.vocab.stoi['<pad>']
    eos_idx = WORD.vocab.stoi['<eos>']
    sos_idx = WORD.vocab.stoi['<sos>']
    unk_idx = WORD.vocab.stoi[WORD.unk_token]

    print('pad_idx: ({0}), eos_idx: ({1}), sos_idx: ({2})'.format(pad_idx, eos_idx, sos_idx))

    print('Finished loading vocab and iterators.')

    VOCAB_SIZE = len(WORD.vocab)

    model = DeepLevenshtein(EMBEDDING_DIM,
                HIDDEN_DIM,
                VOCAB_SIZE).to(DEVICE)

    model.embedding.weight.data[unk_idx] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[pad_idx] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.requires_grad = False
    optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad is True], lr=LEARNING_RATE)
    criterion = LevenshteinCosineLoss()

    min_val_loss = 10
    for epoch in range(EPOCHS):
        trn_loss = trn_model(model, trn_iterator, criterion, optimizer, CLIP)
        val_loss = val_model(model, val_iterator, criterion, optimizer)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), EMB_PATH)

        print("   % Time: {:5.0f} | Epoch: {:5} | Batch: {:4}/{}"
              " | Train loss: {:.4f} | Val loss: {:.4f}"
              .format(time.time() - t0, epoch, trn_iterator.iterations,
                      len(trn_iterator), trn_loss, val_loss))

    model.load_state_dict(torch.load(EMB_PATH))
    model.eval()

    tst_loss = val_model(model, tst_iterator, criterion, optimizer)
    print('tst_loss', tst_loss)

