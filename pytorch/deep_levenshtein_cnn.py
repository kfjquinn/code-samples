
import random
import mmap
import Levenshtein
from itertools import repeat
from functools import partial
from operator import itemgetter

# importing the libraries
import pandas as pd
import numpy as np

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)


class MyLoss(Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, a, b, t):
        """ This uses a siamese loss, specifically the loss is zero when the euclidean distance between words is equal to its Levenshtein distance.

        Dimensions:
            a: batch x cnn embedding
            b: batch x cnn embedding
            t: batch x 1 (int)
        """
        e = ((a - b) ** 2).sum(1)  # euclidean distance
        return torch.mean((e - t) ** 2)  # mean levenshtein loss


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 64, kernel_size=(28, 3), stride=1, padding=0),
        )

        self.linear_layers = Sequential(
            Dropout(0.2),
            Linear(1408, 512)
        )

    def forward(self, x):
        x = self.cnn_layers(x)     # batch x c_out, c_in, num_strides
        x = x.view(x.size(0), -1)  # flatten cnn output, batch x (num_strides * channels)
        x = self.linear_layers(x)
        return x


def read_examples(input_file, levenshtein=True):
    """Read a list of `InputExample`s from an input file."""
    def to_example(bytes):
        line = bytes.decode("utf-8")
        if not line:
            return None
        arguments = line.strip().split('\t')
        if len(arguments) == 3:
            label = Levenshtein.distance(arguments[0], arguments[1]) if levenshtein else int(arguments[2])
            return arguments[0], arguments[1], label
        label = Levenshtein.distance(arguments[1], arguments[2]) if levenshtein else int(arguments[3])
        return arguments[1], arguments[2], label
    with open(input_file, mode="r", encoding="utf8") as file_obj:
        map_file = mmap.mmap(file_obj.fileno(), 0, prot=mmap.PROT_READ)
        examples = list(map(to_example, iter(map_file.readline, b"")))
    return examples


def to_matrix(max_vocab_len, max_sequence_len, utterance):
    """ Create an encoding matrix for an utterance.

    No spaces captured, they are replaced.  The first index is the padding token, the last index is the unknown bin.

    Example:
              b  a  d  %

      #       0  0  0  0  1  1  1 ... 1
      a       0  1  0  0  0  0  0 ... 0
      b       1  0  0  0  0  0  0 ... 0
      c       0  0  0  0  0  0  0 ... 0
      d       0  0  1  0  0  0  0 ... 0
      ...
      z       0  0  0  0  0  0  0 ... 0
      <unk>   0  0  0  1  0  0  0 ... 0
    """
    num_rows = 1 + max_vocab_len + 1

    def to_ascii(letter):
        value = ord(letter.lower()) - 96
        return value if 0 < value < 27 else num_rows - 1

    utterance_numeric = list(map(to_ascii, list(utterance.replace(' ', ''))))[:max_sequence_len]  # truncate for longer than max
    utterance_numeric_padding = list(repeat(0, max_sequence_len - len(utterance_numeric)))  # first row is padding

    i = torch.LongTensor([
        utterance_numeric + utterance_numeric_padding,
        list(range(max_sequence_len))
    ])

    v = torch.ones(max_sequence_len)
    return torch.sparse.FloatTensor(i, v, torch.Size([num_rows, max_sequence_len])).to_dense()


def to_levenshtein(pair):
    return Levenshtein.distance(*pair)


def to_dataloader(examples, batch_size=32):
    _to_matrix = partial(to_matrix, 26, 24)

    label = torch.tensor(list(map(to_levenshtein, examples)), dtype=torch.float)

    tensors_a = torch.stack(list(map(_to_matrix, map(itemgetter(0), examples))), dim=0)
    tensors_b = torch.stack(list(map(_to_matrix, map(itemgetter(1), examples))), dim=0)

    dataset = TensorDataset(tensors_a, tensors_b, label)
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size, shuffle=False)


def train(model, trn_dataloader, val_dataloader, device, debug, epochs, learning_rate, batch_size):
    print('epochs: {0}, learning_rate: {1}, batch_size: {2}'.format(epochs, learning_rate, batch_size))
    model.train()
    best_val_loss = 30
    for _ in range(int(epochs)):
        n_total = 0
        total_trn_loss = 0
        for step, batch in enumerate(trn_dataloader):
            n_total += len(batch)

            batch = tuple(t.to(device) for t in batch)
            matrix_a, matrix_b, target = batch

            model_a = model(matrix_a.unsqueeze(1))  # add channel parameter by unsqueezing
            model_b = model(matrix_b.unsqueeze(1))  # add channel parameter by unsqueezing

            loss = criterion(model_a, model_b, target)

            total_trn_loss += loss.item() * len(batch)

            if debug and step % 1000 == 0:
                total_val_loss = 0
                for val_step, val_batch in enumerate(val_dataloader):
                    val_batch = tuple(t.to(device) for t in val_batch)
                    val_matrix_a, val_matrix_b, val_target = val_batch

                    with torch.no_grad():
                        val_model_a = model(val_matrix_a.unsqueeze(1))  # add channel parameter by unsqueezing
                        val_model_b = model(val_matrix_b.unsqueeze(1))  # add channel parameter by unsqueezing
                        val_loss = criterion(val_model_a, val_model_b, val_target)

                    total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(val_dataloader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), '/home/ec2-user/dl/pytorch_model.bin')

                avg_trn_loss = total_trn_loss / n_total
                print('trn loss: {0:.5f}, val loss: {1:.5f}'.format(avg_trn_loss, avg_val_loss))

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

    print('Finished training, moving to evaluation. ')
    model.load_state_dict(torch.load('/home/ec2-user/dl/pytorch_model.bin'))
    return model


def eval(model, dataloader, examples):
    n_total = 0
    total_val_loss = 0
    for val_step, val_batch in enumerate(dataloader):
        n_total += len(val_batch)
        val_batch = tuple(t.to(device) for t in val_batch)
        val_matrix_a, val_matrix_b, val_target = val_batch

        with torch.no_grad():
            val_model_a = model(val_matrix_a.unsqueeze(1))  # add channel parameter by unsqueezing
            val_model_b = model(val_matrix_b.unsqueeze(1))  # add channel parameter by unsqueezing
            val_loss = criterion(val_model_a, val_model_b, val_target)

        total_val_loss += val_loss.item()
    return total_val_loss / n_total


def test(dataloader, examples, encoder, classifier):
    sig = torch.nn.Sigmoid()

    p = []
    l = []
    for val_step, val_batch in enumerate(dataloader):
        val_batch = tuple(t.to(device) for t in val_batch)
        val_matrix_a, val_matrix_b, tst_labels = val_batch

        with torch.no_grad():
            val_model_a = encoder(val_matrix_a.unsqueeze(1))  # add channel parameter by unsqueezing
            val_model_b = encoder(val_matrix_b.unsqueeze(1))  # add channel parameter by unsqueezing

            features = torch.cat([val_model_a, val_model_b], dim=1)
            outputs = classifier(features)

            probs = sig(outputs.squeeze()).detach().cpu().numpy()
            label = tst_labels.to('cpu').numpy().tolist()

            p.extend(probs)
            l.extend(label)

    return l, p


def show(dataloader, examples, model):
    batch = dataloader.__iter__().next()
    batch = tuple(t.to(device) for t in batch)
    matrix_a, matrix_b, target = batch

    model_a = model(matrix_a.unsqueeze(1))  # add channel parameter by unsqueezing
    model_b = model(matrix_b.unsqueeze(1))  # add channel parameter by unsqueezing

    e = np.round(((model_a - model_b) ** 2).sum(1).cpu().detach().numpy())
    t = target.cpu().detach().numpy()

    for example, euclid, leven in zip(examples, e, t):
        print(example, euclid, leven)


""" Script Beginning """

if __name__ == "__main__":
    """ /apollo/env/KevquinnSandbox/bin/python deep_levenshtein_cnn.py """

    seed_val = 123

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    debug = True
    epochs = 50
    learning_rate = 0.00001
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = MyLoss().to(device)

    # emb_examples = read_examples('/home/ec2-user/emb.csv')
    trn_examples = read_examples('/home/ec2-user/trn_lite.csv')
    val_examples = read_examples('/home/ec2-user/val.csv')
    tst_examples = read_examples('/home/ec2-user/tst.csv')

    # emb_dataloader = to_dataloader(emb_examples, batch_size=batch_size)
    trn_dataloader = to_dataloader(trn_examples, batch_size=batch_size)
    val_dataloader = to_dataloader(val_examples, batch_size=batch_size)
    tst_dataloader = to_dataloader(tst_examples, batch_size=batch_size)

    trained = train(model, trn_dataloader, val_dataloader, device, debug, epochs, learning_rate, batch_size)
    test_loss = eval(trained, tst_dataloader, tst_examples)
    show(tst_dataloader, tst_examples, trained)
    print('test_loss: ', test_loss)

