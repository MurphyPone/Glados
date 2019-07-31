import sys
import pathlib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from termcolor import colored

from model import CharRNN
from utils import *
from visualize import *

# read text and make character conversion utilities
try:
    filename = sys.argv[1]
except:
    filename = 'shakespeare'

with open(f'data/{filename}', 'r') as f:
    text = f.read()
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded_text = np.array([char2int[ch] for ch in text])

# set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# training method
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, vis_iter=10, save_iter=10):
    pathlib.Path('saved_models').mkdir(exist_ok=True)

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    net = net.to(device)

    counter = 0
    n_chars = len(net.chars)
    for e in range(epochs):
        h = net.init_hidden(batch_size)

        for x, y, epoch_part, parts_per_epoch in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)

            # detach hidden states from computation graph
            h = tuple([each.data for each in h])

            # loss calculation
            output, h = net(inputs, h)
            loss = criterion(output, targets.view(batch_size*seq_length).long())

            # optimization step
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # print epoch progress
            progress(epoch_part, parts_per_epoch, f'Epoch {e+1}/{epochs}')
            if e == epochs - 1 and epoch_part == parts_per_epoch - 1: print()

            # print loss stats occasionally
            if counter % vis_iter == vis_iter - 1:
                val_h = net.init_hidden(batch_size)
                mean_val_loss = None
                net.eval()
                for x, y, _, _ in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x.to(device), y.to(device)

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())

                    if mean_val_loss is None: mean_val_loss = val_loss.item()
                    else: mean_val_loss += (val_loss.item() - mean_val_loss) / counter

                net.train()

                plot(counter, loss, 'Loss', 'Training', '#FA5784')                                                      # PROGRESS BAR FOR % DONE W/ EPOCH
                plot(counter, mean_val_loss, 'Loss', 'Validation', '#FFAED4')

            # save model occasionally
            if e % save_iter == save_iter - 1:
                checkpoint = {'n_hidden': net.n_hidden, 'n_layers': net.n_layers, 'state_dict': net.state_dict(), 'tokens': net.chars}
                with open(f'saved_models/rnn_epoch_{e+1}.net', 'wb') as f:
                    torch.save(checkpoint, f)


# Generate the next character given the previous one
def predict_next(net, char, h=None, top_k=None):
        x = np.array([[char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x).to(device)

        h = tuple([each.data for each in h])                                    # detach hidden state from history
        out, h = net(inputs, h)                                                 # get the output of the model
        p = F.softmax(out, dim=1).data.cpu()                                    # get the character probabilities

        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()

        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p / p.sum())

        # return the encoded value of the predicted char and the hidden state
        return int2char[char], h


# Generate `size` characters as a block of text
def generate_text(net, size, first_chars='The', top_k=None):
    net = net.to(device)

    # run first chars through network
    chars = [ch for ch in first_chars]
    h = net.init_hidden(1)
    for ch in first_chars:
        char, h = predict_next(net, ch, h, top_k=top_k)

    # generate new chars
    chars.append(char)
    for _ in range(size):
        char, h = predict_next(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)


net = CharRNN(chars, n_hidden=512, n_layers=2)
box('Network Architecture')
print(net)

# train the model
box(f'Training on {filename.upper()}', color='yellow')
train(net, encoded_text, epochs=60, batch_size=128, seq_length=100, lr=0.001, vis_iter=20)
box('Results', color='green')
print(generate_text(net, 1000, first_chars='A', top_k=5))
