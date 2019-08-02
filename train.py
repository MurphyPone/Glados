import sys
import torch
import torch.nn.functional as F

from model import RNN
from utils import read_data, one_hot, batches, get_device
from visualize import *
from termcolor import colored
import math

batch_size = 128
seq_size = 100
vis_iter = 20
grad_norm = 5
device = get_device()

# generate dataset
try: filename = sys.argv[1]
except: filename = 'input'


## Peter's Helpers
def is_real(text):
    ls = text.split()
    result = 0
    for word in ls:
        if d.check(word):
            result += 1
    return result

def update_js(text):
    ls = text.split()
    result = 0
    for word in ls:
        if 'Jew' in word or 'jew' in word or 'Jews' in word or 'Jews' in word:
            result += 1

    return result

X, Y, X_val, Y_val, n_chars, char2int, int2char, num_batches = read_data(filename, batch_size, seq_size)

# make network and optimizer
net = RNN(n_chars).to(device)
opt = torch.optim.Adam(net.parameters(), lr=0.005)


'''
TRAINING
'''
# get loss for all validation batches
def validation_loss():
    with torch.no_grad():
        val_losses = []
        val_h = net.blank_hidden(batch_size)
        for x, y in batches(X_val, Y_val, batch_size, seq_size):
            out_val, val_h = net(x, val_h)
            val_loss = F.cross_entropy(out_val.transpose(1,2), y)
            val_losses.append(val_loss)
        val_losses = torch.stack(val_losses)
    return val_losses

# train network
def train(epochs=20):
    reals, total, js, iters = 0, 0, 0, 0

    for epoch in range(epochs):

        batch_num = 0
        losses = []
        h = net.blank_hidden(batch_size)
        for x, y in batches(X, Y, batch_size, seq_size):

            # use network predictions to compute loss
            h = tuple([state.detach() for state in h])
            out, h = net(x, h)
            loss = F.cross_entropy(out.transpose(1,2), y)

            # optimization step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_norm)
            opt.step()

            # print training progress
            progress(batch_num, num_batches, iters, epochs * num_batches, epoch)

            # bookkeeping
            losses.append(loss)
            batch_num += 1
            iters += 1

        # plot loss after every epoch
        plot(epoch, torch.stack(losses), 'Loss', 'Training', '#5DE58D', refresh=False)
        plot(epoch, validation_loss(), 'Loss', 'Validation', '#4AD2FF')

        plot(epoch, math.log(total+1), 'Words', '\'words\'', '#52d737')
        plot(epoch, math.log(reals+1), 'Words', 'real words', '#d437d7')
        plot(epoch, js, 'Zionism', 'instances', '#c12b2b')
        print(colored('\n\n' + smpl + '\n', 'cyan')) # print sample

'''
GENERATION
'''
# convert network output to character
def net2char(x, top_k=5):
    # get top k probabilities
    probs = F.softmax(x.squeeze(), dim=0)
    probs, choices = torch.topk(probs, k=top_k)

    # sample from the top k choices
    idx = torch.multinomial(probs, 1)
    return int2char[choices[idx].item()]

# take a single character and encode it so it works as network input
def format_input(x):
    x = torch.tensor([[char2int[x]]])
    return one_hot(x, n_chars).to(device)

# generate multiple example text chunks
def generate(first_chars='A', example_len=100, examples=1):
    with torch.no_grad():
        for _ in range(examples):
            first_chars = list(first_chars)
            chars = ''.join(first_chars)

            # run initial chars through model to generate hidden states
            h = net.blank_hidden()
            for c in first_chars:
                inp = format_input(c)
                x, h = net(inp, h)

            # generate new chars
            for _ in range(example_len):
                choice = net2char(x)
                chars += choice

                inp = format_input(choice)
                x, h = net(inp, h)

            # print the results
            print('-' * 40 + f'\n{chars}')


'''
"DO IT" - Palpatine
'''
train(epochs=30)
generate(example_len=1000, examples=4)
