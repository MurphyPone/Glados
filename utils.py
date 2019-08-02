import torch
import torch.nn.functional as F

def get_device():
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


def one_hot(arr, n_labels):
    return F.one_hot(arr, n_labels).float()


def read_data(filename, batch_size, seq_size, val_ratio=0.1):
    # read data
    with open(f'data/{filename}', 'r') as f:
        text = f.read()

    # make encoding and decoding dictionaries
    chars = set(text)
    int2char = dict(enumerate(chars))
    char2int = {v: k for k, v in int2char.items()}

    # make data divisible by batch and sequence sizes
    idx = -(len(text) % (batch_size * seq_size))
    encoded = [char2int[c] for c in text][:idx+1]

    # make data into batches
    device = get_device()
    X = torch.tensor(encoded[:-1]).view(batch_size, -1)
    Y = torch.tensor(encoded[1:]).view(batch_size, -1)

    # ont hot encode the input data
    X = one_hot(X, len(chars))

    # determine where to split data into training and validation data
    idx = int(val_ratio * X.shape[1])
    idx -= idx % seq_size

    # split the data and put it on the right device
    X = X[:,:-idx].to(device)
    X_val = X[:,-idx:].to(device)

    Y = Y[:,:-idx].to(device)
    Y_val = Y[:,-idx:].to(device)

    # get number of batches per epoch for training data
    num_batches = X.shape[1] // seq_size

    return X, Y, X_val, Y_val, len(chars), char2int, int2char, num_batches


def batches(X, Y, batch_size, seq_size):
    num_batches = X.shape[1] // seq_size
    for i in range(num_batches):
        yield X[:, i*seq_size:(i+1)*seq_size], Y[:, i*seq_size:(i+1)*seq_size]
