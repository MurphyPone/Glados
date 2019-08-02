import sys
import torch
from visdom import Visdom
from termcolor import colored
from math import sqrt

d = {}
viz = Visdom()

def get_line(x, y, name, color='#000', isFilled=False, fillcolor='transparent', width=2, showlegend=False):
        if isFilled:
            fill = 'tonexty'
        else:
            fill = 'none'

        return dict(
            x=x,
            y=y,
            mode='lines',
            type='custom',
            line=dict(
                color=color,
                width=width),
            fill=fill,
            fillcolor=fillcolor,
            name=name,
            showlegend=showlegend
        )


def plot_loss(epoch, loss, name, color='#000'):
    win = name
    title = name + ' Loss'

    if name not in d:
        d[name] = []
    d[name].append((epoch, loss.item()))

    x, y = zip(*d[name])
    data = [get_line(x, y, 'loss', color=color)]

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
        yaxis={'title': 'Loss'}
    )

    viz._send({'data': data, 'layout': layout, 'win': win})


def plot(x, y, data_type, name, color='#000', refresh=True):
    # create dictionary spots if they don't already exist
    if data_type not in d:
        d[data_type] = {}
    if name not in d[data_type]:
        d[data_type][name] = {'points': [], 'color': color}

    # if given a single number, save its float
    if not isinstance(y, (float, int)):
        if len(y.shape) == 0:
            y = float(y)
            # if given a set of numbers, save their mean and confidence interval info
        else:
            y = y.cpu()
            mean, std = y.mean().item(), 3.291 * y.std().item() / sqrt(len(y))
            lower, upper = mean - std, mean + std
            y = (lower, mean, upper)

    # save the modified data
    d[data_type][name]['points'].append((x, y))

    # the actua plotting
    if refresh:
        win = data_type
        title = data_type
        data = []
        for name in d[data_type]:
            x, y = zip(*d[data_type][name]['points'])

            # if extracting mean and confidence internval info, plot the mean with error shading
            if isinstance(y[0], tuple):
                lower, mean, upper = zip(*y)
                data.append(get_line(x, lower, name, color='transparent'))
                data.append(get_line(x, upper, name, color='transparent', isFilled=True, fillcolor=d[data_type][name]['color'] + '44'))
                data.append(get_line(x, mean, name, color=d[data_type][name]['color'], showlegend=True))
            # if extracting single values, plot them as a single line
            else:
                data.append(get_line(x, y, name, color=d[data_type][name]['color'], showlegend=True))

        layout = dict(
            title=title,
            xaxis={'title': 'Epochs'},
            yaxis={'title': data_type}
        )

        viz._send({'data': data, 'layout': layout, 'win': win})



def box(text, color='cyan'):
    l = len(text) + 2
    text = colored(text, color)
    top = '┌' + ('─' * l) + '┐'
    mid = f'│ {text} │'
    bottom = '└' + ('─' * l) + '┘'
    print(f'\n{top}\n{mid}\n{bottom}')

def get_color(ratio):
    r = int(min(max(500 * (1 - ratio), 0), 255))
    g = int(min(max(500 * ratio, 0), 255))
    return[r, g, 0]

def rgb_colored(text, rgb):
    return f'\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\x1b[0m'

def progress(x, X, y, Y, epoch_num):
    '''
    x / X: current inner loop / max inner loop
    y / Y: current outer loop / max outer loop
    epoch_num: number of current epoch
    '''

    x, y = x + 1, y + 1
    x_ratio, y_ratio = x / X, y / Y
    x_filled, y_filled = int(round(20 * x_ratio)), int(round(20 * y_ratio))
    x_bar, y_bar = '█' * x_filled + '-' * (20 - x_filled), '█' * y_filled + '-' * (20 - y_filled)

    x_color, y_color = get_color(x_ratio), get_color(y_ratio)
    x_percent, y_percent = rgb_colored(f'{100 * x_ratio:.0f}%', x_color), rgb_colored(f'{100 * y_ratio:.0f}%', y_color)

    sys.stdout.write(f'\rEpoch {epoch_num} |{x_bar}| {x_percent} \t\t Total |{y_bar}| {y_percent}')
    if x == X and y == Y: sys.stdout.write('\n')
    sys.stdout.flush()
