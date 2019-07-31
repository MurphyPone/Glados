import sys
import numpy as np
import torch
from visdom import Visdom
from termcolor import colored

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


def plot(t, r, data_type, name, color='#000'):
    win = data_type
    title = data_type

    if data_type not in d:
        d[data_type] = {}

    if name not in d[data_type]:
        d[data_type][name] = {'points': [], 'color': color}
    d[data_type][name]['points'].append((t, float(r)))

    data = []
    for name in d[data_type]:
        x, y = zip(*d[data_type][name]['points'])
        data.append(get_line(x, y, name, color=d[data_type][name]['color'], showlegend=True))

    layout = dict(
        title=title,
        xaxis={'title': 'Iterations'},
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

def progress(i, total, action):
    i = i + 1
    ratio = i / total
    percent = 100 * ratio
    filled = int(round(20 * ratio))

    bar = '█' * filled + '-' * (20 - filled)

    color = get_color(ratio)

    done = (percent == 100)
    done = '\n' if done is True else ''
    sys.stdout.write('\r%s |%s| ' % (action.ljust(30), bar) + rgb_colored('{0:.0f}%'.format(percent) + done, color))

    if i == total:
        sys.stdout.write('\n')

    sys.stdout.flush()
