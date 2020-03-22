import torch
import matplotlib.pyplot as plt
from torch import Tensor
from Constants import *

def indexTensor(names: list, max_len: int, allowed_chars: list):
    tensor = torch.zeros(max_len, len(names)).type(torch.LongTensor)
    for i, name in enumerate(names):
        for j, letter in enumerate(name):
            index = allowed_chars.index(letter)

            if index < 0:
                raise Exception(f'{names[j][i]} is not a char in {allowed_chars}')

            tensor[j][i] = index
    return tensor.to(DEVICE)


def targetTensor(names: list, max_len: int, allowed_chars: list):
    batch_sz = len(names)
    ret = torch.zeros(max_len, batch_sz).type(torch.LongTensor)
    for i in range(max_len):
        for j in range(batch_sz):
            index = allowed_chars.index(names[j][i])

            if index < 0:
                raise Exception(f'{names[j][i]} is not a char in {allowed_chars}')

            ret[i][j] = index
    return ret.to(DEVICE)

def plot_losses(loss: list, x_label: str, y_label: str, folder: str = "Plot", filename: str = "Result"):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()
