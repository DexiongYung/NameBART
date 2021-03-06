import os
import string
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Model.Transformer import Transformer
from Utility.NameDataset import NameDataset
from Utility.Noiser import *
from Utility.Utility import *

parser = argparse.ArgumentParser()
parser.add_argument('--continue_training', nargs='?', default=0, type=int)

args = parser.parse_args()


SOS = 'SOS'
EOS = 'EOS'
DECODER_CHARS = [c for c in string.ascii_letters] + ['\'', '-', EOS, SOS]
NUM_DECODER_CHARS = len(DECODER_CHARS)
ENCODER_CHARS = [c for c in string.printable]
NUM_ENCODER_CHARS = len(ENCODER_CHARS)
CONTINUE_TRAINING = args.continue_training == 1
PRINT_EVERY = 100
EPOCH = 200

def train(src: list, trg: list):
    optimizer.zero_grad()

    src = indexTensor([src], len(src), ENCODER_CHARS)
    trg = targetTensor([trg], len(trg), DECODER_CHARS)
    loss = 0

    for i in range(len(trg) - 1):
        prob = transformer.forward(src, trg[0:i + 1])
        loss += criterion(prob[i], trg[i + 1])

    loss.backward()
    optimizer.step()

    return loss


def enumerate_train(dl: DataLoader):
    iter = 0
    total_loss = 0
    all_losses = []

    for i in range(EPOCH):
        for name in dl:
            iter += 1
            noised_name = noise_name(name[0], ENCODER_CHARS, len(name[0]) + 2)
            noised_name_lst = [c for c in noised_name]
            trg = [SOS] + [c for c in name[0]] + [EOS]
            total_loss += train(noised_name_lst, name[0]).item()

            if iter % PRINT_EVERY:
                all_losses.append(total_loss / PRINT_EVERY)
                total_loss = 0
                plot_losses(all_losses, f"Iteration of {PRINT_EVERY}", "Cross Entropy Loss")
                torch.save({'weights': transformer.state_dict()}, "Weights/Bart.path.tar")


transformer = Transformer(NUM_ENCODER_CHARS, NUM_DECODER_CHARS)
criterion = nn.CrossEntropyLoss()
lr = 0.0005
optimizer = torch.optim.Adam(transformer.parameters(), lr=lr)
df = pd.read_csv('Data/first.csv')
ds = NameDataset(df, 'name')
dl = DataLoader(ds, batch_size=1, shuffle=True)

if CONTINUE_TRAINING:
    transformer.load_state_dict(torch.load('Weights/Bart.path.tar')['weights'])

enumerate_train(dl)
