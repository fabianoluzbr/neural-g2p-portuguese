import os
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader


class ParserLexicon(Dataset):
    def __init__(self, inputs, outputs, dict_path):
        with open(inputs) as fi, open(outputs) as fo, open(dict_path) as fd:
            graphemes = json.load(fi)
            phonemes = json.load(fo)
            self.lexicon = json.load(fd)

        self.g2idx = {ch: idx for idx, ch in enumerate(graphemes)}
        self.idx2g = {idx: ch for idx, ch in enumerate(graphemes)}
        self.p2idx = {phn: idx for idx, phn in enumerate(phonemes)}
        self.idx2p = {idx: phn for idx, phn in enumerate(phonemes)}

    def __len__(self):
        return len(self.lexicon)

    def __getitem__(self, index):
        key, value = self.lexicon[index]
        
        x = 0
        y = 0
        ch = 0
        try:
            x = [self.g2idx[ch] for ch in key]
            y = [self.p2idx[phn] for phn in value.split(' ') if phn != '']
        except Exception as e:
            print("key",key)
            raise e
        
        

        return [0] + x + [1], [0] + y + [1]


def collate_fn(batch):
    N = len(batch)
    x, y = zip(*batch)
    in_max_len = max([len(i) for i in x])
    out_max_len = max([len(i) for i in y])

    inputs = torch.ones(in_max_len, N).long()
    outputs = torch.ones(out_max_len, N).long()

    for ind, (i, j) in enumerate(batch):
        li = len(i)
        inputs[:li, ind] = torch.Tensor(i).long()

        lj = len(j)
        outputs[:lj, ind] = torch.Tensor(j).long()

    return inputs, outputs
