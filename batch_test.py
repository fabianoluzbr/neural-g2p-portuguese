#!/usr/bin/env python3

import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import PersianLexicon
from model import Encoder, Decoder
from config import DataConfig, ModelConfig, TestConfig


def load_model(model_path, model):
    model.load_state_dict(torch.load(
        model_path,
        map_location=lambda storage,
        loc: storage
    ))
    model.to(TestConfig.device)
    model.eval()
    return model


class G2P(object):
    def __init__(self):
        # data
        self.ds = PersianLexicon(
            DataConfig.graphemes_path,
            DataConfig.phonemes_path,
            DataConfig.lexicon_path
        )

        # model
        self.encoder_model = Encoder(
            ModelConfig.graphemes_size,
            ModelConfig.hidden_size
        )
        load_model(TestConfig.encoder_model_path, self.encoder_model)

        self.decoder_model = Decoder(
            ModelConfig.phonemes_size,
            ModelConfig.hidden_size
        )
        load_model(TestConfig.decoder_model_path, self.decoder_model)

    def __call__(self, word, visualize):
        x = [0] + [self.ds.g2idx[ch] for ch in word] + [1]
        x = torch.tensor(x).long().unsqueeze(1)
        with torch.no_grad():
            enc = self.encoder_model(x)

        phonemes, att_weights = [], []
        x = torch.zeros(1, 1).long().to(TestConfig.device)
        hidden = torch.ones(
            1,
            1,
            ModelConfig.hidden_size
        ).to(TestConfig.device)
        t = 0
        while True:
            with torch.no_grad():
                out, hidden, att_weight = self.decoder_model(
                    x,
                    enc,
                    hidden
                )

            att_weights.append(att_weight.detach().cpu())
            max_index = out[0, 0].argmax()
            x = max_index.unsqueeze(0).unsqueeze(0)
            t += 1

            phonemes.append(self.ds.idx2p[max_index.item()])
            if max_index.item() == 1:
                break

        if visualize:
            att_weights = torch.cat(att_weights).squeeze(1).numpy().T
            y, x = att_weights.shape
            plt.imshow(att_weights, cmap='gray')
            plt.yticks(range(y), ['<sos>'] + list(word) + ['<eos>'])
            plt.xticks(range(x), phonemes)
            plt.savefig(f'attention/{DataConfig.language}/{word}.png')

        return phonemes

def get_list(path):
    new_l = []
    with open(path,'r') as list_file:
        for row in list_file:
            if row:
                new_l.append(row.strip())
    return new_l
    
def from_list_file(list_phone,file_name):
    
    new_f = open(file_name,"w+")
    for item in list_phone:
        new_f.write(item+'\n')
    new_f.close()

if __name__ == '__main__':
    # get word
    parser = argparse.ArgumentParser()
    parser.add_argument('--word', type=str, default='testando')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    l = get_list('list.txt')
    n_l = []

    g2p = G2P()
    
    for item in l:
        result = g2p(item, args.visualize)
        print(item+"  "+" ".join(result[:-1]))
        n_l.append(item+"  "+" ".join(result[:-1])) 

    from_list_file(n_l,"new_ipa.txt")
