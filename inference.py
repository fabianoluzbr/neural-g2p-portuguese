#!/usr/bin/env python3

import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.data import ParserLexicon
from model import Encoder, Decoder
from utils.config import DataConfig, ModelConfig, TestConfig
from utils.text_tools import tokenize_pt

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
        self.ds = ParserLexicon(
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

def is_ponctuation(token):
    if token in ['.','?','!',',',':',';']:
        return True
    return False

def inference(sentence, char_separator='|', visualize=False):
    
    tokens = tokenize_pt(sentence)
    g2p = G2P()
    phone_phrase = ""

    for item in tokens:
        print(item)
        if is_ponctuation(item):
            phone_phrase += char_separator+item+char_separator+" "
        else:
            result = g2p(item, visualize)[:-1]
            phoneme = char_separator+char_separator.join(result)+char_separator
            phone_phrase += phoneme+" "
        
    return phone_phrase.strip()[1:-1]    

if __name__ == '__main__':
    # get word
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, default='olá, vamos testar esse projeto.')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()
    
    print(inference(args.sentence, char_separator='|'))
        
