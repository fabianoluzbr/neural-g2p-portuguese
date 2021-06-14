import os
import json

import torch

cpu = torch.device('cpu')
gpu = torch.device('cuda')


class DataConfig(object):
    language = os.getenv('LANGUAGE', 'PT-IPA')
    graphemes_path = f'resources/{language}/Graphemes.json'
    phonemes_path = f'resources/{language}/Phonemes.json'
    lexicon_path = f'resources/{language}/Lexicon.json'


class ModelConfig(object):
    with open(DataConfig.graphemes_path) as f:
        graphemes_size = len(json.load(f))

    with open(DataConfig.phonemes_path) as f:
        phonemes_size = len(json.load(f))

    hidden_size = 256


class TrainConfig(object):
    device = gpu if torch.cuda.is_available() else cpu
    lr = 2e-4
    batch_size = 64
    epochs = int(os.getenv('EPOCHS', '30'))
    log_path = f'log/{DataConfig.language}'


class TestConfig(object):
    device = cpu
    encoder_model_path = f'checkpoints/encoder_e{TrainConfig.epochs:02}.pth'
    decoder_model_path = f'checkpoints/decoder_e{TrainConfig.epochs:02}.pth'
