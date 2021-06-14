#!/usr/bin/env python3

import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import re
from packaging import version
import phonemizer
from phonemizer.phonemize import phonemize

_phoneme_punctuations = '.!;:,?'

PHONEME_PUNCTUATION_PATTERN = r'['+_phoneme_punctuations+']+'

'''
def text2phone(text, language):
    
    seperator = phonemizer.separator.Separator(' |', '', '|')
    #try:
    punctuations = re.findall(PHONEME_PUNCTUATION_PATTERN, text)
    if version.parse(phonemizer.__version__) >= version.parse('2.1'):
        ph = phonemize(text, separator=seperator, strip=False, njobs=1, backend='espeak', language=language, preserve_punctuation=True, language_switch='remove-flags')
    else:
        raise RuntimeError(" [!] Use 'phonemizer' version 2.1 or older.")
    return ph.replace("| ?","|?").replace("| .","|.")
'''


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
    
    l = get_list('list.txt')
    n_l = []
    
    for item in l:
        result = text2phone(item, "pt-br")
        print(item+"  "+" ".join(result[:-1]))
        n_l.append(item+"  "+" ".join(result[:-1])) 

    from_list_file(n_l,"new_ipa.txt")
