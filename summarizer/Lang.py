from __future__ import unicode_literals, print_function, division

import os
import time
import math
import numpy as np 
import pandas as pd

import re
import string
import random
import unicodedata
from io import open
from rouge import Rouge

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

debug = True
to_print_epoch = True

SOS_token = 0
EOS_token = 1
UNK_token = 2

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index,self.word2count,self.index2word = {},{},{}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS", UNK_token:"UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1