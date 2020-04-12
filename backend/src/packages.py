from __future__ import unicode_literals, print_function, division

import os
import numpy as np
import pandas as pd
from numpy import array

import re
import time
import math
import random
import json
import string
import unicodedata
from io import open
from tqdm import tqdm
from rouge import Rouge
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate import bleu
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")