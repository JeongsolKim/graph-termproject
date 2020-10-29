import dgl
import torch
from utils import *
import os
import warnings

warnings.filterwarnings('ignore')

preprocess(data='train')
preprocess(data='valid')
preprocess(data='test')