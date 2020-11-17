import dgl
from dgl.data.utils import load_graphs
import torch
import os
import warnings
from preprocess import *
from model import *

warnings.filterwarnings('ignore')

glist, label_dict = load_graphs("./processed/train_graphs_1.bin", [0])
g = glist[0]

model = SimpleFFN(in_feats=1800, nhid=512, num_classes=25)

print(model(g))
# DOTO
# 1. make customized dataset class
# 2. make function for constructing the modified line graphs
# 3. make entire structure for training
# - for SVM
# - for simple FFN
# - for GraphSAGE
# 4. graphics.
