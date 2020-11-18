from preprocess import *
from utils import *
import time
import numpy as np
import torch

out = np.zeros((1,5))
out[0,3] = 1
out[0,4] = 1
label = np.zeros((1,5))
label[0,1] = 1
label[0,3] = 1


out = torch.tensor(out)
label = torch.tensor(label)
loss = torch.nn.BCELoss()

f1 = f1_score(out, label)
print(f1)

_,_,_,_ = load_graph('./valid_query/valid_query_000.txt')
