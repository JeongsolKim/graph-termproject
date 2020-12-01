from graphloaders import *
from utils import *
from model import *
from gradCAM import *
import time
import numpy as np
import torch

# file_path = './analyze/proposed/auc_score.txt'
# auc = np.zeros([25, 10])

# with open(file_path, 'r') as f:
#     lines = f.readlines()
#     for i, line in enumerate(lines):
#         scores = list(map(lambda x: float(x.strip('\n')), line.split('\t')))
#         auc[:,i] = np.array(scores)

# auc_mean = np.mean(auc, 1)
# auc_std = np.std(auc, 1)
    
# np.savetxt('./auc_mean.txt', auc_mean)
# np.savetxt('./auc_std.txt', auc_std)

gradcam('test_query/test_query_080.txt','proposed','cuda:0', 1.0, './model/proposed/model.pt')