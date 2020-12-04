from graphloaders import *
from utils import *
from model import *
from gradCAM import *
import time
import numpy as np
import torch

# file_path = './analyze/save_line_pool/sage_on_line/auc_score.txt'
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

gradcam('train/train_732.txt', './model/proposed/model.pt',  'cuda:0')
# gradcam('valid_query/valid_query_101.txt', './model/proposed/model.pt',  'cuda:0')
