from preprocess import *
from utils import *
from model import *
import time
import numpy as np
import torch

H1, H2, feats, label = load_graph('train/train_000.txt')

model = MyModel(in_dim=feats.size()[-1],
                hidden_dim=64,
                num_classes=label.size()[-1])

output = model(H1, H2, feats)
print(output)

score = f1_score(output, label)
print(score)

# forward_and_save_prediction(model_instance=model, threshold=0.5, file_path='train/train_001.txt', device='cpu', mode='proposed')