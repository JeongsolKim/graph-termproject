from graphloaders import *
from utils import *
from model import *
import time
import numpy as np
import torch

# loader = LineGraphLoader()
# gl, feats, label = loader.load_graph()

# model = MyModel_line(in_dim=feats.size()[-1],
# 					hidden_dim=64,
# 					num_classes=label.size()[-1])

# output = model(gl, feats)
# print(output)
# print(label)
# score = f1_score(output, label)
# print(score)

# forward_and_save_prediction(model_instance=model, threshold=0.5, file_path='train/train_001.txt', device='cpu', mode='proposed')

f1 = total_f1_score('valid_answer/', 'prediction/line/valid_query/')
print(f1)