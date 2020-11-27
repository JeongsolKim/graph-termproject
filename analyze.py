import dgl
import torch
import os
import warnings
import argparse
import datetime
import time
import random
import tqdm
from graphloaders import *
from model import *
from utils import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--model', type=str, default='proposed', help="One of 'ffn', 'sage_on_line', 'proposed'.")
parser.add_argument('--gpu', type=int, default=0, help='Number of GPU to use for training.')
parser.add_argument('--model_load_path', type=str, default='./model/proposed/model.pt')
parser.add_argument('--analyze_save_path', type=str, default='./analyze/proposed/')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Check the device
if args.cuda:
	gpu_num = args.gpu
	device = 'cuda:{}'.format(gpu_num)
else:
	device = 'cpu'

# Data loader and Model
if args.model == 'ffn':
	loader = FeatureLabelLoader()
	model = SimpleFFN(in_feats=1800, nhid=256, num_classes=25).to(device)
	print('\n>> Model: Two layers feed-forward network on features')
elif args.model == 'sage_on_line':
	loader = LineGraphLoader()
	model = MyModel_line(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='mean', activation='sigmoid').to(device)
	print('\n>> Model: GraphSage on linegraph')
elif args.model == 'proposed':
	loader = ModifiedLineGraphLoader()
	model = MyModel(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='mean', activation='sigmoid').to(device)
	print('\n>> Model: Two GraphSages on modified linegraph (proposed)')

# Set loss
Myloss = nn.BCELoss().to(device)

# Load
ckpt = torch.load(args.model_load_path)
model.load_state_dict(ckpt['model_state_dict'])

model.eval()