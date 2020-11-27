import dgl
import torch
import os
import warnings
import argparse
import time
import random
import tqdm
from graphloaders import *
from model import *
from utils import *

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--graph_type', type=str, default='only_feats', help='only_feats or lingraph or modified_linegraph')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--plot', type=bool, default=False, help='Draw learning curve after training.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.graph_type in ['only_feats', 'linegraph', 'modified_linegraph'], \
	print('--graph_type should be among only_feats or linegraph or modified_linegraph, not {}'.format(args.graph_type))

gpu_num = 0
if args.cuda:
	device = 'cuda:{}'.format(gpu_num)
else:
	device = 'cpu'

# Data loader
if args.graph_type == 'linegraph':
	loader = LineGraphLoader()
elif args.graph_type == 'modified_linegraph':
	loader = ModifiedLineGraphLoader()
elif args.graph_type == 'only_feats':
	loader = FeatureLabelLoader()

# Load model
model = SimpleFFN(in_feats=1800, nhid=256, num_classes=25).to(device)

# Load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set loss
Myloss = nn.BCELoss().to(device)

def train(epoch):
	t = time.time()

	# Sampling ratio
	sample_ratio = 0.6

	# Training phase
	model.train()
	
	# Prepare file_list
	train_list = os.listdir('train/')
	random.shuffle(train_list)

	# sampling
	sample_num = int(sample_ratio*len(train_list))
	train_list = train_list[0:sample_num]

	loss_train = 0.0
	f1 = 0.0
	for train_file in train_list:
		feats, labels = loader.load_graph(file_path='train/'+train_file)
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(feats)

		loss = Myloss(output, labels)
		f1_train = f1_score(output, labels)

		loss.backward()
		optimizer.step()

		loss_train += loss.item()/len(train_list)
		f1 += f1_train/len(train_list)

	loss_val, f1_val = evaluate(model, save=False)

	if (epoch%1 == 0 or epoch==0):
		print('Epoch: {:04d} | '.format(epoch+1),
		'loss_train: {:.4f} | '.format(loss_train),
		'f1_train: {:.4f} | '.format(f1_train),
		'loss_val: {:.4f} | '.format(loss_val),
		'f1_val: {:.4f} | '.format(f1_val),
		'time: {:.4f}s'.format(time.time() - t))

	return [loss_train, loss_val], [f1_train, f1_val]


def evaluate(model, file_path='valid_query/', save_path=None, save=False):
	model.eval()

	valid_list = os.listdir(file_path)

	loss_val = 0.0
	f1_val = 0.0
	for valid_file in valid_list:
		feats, labels = loader.load_graph(file_path=file_path + valid_file)
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(feats)

		loss = Myloss(output, labels)
		f1_valid = f1_score(output, labels)

		loss_val += loss.item()/len(valid_list)
		f1_val += f1_valid/len(valid_list)

		if save:
			save_prediction(output, 0.5, file_path + valid_file, save_path, device, mode)

	return loss_val, f1_val

def plot_train_curve(train_data, valid_data, pretitle='Loss graph', argument='loss'):
	plt.figure()
	plt.plot(list(range(len(train_data))), train_data)
	plt.plot(list(range(len(valid_data))), valid_data)
	plt.title(pretitle)
	plt.legend(['Train ' + argument, 'Valid '+ argument])
	plt.show()

###################################### DOTO ######################################
# 1. make customized dataset class
# 2. make function for constructing the modified line graphs
# 3. make entire structure for training
# - for SVM
# - for simple FFN
# - for GraphSAGE
# 4. graphics.

###################################### Model Train ######################################
t_total = time.time()
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []
for epoch in range(args.epochs):
	[train_loss, val_loss],[train_f1, val_f1] = train(epoch)

	# record the performance
	train_loss_history.append(train_loss)
	val_loss_history.append(val_loss)
	train_f1_history.append(train_f1)
	val_f1_history.append(val_f1)

# plot the learning curve (loss and f1 score)
if args.plot:
	plot_train_curve(train_loss_history, val_loss_history)
	plot_train_curve(train_f1_history, val_f1_history)

print("Optimization Finished!")

print("Save the prediction.")
# _,_ = evaluate(model, file_path='valid_query/', save_path='./prediction/ffn/', mode='ffn', save=True)
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
