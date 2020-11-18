import dgl
import torch
import os
import warnings
import argparse
import time
import random
import tqdm
from preprocess import *
from model import *
from utils import *

warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
	device = 'cuda'
else:
	device = 'cpu'

# Load model
model = SimpleFFN(in_feats=1800, nhid=512, num_classes=25).to(device)


# Load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set loss
Myloss = nn.BCEWithLogitsLoss().to(device)

def train(epoch):
	t = time.time()

	# Training phase
	model.train()
	
	# Prepare file_list
	train_list = os.listdir('./train/')
	random.shuffle(train_list)

	loss = 0.0
	f1 = 0.0
	for train_file in train_list:
		_, _, feats, labels = load_graph(file_path='./train/'+train_file, model='ffn')
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(feats)

		loss_train = Myloss(output, labels)
		f1_train = f1_score(output, labels)

		loss_train.backward()
		optimizer.step()

		loss += loss_train.item()/len(train_list)
		f1 += f1_train.item()/len(train_list)

	# Validation phase
	model.eval()

	valid_list = os.listdir('./valid_query/')
	random.shuffle(valid_list)

	loss_val = 0.0
	f1_val = 0.0
	for valid_file in valid_list:
		_, _, feats, labels = load_graph(file_path='./valid_query/'+valid_file, model='ffn')
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(feats)

		loss_valid = Myloss(output, labels)
		f1_valid = f1_score(output, labels)

		loss_val += loss_valid.item()/len(valid_list)
		f1_val += f1_valid.item()/len(valid_list)


	if (epoch%10 == 0 or epoch==0):
		print('Epoch: {:04d}'.format(epoch+1),
		'loss_train: {:.4f}'.format(loss_train.item()),
		'f1_train: {:.4f}'.format(f1_train.item()),
		'loss_val: {:.4f}'.format(loss_val.item()),
		'f1_val: {:.4f}'.format(f1_val.item()),
		'time: {:.4f}s'.format(time.time() - t))

	return [loss_train.item(), loss_val.item()], [f1_train.item(), f1_val.item()]

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
for epoch in tqdm.tqdm(range(args.epochs)):
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
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
