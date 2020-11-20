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
parser.add_argument('--plot', type=bool, default=False, help='Draw learning curve after training.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
	device = 'cuda'
else:
	device = 'cpu'

# Load model
model = MyModel(in_dim=1800, hidden_dim=512, num_classes=25, aggregator='mean', activation='sigmoid').to(device)

# Load optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Set loss
Myloss = nn.BCELoss().to(device)

def train(epoch):
	t = time.time()

	# Training phase
	model.train()
	
	# Prepare file_list
	train_list = os.listdir('train/')
	random.shuffle(train_list)

	loss_train = 0.0
	f1 = 0.0
	for train_file in train_list:
		H1, H2, feats, labels = load_graph(file_path='train/'+train_file, mode='proposed')
		H1.to(device)
		H2.to(device)
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(H1, H2, feats)

		loss = Myloss(output, labels)
		f1_train = f1_score(output, labels)

		loss.backward()
		optimizer.step()

		loss_train += loss.item()/len(train_list)
		f1 += f1_train/len(train_list)

	loss_val, f1_val = evaluate(model, mode='proposed', save=False)

	if (epoch%1 == 0 or epoch==0):
		print('Epoch: {:04d} | '.format(epoch+1),
		'loss_train: {:.4f} | '.format(loss_train),
		'f1_train: {:.4f} | '.format(f1_train),
		'loss_val: {:.4f} | '.format(loss_val),
		'f1_val: {:.4f} | '.format(f1_val),
		'time: {:.4f}s'.format(time.time() - t))

	return [loss_train, loss_val], [f1_train, f1_val]


def evaluate(model, file_path='valid_query/', mode='proposed', save=False):
	model.eval()

	valid_list = os.listdir(file_path)

	loss_val = 0.0
	f1_val = 0.0
	for valid_file in valid_list:
		H1, H2, feats, labels = load_graph(file_path=file_path + valid_file, mode=mode)
		H1 = H1.to(device)
		H2 = H2.to(device)
		feats = feats.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		output = model(H1, H2, feats)

		loss = Myloss(output, labels)
		f1_valid = f1_score(output, labels)

		loss_val += loss.item()/len(valid_list)
		f1_val += f1_valid/len(valid_list)

		if save:
			forward_and_save_prediction(model, 0.5, file_path+valid_file, device, mode)

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
_,_ = evaluate(model, file_path='valid/', save_path='./prediction/sage' mode='ffn', save=True)
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
