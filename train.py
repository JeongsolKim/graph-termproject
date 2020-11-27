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

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--gpu', type=int, default=0, help='Number of GPU to use for training.')
parser.add_argument('--model', type=str, default='proposed', help="One of 'ffn', 'sage_on_line', 'proposed'.")
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--sample_ratio', type=float, default=0.8, help='Fraction of used training data for each epoch (<=1).')
parser.add_argument('--plot', type=bool, default=False, help='Draw learning curve after training.')
parser.add_argument('--model_save_path', type=str, default='./model/proposed/model.pt')
parser.add_argument('--inference_save_path', type=str, default='./prediction/proposed/')

args = parser.parse_args()

# Check input parser
args.cuda = not args.no_cuda and torch.cuda.is_available()
assert args.model in ['ffn', 'sage_on_line', 'proposed'], \
	'--model should be among ffn, sage_on_line, and proposed, not {}'.format(args.model)
assert args.sample_ratio <=1 and args.sample_ratio>0,\
	'Sample ratio should be in (0, 1], not {}.'.format(args.sample_ratio)

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

	# sampling
	sample_num = int(args.sample_ratio*len(train_list))
	train_list = train_list[0:sample_num]

	loss_train = 0.0
	f1 = 0.0
	for train_file in train_list:
		if args.model == 'proposed':
			H1, H2, feats, labels = loader.load_graph(file_path='train/'+train_file)
			H1 = H1.to(device)
			H2 = H2.to(device)
			feats = feats.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			output = model(H1, H2, feats)
		
		elif args.model == 'sage_on_line':
			gl, feats, labels = loader.load_graph(file_path='train/'+train_file)
			gl = gl.to(device)
			feats = feats.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			output = model(gl, feats)

		elif args.model == 'ffn':
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
		print('>> Epoch: {:04d} | '.format(epoch+1),
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
		if args.model == 'proposed':
			H1, H2, feats, labels = loader.load_graph(file_path=file_path + valid_file)
			H1 = H1.to(device)
			H2 = H2.to(device)
			feats = feats.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			output = model(H1, H2, feats)
		
		elif args.model == 'sage_on_line':
			gl, feats, labels = loader.load_graph(file_path=file_path + valid_file)
			gl = gl.to(device)
			feats = feats.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			output = model(gl, feats)

		elif args.model == 'ffn':
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
			save_prediction(output, 0.5, file_path + valid_file, save_path, device)

	return loss_val, f1_val

def plot_train_curve(train_data, valid_data, pretitle='Loss graph', argument='loss'):
	plt.figure()
	plt.plot(list(range(len(train_data))), train_data)
	plt.plot(list(range(len(valid_data))), valid_data)
	plt.title(pretitle)
	plt.legend(['Train ' + argument, 'Valid '+ argument])
	plt.show()


###################################### Model Train ######################################
t_total = time.time()
train_loss_history = []
val_loss_history = []
train_f1_history = []
val_f1_history = []

print('>> [{}] Training starts.'.format(datetime.datetime.now()))
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

print('>> [{}] Optimization Finished!'.format(datetime.datetime.now()))

print('\n>> [{}] Save the model checkpoint.'.format(datetime.datetime.now()))
save_dir = os.path.split(args.model_save_path)[0]
if not os.path.exists(save_dir):
	os.makedirs(save_dir, exist_ok=True)

# torch.save({
# 	'epoch':epoch,
# 	'model_state_dict':model.state_dict(),
# 	'optimizer_state_dict':optimizer.state_dict(),
# 	'loss':Myloss}, args.model_save_path)

print('>> [{}] Inference and save the results.'.format(datetime.datetime.now()))
if not os.path.exists(args.inference_save_path):
	os.makedirs(args.inference_save_path, exist_ok=True)
# _,_ = evaluate(model, file_path='valid_query/', save_path=args.inference_save_path, save=True)
print(">> Total time elapsed: {:.4f}s".format(time.time() - t_total))