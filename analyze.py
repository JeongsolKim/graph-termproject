import dgl
import torch
import os
import warnings
import argparse
import matplotlib.pyplot as plt
from graphloaders import *
from model import *
from utils import *
from sklearn import metrics

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

# Initialize
class_num = 25
valid_list = os.listdir('valid_query/')
y_true = torch.tensor(np.zeros([len(valid_list), class_num])).to(device)
y_pred = torch.tensor(np.zeros([len(valid_list), class_num])).to(device)

# Inference
for i, valid_file in enumerate(valid_list):
	if args.model == 'proposed':
		H1, H2, feats, labels = loader.load_graph(file_path= 'valid_query/' + valid_file)
		H1 = H1.to(device)
		H2 = H2.to(device)
		feats = feats.to(device)
		output = model(H1, H2, feats)
		labels = labels.to(device)
		y_true[i, :] = labels
		y_pred[i, :] = output

	elif args.model == 'sage_on_line':
		gl, feats, labels = loader.load_graph(file_path= 'valid_query/' + valid_file)
		gl = gl.to(device)
		feats = feats.to(device)
		labels = labels.to(device)
		output = model(gl, feats)
		y_true[i, :] = labels
		y_pred[i, :] = output

	elif args.model == 'ffn':
		feats, labels = loader.load_graph(file_path= 'valid_query/' + valid_file)
		feats = feats.to(device)
		labels = labels.to(device)
		output = model(feats)
		y_true[i, :] = labels
		y_pred[i, :] = output

# ROC curve
y_true = y_true.cpu().detach().numpy()
y_pred = y_pred.cpu().detach().numpy()

fpr = {}
tpr = {}
roc_auc = {}
for i in range(class_num):
	fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
	roc_auc[i] = metrics.auc(fpr[i], tpr[i])


if not os.path.exists(args.analyze_save_path):
	os.makedirs(args.analyze_save_path, exist_ok=True)

# Plot of a ROC curve for a specific class
for i in range(class_num):
	plt.figure()
	plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.savefig(args.analyze_save_path+'class_{}.png'.format(i+1))
	