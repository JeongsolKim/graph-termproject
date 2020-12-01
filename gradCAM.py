import dgl
import torch
import torch.nn.functional as F
import os
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from graphloaders import *
from model import *
from utils import *

def get_output_with_label(file_path, model_name, device, c, model_load_path):
	warnings.filterwarnings('ignore')
		
	# Data loader and Model
	if model_name == 'ffn':
		loader = FeatureLabelLoader()
		model = SimpleFFN(in_feats=1800, nhid=256, num_classes=25).to(device)
	elif model_name == 'sage_on_line':
		loader = LineGraphLoader()
		model = MyModel_line(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='mean').to(device)
	elif model_name == 'proposed':
		loader = ModifiedLineGraphLoader()
		model = MyModel(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='mean').to(device)

	# Load
	ckpt = torch.load(model_load_path)
	model.load_state_dict(ckpt['model_state_dict'])
	model.eval()

	# Initialize
	class_num = 25
	Myloss = nn.BCELoss().to(device)

	# Inference
	if model_name == 'proposed':
		H1, H2, feats, labels = loader.load_graph(file_path=file_path, c=c)
		H1 = H1.to(device)
		H2 = H2.to(device)
		feats = feats.to(device)
		feats.requires_grad = True
		labels = labels.to(device)
		output = model(H1, H2, feats)
	elif model_name == 'sage_on_line':
		gl, feats, labels = loader.load_graph(file_path=file_path, c=c)
		gl = gl.to(device)
		feats = feats.to(device)
		feats.requires_grad = True
		labels = labels.to(device)
		output = model(gl, feats)
	elif model_name == 'ffn':
		feats, labels = loader.load_graph(file_path=file_path, c=c)
		feats = feats.to(device)
		feats.requires_grad = True
		labels = labels.to(device)
		output = model(feats)
	else:
		raise NameError

	return feats, output, labels

def gradcam(file_path, model_name, device, c, model_load_path):
	att_dict = load_attackDict()
	
	feats, output, labels = get_output_with_label(file_path, model_name, device, c, model_load_path)
	pred_att = [find_keys_by_value(att_dict, x)[0]+'({}'.format(x)+', %.4f)' %output[0, x] \
		for x in np.where(to_np(output)>0.5)[1] if x>0]
	label_att = [find_keys_by_value(att_dict, x)[0]+'({})'.format(x) \
		for x in np.where(to_np(labels)>0.5)[1] if x>0]

	title = 'Prediction : '+ ' / '.join(pred_att) + '\nLabel : '+' / '.join(label_att)

	alpha_c = []
	for i in range(output.size()[-1]):
		output[0,i].backward(retain_graph=True)
		alpha_c.append(feats.grad.mean(0))
	alpha_c = torch.stack(alpha_c)
	
	cam = F.relu(torch.mm(alpha_c, feats.transpose(0,1)))
	
	fig, ax = plt.subplots()
	im = plt.imshow(to_np(cam), cmap='hot', vmin=0, vmax=1)
	divider = make_axes_locatable(ax)
	cax = divider.new_vertical(size="50%", pad=0.3, pack_start=True)
	fig.add_axes(cax)
	fig.colorbar(im, cax=cax, orientation="horizontal")
	plt.tight_layout()
	plt.savefig('./gradcam.png', bbox_inches='tight')

	print('\n>> Prediction result')
	print(title)
	print('>> Grad-CAM image is saved.')