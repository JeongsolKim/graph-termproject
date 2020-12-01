import dgl
import torch
import torch.nn.functional as F
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from graphloaders import *
from model import *
from utils import *

def gradcam(file_path, model_name, device, c, model_load_path):
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
	

	alpha_c = []
	for i in range(output.size()[-1]):
		output[0,i].backward(retain_graph=True)
		alpha_c.append(feats.grad.mean(0))
	alpha_c = torch.stack(alpha_c)
	
	cam = F.relu(torch.mm(alpha_c, feats.transpose(0,1)))
	
	plt.figure()
	plt.imshow(to_np(cam), cmap='hot')
	plt.colorbar()
	plt.tight_layout()
	plt.savefig('./test.png')
