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

def get_output_with_label(file_path, model_load_path, device, c):
	warnings.filterwarnings('ignore')
		
	# Data loader and Model
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
	H1, H2, feats, labels = loader.load_graph(file_path=file_path, c=c)
	H1 = H1.to(device)
	H2 = H2.to(device)
	feats = feats.to(device)
	feats.requires_grad = True
	labels = labels.to(device)
	output = model(H1, H2, feats)

	return feats, output, labels

def gradcam(file_path, model_load_path, device, c=1.0):
	att_dict = load_attackDict()
	threshold = 0.5
	
	feats, output, labels = get_output_with_label(file_path, model_load_path, device, c)
	pred_att = [find_keys_by_value(att_dict, x)[0]+'({}'.format(x)+', %.4f)' %output[0, x] \
		for x in np.where(to_np(output)>threshold)[1] if x>0]
	label_att = [find_keys_by_value(att_dict, x)[0]+'({})'.format(x) \
		for x in np.where(to_np(labels)>threshold)[1] if x>0]

	title = 'Prediction : '+ ' / '.join(pred_att) + '\nLabel : '+' / '.join(label_att)

	alpha_c = []
	for i in range(output.size()[-1]):
		output[0,i].backward(retain_graph=True)
		alpha_c.append(feats.grad.mean(0))
	alpha_c = torch.stack(alpha_c)
	
	cam = F.relu(torch.mm(alpha_c, feats.transpose(0,1)))

	# check feature
	pred_ids = [x for x in np.where(to_np(output)>threshold)[1] if x>0]
	most_responsible_connection = [np.argmax(to_np(cam)[x]) for x in pred_ids]

	fig, ax = plt.subplots(constrained_layout=True)
	for i, connection in enumerate(most_responsible_connection):
		plt.subplot(len(most_responsible_connection),1,i+1)
		plt.plot(np.arange(0, 1800), to_np(feats)[connection,:])
		plt.legend([pred_att[i]+', connection {}'.format(connection)])
		plt.xlabel('time [sec]')
	fig.tight_layout()
	plt.savefig('responsible_connections.png')
	plt.close()

	# Draw head-map
	fig, ax = plt.subplots()
	im = plt.imshow(to_np(cam), cmap='hot')#, vmin=0, vmax=1)

	for x,y in zip(most_responsible_connection, pred_ids):
		plt.plot(x, y ,'bo', markersize=2)
	plt.xlabel('IP Connections')
	plt.ylabel('Attack ID')
	divider = make_axes_locatable(ax)
	cax = divider.new_vertical(size="30%", pad=0.5, pack_start=True)
	fig.add_axes(cax)
	fig.colorbar(im, cax=cax, orientation="horizontal")
	plt.tight_layout()
	plt.savefig('./gradcam.png', bbox_inches='tight')

	print('\n>> Prediction result')
	print(title)
	print('>> Grad-CAM image is saved.')

	