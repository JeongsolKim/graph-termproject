import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from preprocess import *

def accuracy(output, labels):
	preds = output.max(1)[1].type_as(labels)
	correct = preds.eq(labels).double()
	correct = correct.sum()
	return correct / len(labels)

def f1_score(output, label, threshold=0.5):
	# We only count the attack types while ignoring the benign.
	# I designed the prediction and label vector to indicate the benign as the first entry.
	# Thus, to calculate F1 score, slicing the vector from the second entry to the end.
	attack_pred = output[0, 1:].detach().cpu().numpy()
	attack_label = label[0, 1:].detach().cpu().numpy()
	
	# Precision
	true_predict = 0
	attack_pred_positive = np.where(attack_pred>threshold)[0]
	
	for pred in attack_pred_positive:
		if attack_label[pred] > threshold:
			true_predict += 1
	prec = (1+true_predict)/(1+len(attack_pred_positive))

	# Recall
	predicted = 0
	attack_label_positive = np.where(attack_label>threshold)[0]
	for label in attack_label_positive:
		if attack_pred[label] > threshold:
			predicted += 1
	recall = (1+predicted)/(1+len(attack_label_positive))

	# F1 score
	f1 = 2*prec*recall/(prec+recall)

	return f1

def visualize(labels, g):
	pos = nx.spring_layout(g, seed=1)
	plt.figure(figsize=(8, 8))
	plt.axis('off')
	nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
					 node_color=labels, edge_color='k',
					 arrows=False, width=0.5, style='dotted', with_labels=False)

def forward_and_save_prediction(model_instance, threshold, file_path, save_path, device, mode='proposed'):
	# Load attack_dictionary
	attack_dict = load_attackDict()
	
	# Load data
	H1, H2, feats, labels = load_graph(file_path, mode)
	feats = feats.to(device)

	# Feed forward to the trained model
	if mode == 'ffn':
		output = model_instance(feats)
	
	elif mode == 'proposed':
		H1 = H1.to(device)
		H2 = H2.to(device)
		output = model_instance(H1, H2, feats)

	# Thresholding and get the list of predicted attack types
	output = output.detach().cpu().numpy()
	pred = np.where(output>threshold)[1]

	attack_list = []
	for pred_id in pred:
		# Benign attack ignore.
		if pred_id == 0:
			continue
		else:
			attack_list += [att_type for att_type, att_id in attack_dict.items() if att_id == pred_id]

	# save
	save_name = save_path+file_path.split('.txt')[0] + '_prediction.txt'
	with open(save_name, 'w') as f:
		for i, att in enumerate(attack_list):
			if i < len(attack_list)-1:
				f.write(att+'\t')
			elif i == len(attack_list)-1:
				f.write(att)
