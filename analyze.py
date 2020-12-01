import dgl
import torch
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from graphloaders import *
from model import *
from utils import *
from sklearn import metrics

def benign_process(y_pred):
	# Benign connection could exist in the prediction with other attacks.
	# However, in label, benign connection does not exist.
	# Thus, if there is benign prediction in y_pred with other types of attack,
	# remove the benign prediction to make proper f1-score and ROC curve.

	others_pred = y_pred[:, 1:]
	
	for i, sample in enumerate(others_pred):
		if (sample != 0).any():
			y_pred[i, 0] = 0
		elif (sample == 0).all():
			y_pred[i, 0] = 1

	return y_pred

def save_roc_curve(y_true, y_pred, class_num, analyze_save_path):
	attack_dict = load_attackDict()
	# ROC curve
	fpr = {}
	tpr = {}
	roc_auc = {}
	for i in range(class_num):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	if not os.path.exists(analyze_save_path):
		os.makedirs(analyze_save_path, exist_ok=True)

	# Plot of a ROC curve for a specific class
	for i in range(class_num):
		attack_type = [att_type for att_type, att_id in attack_dict.items() if att_id==i][0]
		save_path = os.path.join(analyze_save_path, 'class_{}: {}.png'.format(i+1, attack_type))
		plt.figure()
		plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], 'k--')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic Curve: {}'.format(attack_type))
		plt.legend(loc="lower right")
		plt.savefig(save_path)

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
	# print(attack_pred_positive, attack_label_positive, f1)
	return f1

def total_f1_score(answer_dir, pred_dir):
	# Initialize
	A = []  # the set of networks containing TCP attacks
	B = []  # the set of networks without any TCP attacks

	# Get sorted file list
	answer_files = sorted(os.listdir(answer_dir))
	pred_files = sorted(os.listdir(pred_dir))

	# Load attack dictionary
	att_dict = load_attackDict()

	# Load file pair and convert to binary vector
	files = zip(answer_files, pred_files)
	for ans_f, pred_f in files:
		attack_ans = []
		attack_pred = []

		with open(os.path.join(answer_dir, ans_f), 'r') as f:
			ans_lines = f.readlines()
		with open(os.path.join(pred_dir, pred_f), 'r') as f:
			pred_lines = f.readlines()

		if ans_lines == []:
			ans_lines = ['-']
			net_type = 'B'
		else:
			net_type = 'A'

		if pred_lines == []:
			pred_lines = ['-']

		for ans in ans_lines[0].split('\t'):
			attack_ans.append(ans.strip())
		for pred in pred_lines[0].split('\t'):
			attack_pred.append(pred.strip())
		
		ans_binary = convert_to_binary_vec(attack_ans, att_dict)
		pred_binary = convert_to_binary_vec(attack_pred, att_dict)

		# f1-score 
		f1 = f1_score(pred_binary, ans_binary)
		if net_type == 'A':
			A.append(f1)
		elif net_type == 'B':
			B.append(f1)

	# average
	total_f1 = 0.5*(np.mean(B) + np.mean(A))

	print('>> F1 score for the non-attack networks: {}'.format(np.mean(B)))
	print('>> F1 score for the attack networks: {}'.format(np.mean(A)))
	print('>> F1 score for the entire networks: {}'.format(total_f1))

	return np.mean(B), np.mean(A), total_f1

def get_auc_score(model_name, device, c, model_load_path, analyze_save_path):
	warnings.filterwarnings('ignore')
		
	# Data loader and Model
	if model_name == 'ffn':
		loader = FeatureLabelLoader()
		model = SimpleFFN(in_feats=1800, nhid=256, num_classes=25).to(device)
	elif model_name == 'sage_on_line':
		loader = LineGraphLoader()
		model = MyModel_line(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='lstm').to(device)
	elif model_name == 'proposed':
		loader = ModifiedLineGraphLoader()
		model = MyModel(in_dim=1800, hidden_dim=256, num_classes=25, aggregator='lstm').to(device)

	# Load
	ckpt = torch.load(model_load_path)
	model.load_state_dict(ckpt['model_state_dict'])
	model.eval()

	# Initialize
	class_num = 25
	file_path= './valid_query/'
	valid_list = os.listdir(file_path)
	y_true = torch.tensor(np.zeros([len(valid_list), class_num])).to(device)
	y_pred = torch.tensor(np.zeros([len(valid_list), class_num])).to(device)

	# Inference
	for i, valid_file in enumerate(valid_list):
		if model_name == 'proposed':
			H1, H2, feats, labels = loader.load_graph(file_path= os.path.join(file_path,valid_file), c=c)
			H1 = H1.to(device)
			H2 = H2.to(device)
			feats = feats.to(device)
			output = model(H1, H2, feats)
			labels = labels.to(device)
			y_true[i, :] = labels
			y_pred[i, :] = output

		elif model_name == 'sage_on_line':
			gl, feats, labels = loader.load_graph(file_path= os.path.join(file_path,valid_file), c=c)
			gl = gl.to(device)
			feats = feats.to(device)
			labels = labels.to(device)
			output = model(gl, feats)
			y_true[i, :] = labels
			y_pred[i, :] = output

		elif model_name == 'ffn':
			feats, labels = loader.load_graph(file_path= os.path.join(file_path,valid_file), c=c)
			feats = feats.to(device)
			labels = labels.to(device)
			output = model(feats)
			y_true[i, :] = labels
			y_pred[i, :] = output
		else:
			raise NameError

	y_true = y_true.cpu().detach().numpy()
	y_pred = y_pred.cpu().detach().numpy()
	return calculate_auc_score(y_true, y_pred, class_num)

def calculate_auc_score(y_true, y_pred, class_num):
	attack_dict = load_attackDict()
	fpr = {}
	tpr = {}
	auc_score = np.zeros([class_num,])

	for i in range(class_num):
		fpr[i], tpr[i], _ = metrics.roc_curve(y_true[:, i], y_pred[:, i])
		auc_score[i] = metrics.auc(fpr[i], tpr[i])

	# print('>> Area Under Curve score: \n', auc_score)

	return auc_score
	
	
	