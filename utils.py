import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import os

def to_np(tensor):
    return tensor.cpu().detach().numpy()

def add_to_dict(ref_dict, key, value:list=None):
	if not value:
		if ref_dict.get(key):
			ref_dict[key] += 1
		else:
			ref_dict[key] = 1
	else:
		if ref_dict.get(key):
			ref_dict[key] += value
		else:   
			ref_dict[key] = value
		
	return ref_dict

def create_attackList_file(file_path='./train/'):
	file_num = len(os.listdir(file_path))
	assert file_num > 0, print('The given directory is empty. Check whether the path is valid.')

	att_dict = {}
	for i in range(file_num):
		try:
			with open(file_path + 'train_'+str('%03d' %i) +'.txt', 'r') as f:
				lines = f.readlines()
				for line in lines:
					sp = line.split('\t')[-1].split('\n')[0]
					att_dict = add_to_dict(att_dict, sp)
		except:
			pass
	
	# sorting
	sort_dict = sorted(att_dict.items(), reverse=False)

	# save
	with open('attacks.txt', 'w') as f:
		for item in sort_dict:
			f.write(item[0]+'\n')

def load_attackDict(file_path='attacks.txt'):
	'''
	If attacks.txt file does not exist, load training data set and create it.
	If attacks.txt exists, load it as dictionary.
	'''

	# print('>> [Preprocess] Load attacks.txt file.')

	if os.path.isfile(file_path):
		# make attack types to vector
		with open(file_path, 'r') as f:
			lines = f.readlines()

		att_vec = list(map(lambda x: x.split('\n')[0], lines))
		att_dict ={}
		for i, att in enumerate(att_vec):
			att_dict[att] = i
		
	else:
		# If the given file does not exist, create it and return.
		print('>> [Preprocess] attacks.txt does not exist. Create it and load.')
		create_attackList_file(file_path='./train/')
		att_dict = load_attackDict()

	return att_dict

# Convert list to binary torch.Tensor
def convert_to_binary_vec(value, ref_dict):
	# the length of the binary vector is same the number of items in ref_dict.
	length = len(ref_dict)
	binary_vec = torch.zeros(size=[1, length])

	if not isinstance(value, list):
		# if value is one object, convert it to list of length 1
		# to allow the for loop. (iterative)
		value = [value]
	else:
		# if value is multiple, remove the repeated one 
		# because the output is binary vector.
		value = list(set(value))

	for v in value:
		try:
			binary_vec[0, ref_dict[v]] = 1
		except:
			raise KeyError('No match key in ref_dict. (utils/convert_to_binary_vec)')
	
	return binary_vec

def answer_to_tensor(file_path):
	'''
	This function returns a binary torch.Tensor 
	which indicates the attack types included in the given file.
	'''
	# Load attack dictionary
	att_dict = load_attackDict()
	
	# read attack types in a given file and convert to binary tensor.
	file_name = os.path.split(file_path)[-1]
	attacks = []
	if file_name.startswith('train'):
		with open(file_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			attacks.append(line.split('\t')[-1].strip())  # append attack type to a list
	
	else: # in the case of valid_answer
		number = file_name.split('_')[-1]
		ans_path = './valid_answer/valid_answer_' + number
		with open(ans_path, 'r') as f:
			lines = f.readlines()

		if lines == []:
			lines = ['-']
		
		for ans in lines[0].split('\t'):
			attacks.append(ans.strip())

	ans_binary = convert_to_binary_vec(attacks, att_dict)

	return ans_binary

def visualize(labels, g):
	pos = nx.spring_layout(g, seed=1)
	plt.figure(figsize=(8, 8))
	plt.axis('off')
	nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
					 node_color=labels, edge_color='k',
					 arrows=False, width=0.5, style='dotted', with_labels=False)

def save_prediction(output, threshold, file_path, save_path, device):
	attack_dict = load_attackDict()

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
	save_name = save_path + os.path.split(file_path)[-1].split('.txt')[0] + '_prediction.txt'
	if not os.path.exists(os.path.split(save_name)[0]):
		os.mkdir(os.path.split(save_name)[0])
		
	with open(save_name, 'w') as f:
		for i, att in enumerate(attack_list):
			if i < len(attack_list)-1:
				f.write(att+'\t')
			elif i == len(attack_list)-1:
				f.write(att)


def data_distribution(file_dir='./train', num_classes=25):
	# Load attack dict
	att_dict = load_attackDict()
	dist = {att_type:0 for att_type in att_dict}

	files = os.listdir(file_dir)
	for fname in files:
		full_path = os.path.join(file_dir, fname)
		with open(full_path, 'r') as f:
			lines = f.readlines()
		for line in lines:
			attack_type = line.split('\t')[-1].strip()
			dist[attack_type] += 1

	return dist
