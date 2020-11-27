import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import os

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

		with open(answer_dir + ans_f, 'r') as f:
			ans_lines = f.readlines()
		with open(pred_dir + pred_f, 'r') as f:
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
	print('F1 for the non-attack networks: {}'.format(np.mean(B)))
	print('F1 for the attack networks: {}'.format(np.mean(A)))

	total_f1 = 0.5*(np.mean(B) + np.mean(A))
	return total_f1

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
	save_name = save_path+file_path.split('.txt')[0] + '_prediction.txt'
	with open(save_name, 'w') as f:
		for i, att in enumerate(attack_list):
			if i < len(attack_list)-1:
				f.write(att+'\t')
			elif i == len(attack_list)-1:
				f.write(att)


# Calculate F1-score, prec, recall. Plot ROC curve, ..
def analysis(answer_dir, predict_dir):
	test_thresholds = np.arange(0, 1, 0.05)

	for threshold in test_thresholds:
		pass
