import numpy as np
import matplotlib.pyplot as plt
import torch
import dgl
import networkx as nx
import os 
import tqdm
from itertools import combinations

#########################################
#       FOT PREPROCESSING THE DATA      #
#########################################

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


def query_to_modified_line_graphs(file_path = './train/train_000.txt'):
    '''
    Here, we will create the modified line graphs H1, H2.
    Input (str) : path of data file (.txt)
    Output : node_dict (dictionary), edge_list_H1 (list of tuples), edge_list_H2 (list of tuples)
    '''

    # Read all lines from one query
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # ----------- NODE CREATE ----------- #
    # Both graphs have the same nodes that are correspond to each TCP connection
    # node_dict = {(src, dst): mapped_id} / keys: connection between two edges, values: continuously ordered integers (0, 1, 2, ...)
    # inward_neighbor = {dst: [list of src]} / This is for EDGE CREATE phase of H1
    # outward_neighbor = {src: [list of dst]} / This is for EDGE CREATE phase of H2

    node_dict = {}
    inward_neighbor = {}
    outward_neighbor = {}

    for line in lines:
        src, dst = line.split('\t')[0:2]
        src = int(src)
        dst = int(dst)
        if node_dict.get((src, dst)) is not None:
            continue
        else:
            node_dict[(src, dst)] = len(node_dict)
            inward_neighbor = add_to_dict(inward_neighbor, key=dst, value=[src])
            outward_neighbor = add_to_dict(outward_neighbor, key=src, value=[dst])

    # ----------- EDGE CREATE ----------- #
    # For H1, nodes are connected if they share the common source IP.
    # For H2, nodes are connected if they share the common destination IP.
    # Edges are created as a form of list of tuples (node id of H1 and H2 are follows the above created node_dict)

    edge_list_H1 = []
    edge_list_H2 = []
    
    # Create edges for H1
    for same_src in outward_neighbor.items():
        src = same_src[0]
        dst_list = same_src[1]
        
        possible_combi = list(combinations(dst_list, 2))
        
        for combi in possible_combi:
            if combi == []:
                continue
            edge1 = (src, combi[0])
            edge2 = (src, combi[1])
            edge_list_H1.append((node_dict[edge1], node_dict[edge2]))

    # Create edges for H2
    for same_dst in inward_neighbor.items():
        dst = same_dst[0]
        src_list = same_dst[1]

        possible_combi = list(combinations(src_list, 2))

        for combi in possible_combi:
            if combi == []:
                continue
            edge1 = (combi[0], dst)
            edge2 = (combi[1], dst)
            edge_list_H2.append((node_dict[edge1], node_dict[edge2]))

    return node_dict, edge_list_H1, edge_list_H2


def create_dgl_graph(node_dict, edge_list_H1, edge_list_H2):
    '''
    Create H1 and H2 as a form of dgl.graph.
    Inputs: node_dict (dictionary), edge_list_H1 (list of tuples), edge_list_H2 (list of tuples)
    Outpus: H1 (dgl.graph), H2 (dgl.graph)
    '''
    
    # Create empty DGLGraph
    H1 = dgl.DGLGraph()
    H2 = dgl.DGLGraph()

    # Add the same node sets.
    H1.add_nodes(len(node_dict))
    H2.add_nodes(len(node_dict))

    # Add proper edges
    H1.add_edges([edge[0] for edge in edge_list_H1], [edge[1] for edge in edge_list_H1])
    H2.add_edges([edge[0] for edge in edge_list_H2], [edge[1] for edge in edge_list_H2])

    return H1, H2

def line_to_feature(line:str, node_dict:dict, feats:np.array):
    '''
    Fill the feature matrix with proper processing.
    Input: line (str), node_dict (dictionary), feats (numpy array)
    Output: feats (numpy array)
    '''

    # Split the given line into src, dst, port, time
    infos = list(map(lambda x:int(x), line.split('\t')[0:4]))

    # Convert (src, dst) into node_id which is used in H1 and H2.
    ips = (infos[0], infos[1])
    node_id = node_dict[ips]

    # Process the port number and add to feats[node_id, time].
    port_num = infos[2]
    time_point = infos[3]

    # port number normalization: max 65535 / min 0 -> max 100 / min 0, (scale = order of 1e-3)
    rescale_port = port_num * 100.0/65535.0

    feats[node_id, time_point] += rescale_port

    return feats


def extract_features(file_path, node_dict):
    '''
    Extract feature that reflects the pattern and connectivity from the one TCP connection history
    Input: file_path (str), node_dict (dictionary)
    Output: feats (numpy array)
    '''
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    node_num = len(node_dict)
    feats = np.zeros((node_num, 1800))
    for line in lines:
        feats = line_to_feature(line, node_dict, feats)
    
    return feats


def load_graph(file_path='./train/train_000.txt', mode='proposed'):
    node_dict, edge_list_H1, edge_list_H2 = query_to_modified_line_graphs(file_path)
    
    if mode=='proposed':
        H1, H2 = create_dgl_graph(node_dict, edge_list_H1, edge_list_H2)
    else:
        (H1, H2) = (None, None)

    feats = extract_features(file_path, node_dict)
    label = answer_to_tensor(file_path)

    # convert to torch.Tensor
    feats = torch.Tensor(feats)

    return H1, H2, feats, label


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


def draw_graph(graph:dgl.DGLGraph):
    nx_g = graph.to_networkx()
    pos = nx.kamada_kawai_layout(nx_g)
    nx.draw(nx_g, pos, with_labels=True, node_color=[[0.7, 0.7, 0.7]])
    plt.show()


def save_dgl_graph(graphs:list, save_dir:str='./processed', save_name:str='train_graphs.bin'):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    graph_labels = {'glabel': torch.tensor(range(len(graphs)))}
    dgl.data.utils.save_graphs(save_dir+'/'+save_name, graphs, graph_labels)


def save_answer_tensor(tensors:list, save_dir:str='./processed', save_name:str='train_ans.pt'):
    # stack tensors in a list
    save_tensor = tensors[0]
    for i in range(len(tensors)-1):
        save_tensor = torch.cat([save_tensor, tensors[i+1]], dim=0)
    
    # And save.
    torch.save(save_tensor, save_dir+'/'+save_name)


    



