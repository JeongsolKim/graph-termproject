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
    infos = line.split('\t')

    # Convert (src, dst) into node_id which is used in H1 and H2.
    ips = (infos[0], infos[1])
    node_id = node_dict[ips]

    # Process the port number and add to feats[node_id, time].
    port_num = int(infos[2])
    time_point = int(infos[3])

    # port number normalization: max 65535 / min 0 -> max 100 / min 0, (scale = order of 1e-3)
    rescale_port = port_num * 100.0/65535.0

    feats[node_id, time_point] += rescale_port

    return feats


def create_node_feature(node_dict, H1:dgl.DGLGraph, H2:dgl.DGLGraph):
    pass

def query_to_graph(file_path = './train/train_000.txt'):
    '''
    This function returns a DGLGraph made from a train/valid/test query data (.txt file -> DGLGraph).
    Nodes: ordered ips in a given file.
    Edges: Connections between scr node and dest node in a given file.
    Node features: None
    Edge features: information about time and port number.
    '''
    # Create empty DGLGraph
    graph = dgl.graph()

    # Read all lines
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # ----------- NODE CREATE ----------- #
    # mapping src and dest ips into continuously ordered integers (0, 1, 2, ...)
    ip_dict = {}
    for line in lines:
        src, dest = line.split('\t')[0:2]
        ip_dict = add_to_dict(ip_dict, src)
        ip_dict = add_to_dict(ip_dict, dest)

    sorted_ip = sorted(ip_dict)
    sorted_ip = sorted(list(map(lambda x: int(x), sorted_ip)))
    sorted_ip = list(map(lambda x: str(x), sorted_ip))

    # Mapping ips to node ids.
    ip_dict ={}
    for i, ip in enumerate(sorted_ip):
        ip_dict[ip] = i

    # Add nodes to DGLGraph
    graph.add_nodes(len(ip_dict))

    # ----------- EDGE CREATE ----------- #
    edge_dict = {}
    for line in lines: 
        infos = line.split('\t')
        ips = '\t'.join(infos[0:2])

        # add str ips: (port, time)
        edge_dict = add_to_dict(edge_dict, key=ips, value=[(infos[2], infos[3])])


    for edge in edge_dict:
        # get a pair of nodes (ips) of each edge and convert to node id.
        src_node =ip_dict[edge.split('\t')[0]]
        dest_node = ip_dict[edge.split('\t')[1]]
        
        # process the port and time information
        # this will be a feature vector for an edge.
        temp_time = torch.zeros(1, 1800)
        for pt in edge_dict[edge]:
            # pt[0]: port / pt[1]: time

            # port number normalization: max 65535 / min 0 -> max 100 / min 0, (scale = order of 1e-3)
            rescale_port = float(pt[0]) * 100.0/65535.0
            # one-hot time * rescaled port num
            temp_time[0, int(pt[1])] += rescale_port

        # Add edges to DGLGraph (one by one)
        graph.add_edges(src_node, dest_node, data={'feature':temp_time})
    
    return graph

def answer_to_tensor(file_path = './train/train_000.txt'):
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
        with open(file_path, 'r') as f:
            lines = f.readlines()

        if lines == []:
            lines = ['-']
        
        for ans in lines[0].split('\t'):
            attacks.append(ans.strip())

    ans_binary = convert_to_binary_vec(attacks, att_dict)
    return ans_binary

def preprocess(data='train', verbose:bool=True):
    
    '''
    This function will preprocess the raw data (.txt file).
    1) It will convert train, valid and test queries (or histories) into DGLGraph and save it.
    2) It also create answer file such that make it easier to make binary labels for training.
    3) Finally, it will convert answers (attack types for each file) into binary torch.Tensor and save it.
    return : None

    If there are already processed data, it will do nothing.
    All the processed files are saved in './processed' directory.
    '''

    assert data in ['train', 'valid', 'test'], print('Data should be one of train, valid or test, not {}.'.format(data))

    if data == 'train':
        query_dir = './train'
        answer_dir = './train'
    elif data == 'valid':
        query_dir = './valid_query'
        answer_dir = './valid_answer'
    elif data == 'test':
        query_dir = './test_query'
        answer_dir = None

    if not os.path.isdir('./processed'):
        os.mkdir('./processed')

    # 0. parameter setting.
    data_num = len(os.listdir(query_dir))
    if verbose: print('>> [Preprocess] Convert %s data query into graph.' %data)
    graphs_per_bin = 50
    q_name = query_dir.split('/')[-1]
    a_name = answer_dir.split('/')[-1] if answer_dir else ''

    # 1. Check there already exists processed files.
    all_graphs_exist = True
    for i in range(int(data_num/graphs_per_bin) + 1):
        if not os.path.isfile('./processed/{}_graphs_'.format(data)+str(i+1)+'.bin'):
            all_graphs_exist = False
            break

    all_answers_exist = True
    for i in range(int(data_num/graphs_per_bin) + 1):
        if not os.path.isfile('./processed/{}_answers_'.format(data)+str(i+1)+'.pt'):
            all_answers_exist = False
            break
    
    # 2. Depends on the existance of files, do proper preprocessing.
    if all_graphs_exist and all_answers_exist and verbose:
        print('>> [Preprocess] {} graph data and answer data exist.'.format(data))
    else:
        for sec_num in range(int(data_num/graphs_per_bin) + 1):
            graphs = []
            answers = []
            if verbose:
                print('>> [Preprocess] Convert %d data... (%02d/%02d)' %(graphs_per_bin, sec_num+1, data_num/graphs_per_bin+1))
                for file_num in tqdm.tqdm(range(min(graphs_per_bin, data_num-sec_num*graphs_per_bin))):
                    if not all_graphs_exist:
                        graph = query_to_graph(query_dir+'/{}_'.format(q_name)+str('%03d' %(file_num+sec_num*graphs_per_bin) +'.txt'))
                        graphs.append(graph)
                        pass
                    if not all_answers_exist and answer_dir:  # do nothing for test dataset. (there is no answer file)
                        answer = answer_to_tensor(answer_dir+'/{}_'.format(a_name)+str('%03d' %(file_num+sec_num*graphs_per_bin) +'.txt'))
                        answers.append(answer)
            else:
                for file_num in range(min(graphs_per_bin, data_num-sec_num*graphs_per_bin)):
                    if not all_graphs_exist:
                        graph = query_to_graph(query_dir+'/{}_'.format(q_name)+str('%03d' %(file_num+sec_num*graphs_per_bin) +'.txt'))
                        graphs.append(graph)
                    
                    if not all_answers_exist and answer_dir:
                        answer = answer_to_tensor(answer_dir+'/{}_'.format(a_name)+str('%03d' %(file_num+sec_num*graphs_per_bin) +'.txt'))
                        answers.append(answer)

            if not all_graphs_exist:
                save_dgl_graph(graphs, save_name='{}_graphs_'.format(data)+str(sec_num+1)+'.bin')
                pass
            if not all_answers_exist and answer_dir:
                save_answer_tensor(answers, save_name='{}_answers_'.format(data)+str(sec_num+1)+'.pt')

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


    



