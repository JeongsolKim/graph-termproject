import numpy as np
import matplotlib.pyplot as plt
import torch
import dgl
import networkx as nx
import os 
import tqdm
from utils import *
from itertools import combinations

#########################################
#   FOT PREPROCESSING & LOAD THE GRAPH  #
#########################################

class LineGraphLoader():
	def __init__(self):
		pass

	def load_graph(self, file_path='./train/train_000.txt', c=1.0):
		# Read all lines from one query
		with open(file_path, 'r') as f:
			lines = f.readlines()
		
		# ----------- NODE CREATE ----------- #
		# Both graphs have the same nodes that are correspond to each TCP connection
		# node_dict = {(src, dst): mapped_id} / keys: connection between two edges, values: continuously ordered integers (0, 1, 2, ...)
		# edge_list = [(src as mapped_id, dst as mapped_id)] / list of tuples. 

		node_dict = {}
		edge_dict = {}
		edge_list_not_unique = []
		for line in lines:
			src, dst = line.split('\t')[0:2]
			src = int(src)
			dst = int(dst)
			
			if node_dict.get(src) is None:
				node_dict[src] = len(node_dict)
			if node_dict.get(dst) is None:
				node_dict[dst] = len(node_dict)

			if edge_dict.get((node_dict[src], node_dict[dst])) is None:
				edge_dict[(node_dict[src], node_dict[dst])] = len(edge_dict)

		# ----------- dgl.DGLGraph() CREATE ----------- #
		g = dgl.DGLGraph()
		g.add_nodes(len(node_dict))
		g.add_edges([edge[0] for edge in edge_dict], [edge[1] for edge in edge_dict])
		
		# ----------- Extract featrues and add to graph ----------- #
		feats = self.extract_features(file_path, node_dict, edge_dict, c)
		feats = torch.Tensor(feats)
		g.edata['feat'] = feats

		# ----------- Convert to line graph ----------- #
		gl = g.line_graph(backtracking=False, shared=True)
		feats = gl.ndata['feat']

		# ----------- Extract corresponding label ----------- #
		label = answer_to_tensor(file_path)

		return gl, feats, label

	def extract_features(self, file_path, node_dict, edge_dict, c):
		with open(file_path, 'r') as f:
			lines = f.readlines()
		
		edge_num = len(edge_dict)
		edge_feats = np.zeros((edge_num, 1800))
		for line in lines:
			edge_feats = self.line_to_feature(line, node_dict, edge_dict, edge_feats, c)
		
		return edge_feats

	def line_to_feature(self, line:str, node_dict:dict, edge_dict:dict, feats:np.array, c):
	    '''
	    Fill the feature matrix with proper processing.
	    Input: line (str), edge_dict (dictinary), feats (numpy array)
	    Output: feats (numpy array)
	    '''

	    # Split the given line into src, dst, port, time
	    infos = list(map(lambda x:int(x), line.split('\t')[0:4]))

	    # Convert (src, dst) into node_id which is used in H1 and H2.
	    edge = (node_dict[infos[0]], node_dict[infos[1]])
	    edge_id = edge_dict[edge]

	    # Process the port number and add to feats[node_id, time].
	    port_num = infos[2]
	    time_point = infos[3]

	    # port number normalization: max 65535 / min 0 -> max 100 / min 0, (scale = order of 1e-3)
	    rescale_port = port_num * c/65535.0

	    feats[edge_id, time_point] += rescale_port

	    return feats


class ModifiedLineGraphLoader():
	def __init__(self):
		pass 

	def query_to_modified_line_graphs(self, file_path = './train/train_000.txt'):
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


	def create_dgl_graph(self, node_dict, edge_list_H1, edge_list_H2):
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

	def line_to_feature(self, line:str, node_dict:dict, feats:np.array, c):
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
		rescale_port = port_num * c/65535.0

		feats[node_id, time_point] += rescale_port

		return feats

	def extract_features(self, file_path, node_dict, c):
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
			feats = self.line_to_feature(line, node_dict, feats, c)
		
		return feats

	def load_graph(self, file_path, mode='proposed', c=1.0):
		node_dict, edge_list_H1, edge_list_H2 = self.query_to_modified_line_graphs(file_path)
		
		if mode=='proposed':
			H1, H2 = self.create_dgl_graph(node_dict, edge_list_H1, edge_list_H2)
		else:
			(H1, H2) = (None, None)

		feats = self.extract_features(file_path, node_dict, c)
		label = answer_to_tensor(file_path)

		# convert to torch.Tensor
		feats = torch.Tensor(feats)

		return H1, H2, feats, label

class FeatureLabelLoader(ModifiedLineGraphLoader):
	def __init__(self):
		super(FeatureLabelLoader, self).__init__()
		pass
	
	def load_graph(self, file_path, c=1.0):
		node_dict, _ , _ = self.query_to_modified_line_graphs(file_path)
		feats = self.extract_features(file_path, node_dict, c)
		label = answer_to_tensor(file_path)

		# convert to torch.Tensor
		feats = torch.Tensor(feats)

		return feats, label



