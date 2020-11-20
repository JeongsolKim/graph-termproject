import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn

class SimpleFFN(nn.Module):
	def __init__(self, in_feats, nhid, num_classes, activation='relu'):
		super(SimpleFFN, self).__init__()
		self.dense1 = nn.Linear(in_feats, nhid, bias=True)
		self.dense2 = nn.Linear(nhid, nhid, bias=True)
		self.dense3 = nn.Linear(nhid, num_classes, bias=True)

	def forward(self, x):
		x = F.relu(self.dense1(x))
		x = F.relu(self.dense2(x))
		x = self.dense3(x)

		x = torch.sigmoid(torch.mean(x, dim=0, keepdim=True))

		return x


class MyModel(nn.Module):
	def __init__(self, in_dim, hidden_dim, num_classes, aggregator='mean', activation='sigmoid'):
		super(MyModel, self).__init__()
		# We make two GNN model for H1 and H2.
		self.SAGE1 = GraphSage(in_dim, hidden_dim, num_classes, aggregator, activation)
		self.SAGE2 = GraphSage(in_dim, hidden_dim, num_classes, aggregator, activation)

	def forward(self, H1, H2, feats):
		H1_rep = self.SAGE1(H1, feats)
		H2_rep = self.SAGE2(H2, feats)

		# last aggregation
		return (H1_rep+H2_rep)/2


class GraphSage(nn.Module):
	def __init__(self, in_dim, hidden_dim, out_dim, aggregator='mean', activation='sigmoid'):
		super(GraphSage, self).__init__()
		self.sageconv1 = dglnn.SAGEConv(in_feats=in_dim,
										out_feats=hidden_dim,
										aggregator_type=aggregator,
										bias=False)
		self.sageconv2 = dglnn.SAGEConv(in_feats=hidden_dim,
										out_feats=hidden_dim,
										aggregator_type=aggregator,
										bias=False)
		self.linear = nn.Linear(hidden_dim, out_dim, bias=False)
		self.activation = activation

	def forward(self, g, feats):
		feats = F.relu(self.sageconv1(g, feats))
		feats = F.relu(self.sageconv2(g, feats))
		with g.local_scope():
			g.ndata['feats'] = feats
			# Read-out
			graph_rep = dgl.mean_nodes(g, 'feats')
			# Linear
			graph_rep = self.linear(graph_rep)

			if self.activation == 'relu':
				return F.relu(graph_rep)
			elif self.activation == 'sigmoid':
				return torch.sigmoid(graph_rep)
			elif self.activation == 'softmax':
				return F.softmax(graph_rep)
			else:
				return graph_rep
