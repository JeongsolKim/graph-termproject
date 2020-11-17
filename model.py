import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFFN(nn.Module):
    def __init__(self, in_feats, nhid, num_classes, activation='relu'):
        super(SimpleFFN, self).__init__()
        self.dense1 = nn.Linear(in_feats, nhid, bias=True)
        self.dense2 = nn.Linear(nhid, nhid, bias=True)
        self.dense3 = nn.Linear(nhid, num_classes, bias=True)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.log_softmax(self.dense3(x))

        x = torch.mean(x, dim=0, keepdim=True)

        return x


