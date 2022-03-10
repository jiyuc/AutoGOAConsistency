import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='pool')

        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='pool')

    def forward(self, graph, inputs):
        # inputs are features of nodes

        # first layer
        h = self.conv1(graph, inputs)
        # batch normalisation before activation
        # reference: https://arxiv.org/pdf/1502.03167.pdf
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)
        h = F.relu(h)  # activation of 1st layer

        # second layer
        h = self.conv2(graph, h)
        h = F.normalize(h, p=2.0, dim=1, eps=1e-12, out=None)  # batch normalisation
        h = F.relu(h)  # activation of 2nd layer
        return h


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata['score']


class GOKGModel(nn.Module):
    def __init__(self, in_features=128, hidden_features=64, out_features=16):
        super().__init__()
        # node embedding
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.h = in_features
        # edge classification
        self.pred = MLPPredictor(out_features, 5)

    def forward(self, g, x):
        self.h = self.sage(g, x)
        return self.pred(g, self.h)

    def get_last_hidden(self):
        return self.h

    @staticmethod
    def get_pre_trained():
        model = torch.load('./sage_go_kg_64dim.pt')
        return model.get_last_hidden()
