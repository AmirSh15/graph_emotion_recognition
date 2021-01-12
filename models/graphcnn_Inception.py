import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from torch.nn import init
sys.path.append("models/")
from mlp import MLP

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

class GraphConv(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)

    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)

        repeated_A =A.repeat(b,1,1)
        agg_feats = torch.bmm(repeated_A, features)

        out = self.MLP(agg_feats.view(-1,d))
        out = F.relu(self.batchnorm(out)).view(b,-1,self.out_dim)

        return out



class Inception_layer(nn.Module):
    def __init__(self, input_dim):
        super(Inception_layer, self).__init__()
        self.input_dim = input_dim
        self.num_layers = 2
        # self.GCN_1 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim+128)/4), 128)
        # self.GCN_2 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim+128)/4), 128)
        # self.GCN_3 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim+128)/4), 128)
        # self.GCN_4 = GraphConv(self.num_layers, 128, int((128+64)/4), 64)
        # self.GCN_5 = GraphConv(self.num_layers, 128, int((128+32)/4), 32)
        # self.GCN_6 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim+128)/4), 128)

        self.GCN_4 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 64) / 4), 64)
        self.GCN_5 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 128) / 4), 128)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).cuda()])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def forward(self, A, h, padded_neighbor_list):
        b,c,d = h.shape
        # out_1 = self.GCN_6(self.maxpool(h.view(-1,d), padded_neighbor_list).view(h.shape), A)
        # out_2 = self.GCN_5(self.GCN_1(h, A),A)
        # out_3 = self.GCN_4(self.GCN_2(h, A),A)
        # out_4 = self.GCN_3(h, A)
        # out = torch.cat((out_1, out_2, out_3, out_4), dim=2 )

        out_1 = self.maxpool(h.view(-1, d), padded_neighbor_list).view(h.shape)
        out_2 = self.GCN_5(h, A)
        out_3 = self.GCN_4(h, A)
        # out_4 = self.GCN_3(h, A)
        out = torch.cat((out_1, out_2, out_3), dim=2 )
        
        return out



class Graph_Inception(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim, final_dropout,
                 device, dataset, batch_size):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(Graph_Inception, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.dataset = dataset
        self.batch_size = batch_size

        ###List of Inception layers
        self.Inceptions = torch.nn.ModuleList()
        c = 0
        for i in range(self.num_layers):
            self.Inceptions.append(Inception_layer(input_dim + c))
            c += 192

        ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(input_dim, affine=False)
        

        #Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
                            nn.Linear((c+input_dim)*3, 512),
                            nn.Dropout(p=self.final_dropout),
                            nn.PReLU(512),
                            nn.Linear(512, output_dim))


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def forward(self, batch_graph, A, batch_size=256):
        X_concat = torch.cat([graph.node_features.view(1,-1,136) for graph in batch_graph], 0).to(self.device)

        B, N, D = X_concat.shape

        X_concat = X_concat.view(-1, D)
        X_concat = self.bn0(X_concat)
        X_concat = X_concat.view(B, N, D)


        padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)

        h = X_concat
        for layer in self.Inceptions:
            h = layer(A, h, padded_neighbor_list)

        max_pool = torch.max(h,dim=1)[0]
        min_pool = torch.min(h,dim=1)[0]
        mean_pool = torch.mean(h,dim=1)
        pooled = torch.cat((max_pool, min_pool, mean_pool), dim=1)

        score = self.classifier(pooled)

        return score
