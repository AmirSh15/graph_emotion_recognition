import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
from torch.nn import init
from scipy.linalg import fractional_matrix_power
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

def Comp_degree(A):
    """ compute degree matrix of a graph """
    out_degree = torch.sum(A, dim=0)
    in_degree = torch.sum(A, dim=1)

    diag = torch.eye(A.size()[0]).cuda()

    degree_matrix = diag*in_degree + diag*out_degree - torch.diagflat(torch.diagonal(A))

    return degree_matrix


class GraphConv_kipf(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=128):
        super(GraphConv_kipf, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.weight = nn.Parameter(
        #         torch.FloatTensor(in_dim, out_dim))
        # self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        # init.xavier_uniform_(self.weight)
        # init.constant_(self.bias, 0)

        self.MLP = MLP(num_layers, in_dim, hidden_dim, out_dim)
        self.batchnorm = nn.BatchNorm1d(out_dim)

        for i in range(num_layers):
            init.xavier_uniform_(self.MLP.linears[i].weight)
            init.constant_(self.MLP.linears[i].bias, 0)


    def forward(self, features, A):
        b, n, d = features.shape
        assert(d==self.in_dim)
        A_norm = A + torch.eye(n).cuda()
        deg_mat = Comp_degree(A_norm)
        frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.cpu(),
                                                                -0.5)).cuda()
        # frac_degree = torch.FloatTensor(fractional_matrix_power(deg_mat.cpu().detach().numpy(),
        #                                                         -0.5)).cuda()

        A_hat = torch.matmul(torch.matmul(frac_degree
            , A_norm), frac_degree)


        repeated_A =A_hat.repeat(b,1,1)
        agg_feats = torch.bmm(repeated_A, features)
    
        # out = torch.einsum('bnd,df->bnf', (agg_feats, self.weight))
        # out = out + self.bias
        out = self.MLP(agg_feats.view(-1, d))
        out = self.batchnorm(out).view(b, -1, self.out_dim)

        return out 

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

        self.GCN_1 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 128) / 4), 128)
        self.GCN_2 = GraphConv(self.num_layers, self.input_dim, int((self.input_dim + 64) / 4), 64)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim=0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).cuda()])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim=1)[0]
        return pooled_rep

    def forward(self, A, h, padded_neighbor_list):
        b,c,d = h.shape

        out_1 = self.maxpool(h.view(-1, d), padded_neighbor_list).view(h.shape)
        out_2 = self.GCN_1(h, A)
        out_3 = self.GCN_2(h, A)

        out = torch.cat((out_1, out_2, out_3), dim=2 )
        
        return out



class Graph_Inception(nn.Module):
    def __init__(self, num_layers, input_dim, output_dim,
                 final_dropout,
                 device, dataset,batch_size, num_nodes, A):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            device: which device to use
        '''

        super(Graph_Inception, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.dataset = dataset
        self.batch_size = batch_size

        ###Adj matrix
        self.Adj = torch.nn.Parameter(torch.rand([num_nodes,num_nodes]), requires_grad=True)
        # self.Adj = torch.nn.Parameter(A, requires_grad=False)

        ###Pool matrix
        self.Pool = torch.nn.Parameter(torch.ones(size=([num_nodes])), requires_grad=True)


        ###List of Inception layers
        self.Inceptions = torch.nn.ModuleList()
        c = 0
        for i in range(self.num_layers):
            self.Inceptions.append(Inception_layer(input_dim+c))
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


    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1,-1,graph.node_features.shape[1]) for graph in batch_graph], 0).to(self.device)
        A = F.relu(self.Adj)

        B, N, D = X_concat.shape

        X_concat = X_concat.view(-1, D)
        X_concat = self.bn0(X_concat)
        X_concat = X_concat.view(B, N, D)


        padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)

        h = X_concat
        for layer in self.Inceptions:
            h = layer(A, h, padded_neighbor_list)

        max_pool,ind = torch.max(h,dim=1)
        min_pool = torch.min(h,dim=1)[0]
        mean_pool = torch.mean(h,dim=1)
        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1)
        pooled = torch.cat((max_pool, weighted_pool, mean_pool), dim=1)

        score = self.classifier(pooled)

        return score, ind



class Graph_CNN_kipf(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device, dataset,batch_size):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(Graph_CNN_kipf, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.dataset = dataset
        self.batch_size = batch_size

        ###Adj matrix
        self.Adj = torch.nn.Parameter(torch.rand([90,90]), requires_grad=False)

        ###Pool matrix
        self.Pool = torch.nn.Parameter(torch.ones(size=([90])), requires_grad=False)

        ###List of GCN layers
        self.GCN_1 = GraphConv_kipf(input_dim, 128)
        self.GCN_2 = GraphConv_kipf(128, 128)

        ###List of pooling layers

        ###List of batchnorms
        self.bn0 = nn.BatchNorm1d(136, affine=False)
        

        #Linear functions that maps the hidden representations to labels
        self.classifier = nn.Sequential(
                            nn.Linear(1*128, 512),
                            nn.Dropout(p=self.final_dropout),
                            nn.PReLU(512),
                            nn.Linear(512, output_dim))



    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features.view(1,-1,136) for graph in batch_graph], 0).to(self.device)
        # A = F.relu(self.Adj)
        
        A = np.zeros([90, 90])
        Num_hop = 1
        for i in range(A.shape[0]):
            # A[i, i] = 1
            for j in range(A.shape[0]):
                if (i - j <= Num_hop) and (i - j > 0):
                    A[i, j] = 1
                    A[j, i] = 1
        A = torch.FloatTensor(A).to(self.device)

        B, N, D = X_concat.shape

        # X_concat = X_concat.view(-1, D)
        # X_concat = self.bn0(X_concat)
        # X_concat = X_concat.view(B, N, D)

        h = F.relu(self.GCN_1(X_concat, A))
        h = F.softmax(self.GCN_2(h, A))

        # max_pool = torch.max(h,dim=1)[0]
        # min_pool = torch.min(h,dim=1)[0]
        # mean_pool = torch.mean(h,dim=1)
        repeated_pool = self.Pool.repeat(h.shape[0], 1, 1)
        weighted_pool = torch.bmm(repeated_pool, h).view(h.shape[0],-1)
        # pooled = torch.cat((max_pool, min_pool, mean_pool), dim=1)
        pooled = weighted_pool

        score = self.classifier(pooled)

        return score