import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GCNConv, global_add_pool, \
    global_mean_pool, global_max_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from transformers import AutoModelWithLMHead, RobertaConfig, RobertaModel
import pdb

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        '''
            emb_dim (int): node embedding dimensionality
        '''
        super(GINConv, self).__init__(aggr=aggr)

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), 
                                        torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        self.bond_encoder = BondEncoder(emb_dim = emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNNComplete(nn.Module):
    """
    Wrapper for GIN/GCN/GAT/GraphSAGE
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum
        drop_ratio (float): dropout rate
        gnn_type (str): gin, gcn, graphsage, gat

    Output:
        node representations """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0., gnn_type="gin"):

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        super(GNNComplete, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))

        ###List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("not implemented.")
        return node_representation


class GNN_graphpredComplete(nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        args.num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        args.JK (str): last, concat, max or sum.
        args.graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpredComplete, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "sum":
            self.pool = global_add_pool
        elif args.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif args.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim,
                                               self.num_tasks)
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)
        return

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))
        return

    def get_graph_representation(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)

        return graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, \
                                              data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)

        return output

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #[N, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) #[N, D]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) #[N, D]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x:[B, N, D]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FingerPrintEncoder(nn.Module):
    def __init__(self, word_dim, out_dim, num_head=8, num_layer=1):
        super(FingerPrintEncoder, self).__init__()
        self.embedding = nn.Embedding(2048, word_dim)
        encoder_layer = nn.TransformerEncoderLayer(word_dim, nhead=num_head, dim_feedforward=out_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layer)
        self.pe = PositionalEncoding(word_dim)
        self.linear = nn.Linear(1024, out_dim)

    def forward(self, x):
        # === Multi-channel ===
        x = self.embedding(x.int()) # [B, N, D]
        x = self.pe(x)
        output = self.transformer(x)
        output = self.linear(output.sum(2))
    
        return output

# ======= Auto Encoder Models =========
def cosine_similarity(p, z, average=True):
    p = F.normalize(p, p=2, dim=1)
    z = F.normalize(z, p=2, dim=1)
    loss = -(p * z).sum(dim=1)
    if average:
        loss = loss.mean()
    return loss

class AutoEncoder(torch.nn.Module):

    def __init__(self, input_dim, emb_dim, loss, detach_target):
        super(AutoEncoder, self).__init__()
        self.loss = loss
        self.emb_dim = emb_dim
        self.input_dim = input_dim
        self.detach_target = detach_target

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()
        elif loss == 'cosine':
            self.criterion = cosine_similarity

        self.fc_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.input_dim),
        )
        # why not have BN as last FC layer:
        # https://stats.stackexchange.com/questions/361700/
        return

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()
        x = self.fc_layers(x)
        loss = self.criterion(x, y)

        return loss

class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()

        self.fc_mu = nn.Linear(self.input_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.input_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.input_dim),
        )
        return

    def encode(self, x):
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        if self.detach_target:
            y = y.detach()

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decoder(z)

        reconstruction_loss = self.criterion(y_hat, y)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = reconstruction_loss + self.beta * kl_loss

        return loss

def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        return 

class Multi_View_Fusion(nn.Module):
    def __init__(self, emb_dim):
        super(Multi_View_Fusion, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.Tanh(),
            nn.Linear(2*emb_dim, 1, bias=False)
        )
        self.project.apply(init_weights)

    def forward(self, reprs):
        # repr : (N, M, D)
        tmp = F.normalize(reprs)
        w = self.project(tmp).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                     # (M, 1)
        beta = beta.expand((reprs.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * reprs).sum(1), beta.squeeze(2)[0]                       # (N, D)

# class Multi_View_Fusion(nn.Module):
#     def __init__(self, emb_dim: int):
#         super(Multi_View_Fusion, self).__init__()
#         self.sqrt_dim = np.sqrt(emb_dim)        # Dim of hidden dim, which is 300 in our senario
#         self.W_q = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.W_k = nn.Linear(emb_dim, emb_dim, bias=False)
#         self.W_v = nn.Linear(emb_dim, emb_dim, bias=False)

#     def forward(self, x):
        
#         Q = self.W_q(x)
#         K = self.W_k(x)
#         V = self.W_v(x)
#         score = (torch.bmm(Q, K.transpose(1, 2)) / self.sqrt_dim).mean(0)
#         attn = F.softmax(score, -1)[0, :].unsqueeze(1)       # Take the first row as coeff
#         attn = attn.expand((Q.shape[0],) + attn.shape)          
        
#         return (attn * V).sum(1), attn.squeeze(2)[0]