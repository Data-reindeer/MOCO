#  Ref: https://github.com/snap-stanford/pretrain-gnns/blob/master/bio/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import MessagePassing, GINEConv, GCNConv, GATConv, global_add_pool, \
    global_mean_pool, global_max_pool

from transformers import AutoModelWithLMHead, RobertaModel
import pdb


num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.

    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        aggr (str): aggreagation method

    See https://arxiv.org/abs/1810.00826 """

    def __init__(self, emb_dim, aggr="add"):
        super(GINConv, self).__init__()
        # multi-layer perceptron
        self.aggr = aggr
        self.mlp = nn.Sequential(nn.Linear(emb_dim, 2 * emb_dim),
                                 nn.ReLU(),
                                 nn.Linear(2 * emb_dim, emb_dim))
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(
            edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # Different from original concat
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


# class GCNConv(MessagePassing):

#     def __init__(self, emb_dim, aggr="add"):
#         super(GCNConv, self).__init__()

#         self.aggr = aggr
#         self.emb_dim = emb_dim
#         self.linear = nn.Linear(emb_dim, emb_dim)
#         self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
#         self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

#         nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
#         nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

#     def norm(self, edge_index, num_nodes, dtype):
#         ### assuming that self-loops have been already added in edge_index
#         edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
#                                  device=edge_index.device)
#         row, col = edge_index
#         deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

#         return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
#                           self.edge_embedding2(edge_attr[:, 1])

#         norm = self.norm(edge_index, x.size(0), x.dtype)

#         x = self.linear(x)

#         return self.propagate(self.aggr, edge_index, x=x,
#                               edge_attr=edge_embeddings, norm=norm)

#     def message(self, x_j, edge_attr, norm):
#         return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
                          self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):

        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias
        return aggr_out


# class GraphSAGEConv(MessagePassing):

#     def __init__(self, emb_dim, aggr="mean"):
#         super(GraphSAGEConv, self).__init__()

#         self.emb_dim = emb_dim
#         self.linear = nn.Linear(emb_dim, emb_dim)
#         self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
#         self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

#         nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
#         nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

#         self.aggr = aggr

#     def forward(self, x, edge_index, edge_attr):
#         # add self loops in the edge space
#         edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

#         # add features corresponding to self-loop edges.
#         self_loop_attr = torch.zeros(x.size(0), 2)
#         self_loop_attr[:, 0] = 4  # bond type for self-loop edge
#         self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
#         edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

#         edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + \
#                           self.edge_embedding2(edge_attr[:, 1])

#         x = self.linear(x)

#         return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

#     def message(self, x_j, edge_attr):
#         return x_j + edge_attr

#     def update(self, aggr_out):
#         return F.normalize(aggr_out, p=2, dim=-1)


class GNN(nn.Module):
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

        super(GNN, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr = "add")) # default aggregation is 'add'
                # self.gnns.append(GINEConv(nn=nn.Sequential(nn.Linear(emb_dim,emb_dim)), edge_dim=2)) # default aggregation is 'add'
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))

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

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

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

class GNN_graphpred(nn.Module):
    """
    For fine-tuning part
    Load pre-training parameters, and integrate different graph-level tasks into one model.  

    Args:
        args.num_layer (int): the number of GNN layers
        arg.emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        args.JK (str): last, concat, max or sum.
        args.graph_pooling (str): sum, mean, max, attention, set2set

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536 """

    def __init__(self, args, num_tasks, molecule_model=None):
        super(GNN_graphpred, self).__init__()

        if args.num_layer < 2:
            raise ValueError("# layers must > 1.")

        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        # Different kind of graph pooling
        if args.graph_pooling == "add":
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
        self.molecule_model.load_state_dict(torch.load(model_file, map_location='cuda:0'))
        return


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

class roberta(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(roberta, self).__init__()
        self.model_tmp = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        self.output_layer = MLP(768, hidden_channels, out_channels, 1)
    
    def forward(self, input_ids, attention_mask):
        x = self.model_tmp(input_ids, attention_mask)
        x = x.pooler_output
        x = self.output_layer(x)

        return x

        
class VariationalAutoEncoder(torch.nn.Module):
    def __init__(self, emb_dim, loss, detach_target, beta=1):
        super(VariationalAutoEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.loss = loss
        self.detach_target = detach_target
        self.beta = beta

        self.criterion = None
        if loss == 'l1':
            self.criterion = nn.L1Loss()
        elif loss == 'l2':
            self.criterion = nn.MSELoss()

        self.fc_mu = nn.Linear(self.emb_dim, self.emb_dim)
        self.fc_var = nn.Linear(self.emb_dim, self.emb_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
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

class Multi_View_Fusion(nn.Module):
    def __init__(self, emb_dim):
        super(Multi_View_Fusion, self).__init__()
        self.W = nn.Linear(emb_dim, emb_dim)
        self.att = nn.Parameter(torch.Tensor(emb_dim, 1))
        self.emb_dim = emb_dim
        self.softmax = nn.Softmax(dim=1)
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)

    def forward(self, reprs):
        # repr : [3, n, emb_dim] 
        reprs = reprs.view((-1, 3, self.emb_dim)) # resize to [n, 3, emb_dim]
        H = self.W(reprs)
        H = torch.tanh(H)                                             # H: [n, 3, emb_dim]

        H = H.view((-1, self.emb_dim))
        logits = torch.mm(H, self.att)
        w = self.softmax(logits.view((-1, 3, 1)))  # w: [n, 3, 1]
        fused_repr = torch.bmm(reprs.view((-1, self.emb_dim, 3)), w).squeeze(2)    # fused_repr: [n, emb_dim]

        return fused_repr


            



