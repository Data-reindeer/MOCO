import torch
import numpy as np
import random
import argparse
from torch import optim
from torch.nn.modules import activation
from time import perf_counter as t
# import pdb

# import yaml
from torch.nn import *  
from torch.optim import * 
from torch.optim.lr_scheduler import * 

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
from model import Net2D, E_GCL, Model

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_random_indices(length, seed=12345):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices




def train_qm9(model: Model, net2d, net3d, graph_batch):
    model.train()
    optimizer.zero_grad()
    z1 = net2d(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
    z2 = net3d(graph_batch.x, graph_batch.edge_index, graph_batch.pos, graph_batch.batch)

    loss = model.loss(z1, z2, batch_size=graph_batch.num_graphs)
    loss.backward()
    optimizer.step()

    return loss.item()

def train_fine_tune(model: Net2D, graph_batch):
    model.train()
    optimizer.zero_grad()
    z = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch).squeeze()
    loss = model.loss(z, graph_batch.y[:, 0])
    loss.backward()
    optimizer.step()

    return loss.item()



if __name__ == '__main__':
    seed_all(123) 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    eval = True
    if eval: # Fine-tune
        num_train = 50000
        batch_size = 128
        num_epochs = 100
        acti = torch.nn.ReLU()
        data = QM9(root='datasets/')

        num_target = 1
        num_proj_hidden = 200
        learning_rate = 8e-5

    else: # Pre-training
        num_train = 50000
        batch_size = 500
        num_epochs = 100
        acti = torch.nn.ReLU()
        data = QM9(root='datasets/')

        num_target = 256
        num_proj_hidden = 200     
        learning_rate = 8e-5

    all_idx = get_random_indices(len(data), 123)
    model_idx = all_idx[:100000]
    test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(data))]
    val_idx = all_idx[len(model_idx) + len(test_idx):]
    train_idx = model_idx[:num_train]

    train_loader = DataLoader(data[train_idx], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(data[val_idx], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data[test_idx], batch_size=batch_size, shuffle=True)

    ft_train_loader = DataLoader(data[num_train:100000], batch_size=batch_size, shuffle=True)

    # data = data.to(device)
    net2d = Net2D(in_dim=data[0].num_features, hidden_dim = 512, out_dim=num_target, activation=acti).to(device)
    net3d = E_GCL(input_nf=data[0].num_features, output_nf=num_target, hidden_nf=64).to(device)

    model = Model(net2d, net3d, num_hidden=num_target, num_proj_hidden= num_proj_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start = t()
    prev = start
    Path = "runs/net2d_para.pth"
    if not eval:
        for epoch in range(1, num_epochs + 1):
            for i, g  in enumerate(train_loader):
                g.to(device)
                loss = train_qm9(model, net2d, net3d, g)

            now = t()
            print(f'(Pre-training time) | Epoch={epoch:03d}, loss={loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
        
        torch.save(net2d.state_dict(), Path)

    else:
        pretrained_gnn_dict = torch.load(Path)
        model_state_dict = net2d.state_dict()
        state_dict = {k:v for k,v in pretrained_gnn_dict.items() if k in model_state_dict.keys()}
        del state_dict['conv.6.lin.weight']
        del state_dict['conv.6.bias']
        model_state_dict.update(state_dict) 
        net2d.load_state_dict(model_state_dict)

        for epoch in range(1, num_epochs + 1):
            for g in ft_train_loader:
                g.to(device)
                loss = train_fine_tune(net2d, g)
            
            now = t()
            print(f'(Fine-tuning train time) | Epoch={epoch:03d}, loss={loss:.4f}, '
                f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
        print("=== Fine-tune training finish! ===")

        net2d.eval()
        loss = torch.nn.L1Loss()
        result = 0 
        for g in valid_loader:
            g.to(device)
            z = net2d(g.x, g.edge_index, g.batch).squeeze()
            result += loss(z, g.y[:, 0])
        result = result / z.shape[0]

        print(f'Validation MAE Loss | {result:.4f}')


    print("=== Finish ===")