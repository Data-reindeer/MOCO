import torch
import numpy as np
import random
from torch import optim
from torch.nn.modules import activation
from time import perf_counter as t
import pdb
import pandas as pd
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

# import yaml
from torch.nn import *  
from torch.optim import * 
from torch.optim.lr_scheduler import * 

from splitters import scaffold_split, random_split, random_scaffold_split
from datasets import MoleculeDataset
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

# ============= tox21 ============
def train_general(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
        else:
            print('{} is invalid'.format(i))

    if len(roc_list) < y_true.shape[1]:
        print(len(roc_list))
        print('Some target is missing!')
        print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list), 0, y_true, y_scores
# ============= tox21 ============





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
    loss = model.loss(z, graph_batch.y[:, 1])
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
        num_epochs = 1000
        num_workers = 8
        acti = torch.nn.ReLU()
        # data = QM9(root='datasets/')

        # Fine-tune on tox21(just for test)
        # TODO: More standard format codes with arguments
        dataset_folder = '../datasets/'
        dataset = MoleculeDataset(dataset_folder + 'tox21', dataset='tox21')
        print(dataset)
        eval_metric = roc_auc_score

        data = smiles_list = pd.read_csv(dataset_folder + 'tox21' + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')

        num_target = 1
        num_proj_hidden = 200
        learning_rate = 8e-5
        weight_decay = 1.0e-11


    else: # Pre-training
        num_train = 50000
        batch_size = 500
        num_workers = 8
        num_epochs = 1000
        acti = torch.nn.ReLU()
        data = QM9(root='datasets/')

        num_target = 256
        num_proj_hidden = 200     
        learning_rate = 8e-5
        weight_decay = 1.0e-11

    # all_idx = get_random_indices(len(data), 123)
    # model_idx = all_idx[:100000]
    # test_idx = all_idx[len(model_idx): len(model_idx) + int(0.1 * len(data))]
    # val_idx = all_idx[len(model_idx) + len(test_idx):]
    # train_idx = model_idx[:num_train]

    # train_loader = DataLoader(data[train_idx], batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(data[val_idx], batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(data[test_idx], batch_size=batch_size, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # ft_train_loader = DataLoader(data[num_train:100000], batch_size=batch_size, shuffle=True)

    # data = data.to(device)
    # net2d = Net2D(in_dim=data[0].num_features, hidden_dim = 200, out_dim=num_target, activation=acti).to(device)
    # net3d = E_GCL(input_nf=data[0].num_features, output_nf=num_target, hidden_nf=64).to(device)

    # model = Model(net2d, net3d, num_hidden=num_target, num_proj_hidden= num_proj_hidden).to(device)
    net2d = Net2D(in_dim = train_dataset[0].num_features, hidden_dim = 200, out_dim=num_target, activation=acti).to(device)
    # TODO:  Wrong implementation here
    optimizer = torch.optim.Adam(net2d.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    Path = "../runs/net2d_para.pth"
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
            Path_ft = "../runs/net2d_para_ft.pth"
            pretrained_gnn_dict = torch.load(Path)
            model_state_dict = net2d.state_dict()
            state_dict = {k:v for k,v in pretrained_gnn_dict.items() if k in model_state_dict.keys()}
            del state_dict['conv.6.lin.weight']
            del state_dict['conv.6.bias']
            model_state_dict.update(state_dict) 
            net2d.load_state_dict(model_state_dict)  
        
    # ==================== tox21 ======================
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0


    train_func = train_general
    eval_func = eval_general

    for epoch in range(1, num_epochs + 1):
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        train_roc, train_acc, train_target, train_pred = eval_func(net2d, device, train_loader)
        val_roc, val_acc, val_target, val_pred = eval_func(net2d, device, val_loader)
        test_roc, test_acc, test_target, test_pred = eval_func(net2d, device, test_loader)

        train_roc_list.append(train_roc)
        train_acc_list.append(train_acc)
        val_roc_list.append(val_roc)
        val_acc_list.append(val_acc)
        test_roc_list.append(test_roc)
        test_acc_list.append(test_acc)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))

    

    # else:
    #     Path_ft = "runs/net2d_para_ft.pth"
    #     pretrained_gnn_dict = torch.load(Path)
    #     model_state_dict = net2d.state_dict()
    #     state_dict = {k:v for k,v in pretrained_gnn_dict.items() if k in model_state_dict.keys()}
    #     del state_dict['conv.6.lin.weight']
    #     del state_dict['conv.6.bias']
    #     model_state_dict.update(state_dict) 
    #     net2d.load_state_dict(model_state_dict)

    #     best_valid_loss = 1000
    #     loss_f = torch.nn.L1Loss(reduction = 'sum')

    #     for epoch in range(1, num_epochs + 1):
    #         for g in ft_train_loader:
    #             g.to(device)
    #             loss = train_fine_tune(net2d, g)

    #         now = t()
    #         print(f'(Fine-tuning training) | Epoch={epoch:03d}, loss={loss:.4f}, '
    #             f'this epoch {now - prev:.4f}, total {now - start:.4f}')
    #         prev = now

    #         result = 0
    #         for g in valid_loader:
    #             g.to(device)
    #             z = net2d(g.x, g.edge_index, g.batch).squeeze()
    #             result += loss_f(z, g.y[:, 1]).detach().item()
    #         result /= len(val_idx)
    #         print(f'Fine-tuning validation | Epoch={epoch:03d}, MAE loss={result:.4f})')
    #         if result < best_valid_loss: torch.save(net2d, Path_ft)
    #     print("=== Fine-tune training finish! ===")

    #     net2d = torch.load(Path_ft)
    #     net2d.eval()
    #     result = 0 
    #     for g in test_loader:
    #         g.to(device)
    #         z = net2d(g.x, g.edge_index, g.batch).squeeze()
    #         result += loss_f(z, g.y[:, 0])
    #     result = result / len(test_idx)
    #     print(f'Test MAE Loss | {result:.4f}')

    print("=== Finish ===")