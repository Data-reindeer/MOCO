from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nni
from nni.utils import merge_parameter
from tqdm import tqdm
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree 
from splitters import random_split
from paras import args
from datasets.molecule_datasets import MoleculeDataset
from models.model2D import GNN_case, FingerPrintEncoder, MLP
from models.schnet import SchNet

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# TODO: clean up
def train_general(model, device, loader, optimizer):
# def train_general(device, loader, optimizer):
    model.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        if args.type_view == 'fingerprint':
            output_layer.train()
            h = model(batch.fingerprint)
            pred = output_layer(h)
        elif args.type_view == '3d':
            output_layer.train()
            h = model(batch.x[:, 0], batch.positions, batch.batch)
            pred = output_layer(h)

        elif args.type_view == 'smiles':
            output_layer.train()
            h = model(batch.roberta_tensor)
            pred = output_layer(h)

        elif args.type_view == '2d':
            h = model(batch.x[:,0], batch.edge_index, batch.edge_attr)
            h = global_mean_pool(h, batch.batch)
            pred = output_layer(h)

        y = batch.y.view(pred.shape)    
        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, device, loader):
# def eval_general(device, loader):
    model.eval()
    output_layer.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            if args.type_view == 'fingerprint':
                h = model(batch.fingerprint)
                pred = output_layer(h)

            elif args.type_view == '3d':
                h = model(batch.x[:, 0], batch.positions, batch.batch)
                pred = output_layer(h)

            elif args.type_view == 'smiles':
                h = model(batch.roberta_tensor)
                pred = output_layer(h)

            elif args.type_view == '2d':
                h = model(batch.x[:,0], batch.edge_index, batch.edge_attr)
                h = global_mean_pool(h, batch.batch)
                pred = output_layer(h)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == '__main__':
    params = nni.get_next_parameter()
    args = merge_parameter(args, params)
    seed_all(args.runseed)

    # device = torch.device('cuda:' + str(args.device)) \
    #    if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = 1
    dataset_folder = '../datasets/GEOM/processed/Case_Zero_AromRing_nmol_10000_Roberta'
    dataset = MoleculeDataset(dataset_folder, dataset=args.dataset)

    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.y.shape[0]/num_tasks)))
    print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.y.shape[0]/num_tasks)))

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, null_value=0, frac_train=0.7, frac_valid=0.15,
        frac_test=0.15, seed=args.seed)
    print('randomly split')

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    full_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)              

    # set up model
    if args.type_view == 'fingerprint': 
        model = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim, num_layer=args.transformer_layer).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        
        model_param_group = [{'params': model.parameters(),'lr': args.lr * args.fp_lr_scale},
                          {'params': output_layer.parameters()}]  

    elif args.type_view == '3d':
        model = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        model_param_group = [{'params': model.parameters(),'lr': args.lr * args.schnet_lr_scale},
                          {'params': output_layer.parameters()}]  

    elif args.type_view == 'smiles':
        model = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        model_param_group = [{'params': model.parameters(),'lr': args.lr * args.mlp_lr_scale},
                          {'params': output_layer.parameters()}]  

    elif args.type_view == '2d':
        model = GNN_case(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, 
                    drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        model_param_group = [{'params': model.parameters(),
                          'lr': args.lr * args.gnn_lr_scale},{'params': output_layer.parameters()}] 


    print(model)                
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    reg_criterion = torch.nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    train_func = train_general
    eval_func = eval_general
    print('=== Parameter set ===')

    for epoch in range(1, args.epochs + 1):
        # loss_acc = train_func(device, train_loader, optimizer)
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval_func(model, device, train_loader)
            # train_result, train_target, train_pred = eval_func(device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred = eval_func(model, device, val_loader)
        test_result, test_target, test_pred = eval_func(model, device, test_loader)
        # val_result, val_target, val_pred = eval_func(device, val_loader)
        # test_result, test_target, test_pred = eval_func(device, test_loader)

        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()
        nni.report_intermediate_result(test_result['MAE'])

        if val_result['MAE'] < best_val_rmse:
            best_val_rmse = val_result['MAE']
            best_val_idx = epoch - 1

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
    nni.report_final_result(test_result_list[best_val_idx]['MAE'])