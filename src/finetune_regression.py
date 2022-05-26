from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nni
from nni.utils import merge_parameter
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from torch_geometric.nn import global_mean_pool
from splitters import scaffold_split, random_split, random_scaffold_split

from paras import args
from datasets.regression_datasets import MoleculeDatasetComplete
from models.regression_model2D import GNN_graphpredComplete, GNNComplete, FingerPrintEncoder, MLP, Multi_View_Fusion
from models.schnet import SchNet


def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train_general(model, device, loader, optimizer):
    model.train()
    model_3D.train()
    model_2D.train()
    model_fp.train()
    model_smiles.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
        molecule_2D_repr = global_mean_pool(model_2D(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        molecule_fp_repr = model_fp(batch.fingerprint)
        # === Aggregate Multi-view representation ===
        molecule_smiles_repr = model_smiles(batch.roberta_tensor)
        reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr, molecule_smiles_repr), dim = 1)
        molecule_repr, coeff = model(reprs)
        # tmp_coeff = coeff.expand((reprs.shape[0],) + coeff.shape)
        # molecule_repr = (tmp_coeff * reprs).sum(1)
        pred = output_layer(molecule_repr).squeeze()

        # # ============= Single View ===========
        # molecule_2D_repr =  global_mean_pool(model(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
        # pred = output_layer(molecule_2D_repr).squeeze()

        y = batch.y.squeeze()

        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

        if step+1 == len(loader): 
            coeff_list.append(coeff.tolist())

    return total_loss / len(loader)


def eval_general(model, device, loader):
    model.eval()
    model_3D.eval()
    model_2D.eval()
    model_fp.eval()
    model_smiles.eval()
    output_layer.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
            molecule_2D_repr = global_mean_pool(model_2D(batch.x, batch.edge_index, batch.edge_attr),batch.batch)
            molecule_fp_repr = model_fp(batch.fingerprint)
            # === Aggregate Multi-view representation ===
            molecule_smiles_repr = model_smiles(batch.roberta_tensor)
            reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr, molecule_smiles_repr), dim = 1)
            molecule_repr, _ = model(reprs)
            pred = output_layer(molecule_repr).squeeze(1)

            # # ===== Single View ======
            # molecule_2D_repr = global_mean_pool(model(batch.x, batch.edge_index, batch.edge_attr),batch.batch)
            # pred = output_layer(molecule_2D_repr).squeeze(1)
    
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
    dataset_folder = '../datasets/molecule_datasets/'
    dataset = MoleculeDatasetComplete(dataset_folder + args.dataset, dataset=args.dataset)
    print('dataset_folder:', dataset_folder)
    print(dataset)

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed)
        print('randomly split')
    elif args.split == 'random_scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = random_scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1, seed=args.seed)
        print('random scaffold')
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # set up model
    coeff_list = []

    model_2D = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
    model_3D = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    model_fp = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim, num_layer=args.transformer_layer).to(device)
    model_smiles = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
    model = Multi_View_Fusion(args.emb_dim).to(device)
    output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
    print(model_2D)
    print(output_layer)

    # ==== Load the pretrained model ====
    model_2D.load_state_dict(torch.load(args.input_model_file+'model_2D_0_model.pth', map_location='cuda:0'))
    model_3D.load_state_dict(torch.load(args.input_model_file+'model_3D_0_model.pth', map_location='cuda:0'))
    model_fp.load_state_dict(torch.load(args.input_model_file+'model_FP_0_model.pth', map_location='cuda:0'))
    model_smiles.load_state_dict(torch.load(args.input_model_file+'model_sm_0_model.pth', map_location='cuda:0'))
    model.load_state_dict(torch.load(args.input_model_file+'model_fuse_0_model.pth', map_location='cuda:0'))
    print('=== Model loaded ===')

    model_param_group = []
    model_param_group.append({'params': model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_3D.parameters(), 'lr': args.lr * args.gnn_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_fp.parameters(), 'lr': args.lr * args.gnn_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_smiles.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': model.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    model_param_group.append({'params': output_layer.parameters(),'lr': args.lr * args.lr_scale})                
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    reg_criterion = torch.nn.MSELoss()


    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    train_func = train_general
    eval_func = eval_general
    print('=== Parameter set ===')

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval_func(model, device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred = eval_func(model, device, val_loader)
        test_result, test_target, test_pred = eval_func(model, device, test_loader)

        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()
        nni.report_intermediate_result(test_result['RMSE'])
        print(coeff_list[-1])

        if val_result['RMSE'] < best_val_rmse:
            best_val_rmse = val_result['RMSE']
            best_val_idx = epoch - 1

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
    print(coeff_list[-1])
    nni.report_final_result(test_result_list[best_val_idx]['RMSE'])
    # if args.output_model_dir is not '':
    #     output_model_path = join(args.output_model_dir, 'model_final.pth')
    #     saved_model_dict = {
    #         'molecule_model': molecule_model.state_dict(),
    #         'model': model.state_dict()
    #     }
    #     torch.save(saved_model_dict, output_model_path)
