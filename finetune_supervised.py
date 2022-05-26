from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import nni
from nni.utils import merge_parameter
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from splitters import scaffold_split, random_split, random_scaffold_split

from paras import args
from datasets.molecule_datasets import MoleculeDataset
from models.model2D import GNN, FingerPrintEncoder, MLP, Multi_View_Fusion
from models.schnet import SchNet

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def get_num_task(dataset):
    # Get output dimensions of different tasks
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp']:
        return 1
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    raise ValueError('Invalid dataset name.')

# TODO: clean up
def train_general(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
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
            h = model_2D(batch.x, batch.edge_index, batch.edge_attr)
            h = global_mean_pool(h, batch.batch)
            pred = output_layer(h)
        elif args.type_view == 'fuse':
            model_3D.train()
            model_2D.train()
            model_fp.train()
            model_smiles.train()
            output_layer.train()

            molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
            molecule_2D_repr = global_mean_pool(model_2D(batch.x, batch.edge_index, batch.edge_attr), batch.batch)
            molecule_fp_repr = model_fp(batch.fingerprint)
            # === Aggregate Multi-view representation ===
            molecule_smiles_repr = model_smiles(batch.roberta_tensor)
            reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr, molecule_smiles_repr), dim = 1)
            molecule_repr, coeff = model(reprs)
            pred = output_layer(molecule_repr)

        y = batch.y.view(pred.shape).to(torch.float64)
        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
        
        if step+1 == len(loader): 
            coeff_list.append(coeff.tolist())
        
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
                h = model(batch.x, batch.edge_index, batch.edge_attr)
                h = global_mean_pool(h, batch.batch)
                pred = output_layer(h)
            elif args.type_view == 'fuse':
                model_3D.eval()
                model_2D.eval()
                model_fp.eval()
                model_smiles.eval()
                output_layer.eval()

                molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
                molecule_2D_repr = global_mean_pool(model_2D(batch.x, batch.edge_index, batch.edge_attr),batch.batch)
                molecule_fp_repr = model_fp(batch.fingerprint)
                # === Aggregate Multi-view representation ===
                molecule_smiles_repr = model_smiles(batch.roberta_tensor)
                reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr, molecule_smiles_repr), dim = 1)
                molecule_repr, _ = model(reprs)
                pred = output_layer(molecule_repr)
    
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


if __name__ == '__main__':
    # params = nni.get_next_parameter()
    # args = merge_parameter(args, params)
    seed_all(args.runseed)

    device = torch.device('cuda:' + str(args.device)) \
       if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = '../datasets/molecule_datasets/'
    dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.y.shape[0]/num_tasks)))
    print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.y.shape[0]/num_tasks)))


    eval_metric = roc_auc_score

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
    if args.type_view == 'fingerprint': 
        model = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim, num_layer=args.transformer_layer).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        
        model_param_group = [{'params': model.parameters()},
                          {'params': output_layer.parameters(),'lr': args.lr * args.lr_scale}]  

    elif args.type_view == '3d':
        model = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        model_param_group = [{'params': model.parameters()},
                          {'params': output_layer.parameters(),'lr': args.lr * args.lr_scale}]  

    elif args.type_view == 'smiles':
        model = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        model_param_group = [{'params': model.parameters()},
                          {'params': output_layer.parameters(),'lr': args.lr * args.lr_scale}]  

    elif args.type_view == '2d':
        model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, 
                    drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
        model_param_group = [{'params': model.molecule_model.parameters(),
                          'lr': args.lr * args.gnn_lr_scale}] 
    
    elif args.type_view == 'fuse':
        coeff_list = []
        model_2D = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, 
                    drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
        model_3D = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                    num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
        model_fp = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim, num_layer=args.transformer_layer).to(device)
        model_smiles = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
        model = Multi_View_Fusion(args.emb_dim).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)

        # ==== Load the pretrained model ====
        model_2D.load_state_dict(torch.load(args.input_model_file+'model_2D_0_model.pth', map_location='cuda:0'))
        model_3D.load_state_dict(torch.load(args.input_model_file+'model_3D_0_model.pth', map_location='cuda:0'))
        model_fp.load_state_dict(torch.load(args.input_model_file+'model_FP_0_model.pth', map_location='cuda:0'))
        model_smiles.load_state_dict(torch.load(args.input_model_file+'model_sm_0_model.pth', map_location='cuda:0'))
        model.load_state_dict(torch.load(args.input_model_file+'model_fuse_0_model.pth', map_location='cuda:0'))

        model_param_group = []
        model_param_group.append({'params': model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale}) # TODO :lr_scale 
        model_param_group.append({'params': model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale}) # TODO :lr_scale 
        model_param_group.append({'params': model_fp.parameters(), 'lr': args.lr * args.fp_lr_scale}) # TODO :lr_scale 
        model_param_group.append({'params': model_smiles.parameters(), 'lr': args.lr * args.mlp_lr_scale})
        model_param_group.append({'params': model.parameters(), 'lr': args.lr * args.fuse_lr_scale})
        model_param_group.append({'params': output_layer.parameters(),'lr': args.lr * args.lr_scale})


    print(model)                
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_acc_list, val_acc_list, test_acc_list = [], [], []
    best_val_roc, best_val_idx = -1, 0

    train_func = train_general
    eval_func = eval_general

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        train_roc = train_acc = 0
        
        val_roc, val_acc, val_target, val_pred = eval_func(model, device, val_loader)
        test_roc, test_acc, test_target, test_pred = eval_func(model, device, test_loader)

        train_roc_list.append(train_roc)
        train_acc_list.append(train_acc)
        val_roc_list.append(val_roc)
        val_acc_list.append(val_acc)
        test_roc_list.append(test_roc)
        test_acc_list.append(test_acc)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        # nni.report_intermediate_result(test_roc)
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1

    # nni.report_final_result(test_roc_list[best_val_idx])
    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
