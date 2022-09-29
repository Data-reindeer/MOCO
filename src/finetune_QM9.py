from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nni
from nni.utils import merge_parameter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from torch_geometric.nn import global_mean_pool

from splitters import scaffold_split, random_split, random_scaffold_split

from paras import args
from datasets.qm9_data import MoleculeDatasetComplete
from models.regression_model2D import GNN_graphpredComplete, GNNComplete, FingerPrintEncoder, MLP, Multi_View_Fusion
from models.schnet import SchNet
from models.egnn import EGNN

meann, mad = 0, 1
def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def compute_mean_mad(values):
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

# def train_general(model, device, loader, optimizer):
def train_general(device, loader, optimizer):
    model.train()
    model_3D.train()
    model_2D.train()
    model_fp.train()
    model_smiles.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)
        # molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
        molecule_3D_repr, _ = model_3D(batch.x[:, 0], batch.positions, edges=batch.edge_index, batch=batch.batch)
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

        if args.qm_label == 'mu':
            y = batch.mu
        elif args.qm_label == 'alpha':
            y = batch.alpha
        elif args.qm_label == 'homo':
            y = batch.homo
        elif args.qm_label == 'lumo':
            y = batch.lumo
        elif args.qm_label == 'gap':
            y = batch.gap
        elif args.qm_label == 'r2':
            y = batch.r2
        elif args.qm_label == 'zpve':
            y = batch.zpve
        elif args.qm_label == 'U0':
            y = batch.U0
        elif args.qm_label == 'U':
            y = batch.U
        elif args.qm_label == 'H':
            y = batch.H
        elif args.qm_label == 'G':
            y = batch.G
        elif args.qm_label == 'Cv':
            y = batch.Cv
        else:
            raise ValueError('Invalid label option.')

        # Special for QM9 normalization      
        # y = ((y-meann)/mad).squeeze()   
        y = y.squeeze()  
        loss = reg_criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

        if step+1 == len(loader): 
            coeff_list.append(coeff.tolist())

    return total_loss / len(loader)


# def eval_general(model, device, loader):
def eval_general(device, loader):
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
            # molecule_3D_repr = model_3D(batch.x[:, 0], batch.positions, batch.batch)
            molecule_3D_repr, _ = model_3D(batch.x[:, 0], batch.positions, edges=batch.edge_index, batch=batch.batch)
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
    
        if args.qm_label == 'mu':
            true = batch.mu.view(pred.shape)
        elif args.qm_label == 'alpha':
            true = batch.alpha.view(pred.shape)
        elif args.qm_label == 'homo':
            true = batch.homo.view(pred.shape)
        elif args.qm_label == 'lumo':
            true = batch.lumo.view(pred.shape)
        elif args.qm_label == 'gap':
            true = batch.gap.view(pred.shape)
        elif args.qm_label == 'r2':
            true = batch.r2.view(pred.shape)
        elif args.qm_label == 'zpve':
            true = batch.zpve.view(pred.shape)
        elif args.qm_label == 'U0':
            true = batch.U0.view(pred.shape)
        elif args.qm_label == 'U':
            true = batch.U.view(pred.shape)
        elif args.qm_label == 'H':
            true = batch.H.view(pred.shape)
        elif args.qm_label == 'G':
            true = batch.G.view(pred.shape)
        elif args.qm_label == 'Cv':
            true = batch.Cv.view(pred.shape)
        else:
            raise ValueError('Invalid label option.')
        

        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    # y_pred = (torch.cat(y_pred, dim=0)*mad + meann).cpu().numpy()
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
    # dataset_folder = '../datasets/molecule_datasets/'
    dataset_folder = '../datasets/'
    dataset = MoleculeDatasetComplete(dataset_folder + args.dataset, dataset=args.dataset)
    print('dataset_folder:', dataset_folder)
    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    # print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.mu.shape[0]/num_tasks)))
    # print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.mu.shape[0]/num_tasks)))

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,
            frac_valid=0.1, frac_test=0.1)
        print('split via scaffold')
    elif args.split == 'random':
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, null_value=0, frac_train=0.840779, frac_valid=0.076434,
            frac_test=0.082787, seed=args.seed)
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

    if args.dataset == 'qm9':
        if args.qm_label == 'mu':
            ty = train_dataset.data.mu
        elif args.qm_label == 'alpha':
            ty = train_dataset.data.alpha
        elif args.qm_label == 'homo':
            ty = train_dataset.data.homo
        elif args.qm_label == 'lumo':
            ty = train_dataset.data.lumo
        elif args.qm_label == 'gap':
            ty = train_dataset.data.gap
        elif args.qm_label == 'r2':
            ty = train_dataset.data.r2
        elif args.qm_label == 'zpve':
            ty = train_dataset.data.zpve
        elif args.qm_label == 'U0':
            ty = train_dataset.data.U0
        elif args.qm_label == 'U':
            ty = train_dataset.data.U
        elif args.qm_label == 'H':
            ty = train_dataset.data.H
        elif args.qm_label == 'G':
            ty = train_dataset.data.G
        elif args.qm_label == 'Cv':
            ty = train_dataset.data.Cv
        else:
            raise ValueError('Invalid label option.')
        meann, mad = compute_mean_mad(ty)
    # set up model
    coeff_list = []

    model_2D = GNNComplete(num_layer=args.num_layer, emb_dim=args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
    # model_3D = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
    #             num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    model_3D = EGNN(in_node_nf=15, hidden_nf=args.emb_dim, out_node_nf=args.emb_dim).to(device)
    model_fp = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim, num_layer=args.transformer_layer).to(device)
    model_smiles = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
    model = Multi_View_Fusion(args.emb_dim).to(device)
    output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
    # print(model_2D)
    print(output_layer)

    # ==== Load the pretrained model ====
    
    model_2D.load_state_dict(torch.load(args.input_model_file+'model_2D_search_model.pth', map_location='cuda:0'))
    # model_2D.load_state_dict(torch.load(args.input_model_file+'regression_model.pth', map_location='cuda:0'))
    model_3D.load_state_dict(torch.load(args.input_model_file+'model_3D_search_model.pth', map_location='cuda:0'))
    model_fp.load_state_dict(torch.load(args.input_model_file+'model_FP_search_model.pth', map_location='cuda:0'))
    model_smiles.load_state_dict(torch.load(args.input_model_file+'model_sm_search_model.pth', map_location='cuda:0'))
    model.load_state_dict(torch.load(args.input_model_file+'model_fuse_search_model.pth', map_location='cuda:0'))
    print('=== Model loaded ===')

    model_param_group = []
    model_param_group.append({'params': model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_fp.parameters(), 'lr': args.lr * args.fp_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': model_smiles.parameters(), 'lr': args.lr * args.mlp_lr_scale})
    model_param_group.append({'params': model.parameters(), 'lr': args.lr * args.fuse_lr_scale})
    model_param_group.append({'params': output_layer.parameters(),'lr': args.lr * args.lr_scale})                
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    # scheduler = StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    reg_criterion = torch.nn.L1Loss()


    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE']
    best_val_rmse, best_val_idx = 1e10, 0

    train_func = train_general
    eval_func = eval_general
    print('=== Parameter set ===')

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval_func(device, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}

        val_result, val_target, val_pred = eval_func(device, val_loader)
        test_result, test_target, test_pred = eval_func(device, test_loader)

        train_result_list.append(train_result)
        val_result_list.append(val_result)
        test_result_list.append(test_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))
        print()
        nni.report_intermediate_result(test_result['MAE'])
        print(coeff_list[-1])

        if val_result['MAE'] < best_val_rmse:
            best_val_rmse = val_result['MAE']
            best_val_idx = epoch - 1
        scheduler.step()

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))
    print(coeff_list[-1])
    nni.report_final_result(test_result_list[best_val_idx]['MAE'])
    # if args.output_model_dir is not '':
    #     output_model_path = join(args.output_model_dir, 'model_final.pth')
    #     saved_model_dict = {
    #         'molecule_model': molecule_model.state_dict(),
    #         'model': model.state_dict()
    #     }
    #     torch.save(saved_model_dict, output_model_path)