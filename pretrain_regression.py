import time
import random
import numpy as np
from tqdm import tqdm
import nni
from nni.utils import merge_parameter
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader

from paras import args
import os
from datasets.molecule_3D_masking_dataset import Molecule3DMaskingDataset
from models.regression_model2D import MLP, Multi_View_Fusion, FingerPrintEncoder, GNNComplete 
from models.schnet import SchNet
from models.contrast import dual_CL



def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]\n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))

            torch.save(molecule_model_2D.state_dict(), args.output_model_dir \
                        + dir_name + 'model_2D_{}_model.pth'.format(args.coeff_mode))
            if args.type_view == 'fuse':
                torch.save(molecule_model_contrast_3d.state_dict(), args.output_model_dir \
                            + dir_name + 'model_3D_{}_model.pth'.format(args.coeff_mode))
                torch.save(molecule_model_contrast_fp.state_dict(), args.output_model_dir \
                            + dir_name + 'model_FP_{}_model.pth'.format(args.coeff_mode))
                torch.save(molecule_model_contrast_smiles.state_dict(), args.output_model_dir \
                            + dir_name + 'model_sm_{}_model.pth'.format(args.coeff_mode))
                torch.save(molecule_model_contrast.state_dict(), args.output_model_dir \
                            + dir_name + 'model_fuse_{}_model.pth'.format(args.coeff_mode))

    return

def train(args, molecule_model_2D, model_contrast, type_view, device, loader, optimizer):
    start_time = time.time()

    # === Attention ===
    molecule_model_2D.train()
    model_contrast.train()

    
    if type_view == 'fuse':
        molecule_model_contrast_3d.train()
        molecule_model_contrast_fp.train()
        molecule_model_contrast_smiles.train()   

    CL_loss_accum, CL_acc_accum = 0, 0
    AE_loss_accum = 0
    l = tqdm(loader)
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if type_view == '3d':
            molecule_repr = model_contrast(batch.x[:, 0], batch.positions, batch.batch)
            CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_repr, args)

        elif type_view == 'fingerprint':
            att_mask = torch.ones(batch.fingerprint.shape, dtype=int)
            molecule_repr = molecule_model_contrast(batch.fingerprint, att_mask)
            CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_repr, args)

        elif type_view == 'smiles':
            molecule_repr = model_contrast(batch.roberta_tensor)
            CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_repr, args)

        elif type_view == 'fuse':
            molecule_3D_repr = molecule_model_contrast_3d(batch.x[:, 0], batch.positions, batch.batch)
            molecule_fp_repr = molecule_model_contrast_fp(batch.fingerprint)
            # === Aggregate Multi-view representation ===
            molecule_smiles_repr = molecule_model_contrast_smiles(batch.roberta_tensor)
            reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr, molecule_smiles_repr), dim = 1)
            molecule_repr, coeff = model_contrast(reprs)

            
            if step+1 == len(loader): 
                coeff_list.append(coeff.tolist())

            CL_loss1, CL_acc1 = dual_CL(molecule_2D_repr, molecule_repr, args)
            CL_loss2, CL_acc2 = dual_CL(molecule_3D_repr, molecule_repr, args)
            CL_loss3, CL_acc3 = dual_CL(molecule_fp_repr, molecule_repr, args)
            CL_loss4, CL_acc4 = dual_CL(molecule_smiles_repr, molecule_repr, args)

            CL_loss = (CL_loss1 + CL_loss2 + CL_loss3 + CL_loss4) / 4
            CL_acc = (CL_acc1 + CL_acc2 + CL_acc3 + CL_acc4) / 4      
        

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc

        loss = 0
        loss += CL_loss 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)

    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, AE_loss_accum, time.time() - start_time))

    nni.report_intermediate_result(CL_acc_accum)

    return

if __name__ == '__main__':
    print('===start===\n')
    params = nni.get_next_parameter()
    args = merge_parameter(args, params)

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)

    if 'GEOM' in args.dataset:
        data_root = '../datasets/{}/'.format(args.dataset) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset)
        dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio, type_view = args.type_view)
    else:
        raise Exception
    
    # dir_name = '[Keep-Reression-lr{}]2d{}_3d{}_fp{}_sm{}_fuse{}_params/'.format(args.lr, args.gnn_lr_scale, args.schnet_lr_scale, 
    #                                                         args.fp_lr_scale, args.mlp_lr_scale, args.fuse_lr_scale)
    dir_name = 'Regression_models/'
    if not os.path.exists(args.output_model_dir + dir_name): os.makedirs(args.output_model_dir + dir_name)

    num_nodes = dataset.data.num_nodes
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # === 2D Encoder Building ===  
    molecule_model_2D = GNNComplete(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio).to(device)
    molecule_readout_func = global_mean_pool

    model_param_group = []
    coeff_list = []
    
    # === 3D Encoder Building ===
    # TODO: More 3D networks for molecules with equivariance
    if args.type_view == '3d':
        molecule_model_contrast = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                                        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)                              
        model_param_group.append({'params': molecule_model_contrast.parameters(), 'lr': args.lr * args.schnet_lr_scale})

    # === Fingerprint Encoder Building ===
    elif args.type_view == 'fingerprint':
        molecule_model_contrast = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim).to(device)
        model_param_group.append({'params': molecule_model_contrast.parameters(), 'lr': args.lr * args.fp_lr_scale})

    # === SMILES Encoder Building ===
    elif args.type_view == 'smiles':
        molecule_model_contrast = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)
        model_param_group.append({'params': molecule_model_contrast.parameters(), 'lr': args.lr * args.mlp_lr_scale})
    
    # === Multi-view Fusion via Attention ===
    elif args.type_view == 'fuse':
        molecule_model_contrast_3d = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                                            num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

        molecule_model_contrast_fp = FingerPrintEncoder(word_dim=64, out_dim=args.emb_dim).to(device)

        molecule_model_contrast_smiles = MLP(768, args.emb_dim, args.emb_dim, 1).to(device)

        molecule_model_contrast = Multi_View_Fusion(args.emb_dim).to(device)
        
 
        model_param_group.append({'params': molecule_model_contrast_fp.parameters(), 'lr': args.lr * args.fp_lr_scale}) # TODO :lr_scale 
        model_param_group.append({'params': molecule_model_contrast_3d.parameters(), 'lr': args.lr * args.schnet_lr_scale}) # TODO :lr_scale 
        model_param_group.append({'params': molecule_model_contrast_smiles.parameters(), 'lr': args.lr * args.mlp_lr_scale})
        model_param_group.append({'params': molecule_model_contrast.parameters(), 'lr': args.lr * args.fuse_lr_scale})
       
    # === Adam optimizer ===
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, molecule_model_contrast, args.type_view, device, loader, optimizer)
        np.savetxt( args.output_model_dir + dir_name + "attentions.txt",np.array(coeff_list))
    
    # save_model(save_best=False)