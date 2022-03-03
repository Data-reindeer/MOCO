import time
import random
import numpy as np
from tqdm import tqdm
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import DataLoader

from paras import args
from datasets import Molecule3DMaskingDataset
from models.model2D import GNN
from models.schnet import SchNet
from models.contrast import dual_CL



def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
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
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + 'net2D_model.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + 'model_complete.pth')

        else:
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir + 'net2D_model_final.pth')
            saver_dict = {
                'model': molecule_model_2D.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir + 'model_complete_final.pth')
    return

def train(args, molecule_model_2D, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    molecule_model_3D.train()

    CL_loss_accum, CL_acc_accum = 0, 0
    l = tqdm(loader)
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if args.model_3d == 'schnet':
            molecule_3D_repr = molecule_model_3D(batch.x[:, 0], batch.positions, batch.batch)

        CL_loss, CL_acc = dual_CL(molecule_2D_repr, molecule_3D_repr, args)

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
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)

    if 'GEOM' in args.dataset:
        data_root = '../datasets/{}/'.format(args.dataset) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset)
        dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio)
    else:
        raise Exception
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # === 2D and 3D Model Building ===
    print('2D model:{} \t 3D model:{}'.format(args.net2d, args.net3d))
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
    molecule_readout_func = global_add_pool

    # TODO: More 3D networks for molecules with equivariance
    molecule_model_3D = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
    num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

    # === Adam optimizer ===
    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, device, loader, optimizer)

    save_model(save_best=False)
