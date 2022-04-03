import time
import random
import numpy as np
from tqdm import tqdm
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.loader import DataLoader



from paras import args
from datasets.molecule_3D_masking_dataset import Molecule3DMaskingDataset
from models.model2D import GNN, MLP, roberta, VariationalAutoEncoder, Multi_View_Fusion
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
            torch.save(molecule_model_2D.state_dict(), args.output_model_dir +'net2D_model.pth')

    return

def train(args, molecule_model_2D, model_contrast, type_view, device, loader, optimizer):
    start_time = time.time()

    molecule_model_2D.train()
    model_contrast.train()

    CL_loss_accum, CL_acc_accum = 0, 0
    AE_loss_accum = 0
    l = tqdm(loader)
    for step, batch in enumerate(l):
        batch = batch.to(device)

        node_repr = molecule_model_2D(batch.x, batch.edge_index, batch.edge_attr)
        molecule_2D_repr = molecule_readout_func(node_repr, batch.batch)

        if type_view == '3d':
            molecule_repr = model_contrast(batch.x[:, 0], batch.positions, batch.batch)
        elif type_view == 'fingerprint':
            molecule_fp_repr = molecule_model_contrast_fp(batch.fingerprint)
            molecule_3D_repr = molecule_model_contrast_3d(batch.x[:, 0], batch.positions, batch.batch)
        elif type_view == 'smiles':
            molecule_repr = model_contrast(input_ids=batch.input_ids, attention_mask=batch.attention_mask)

        # === Aggregate Multi-view representation ===
        
        reprs = torch.stack((molecule_2D_repr, molecule_3D_repr, molecule_fp_repr), dim = 0)
        fusion_repr = Fusion_model(reprs)

        # === Contrastive learning ===
        CL_loss, CL_acc = dual_CL(molecule_2D_repr, fusion_repr, args)
        # CL_loss = (CL_loss1 + CL_loss2 + CL_loss3)/3
        # CL_acc = (CL_acc1 + CL_acc2 + CL_acc3)/3

        # AE_loss1 = AE_2D_3D_model(molecule_2D_repr, molecule_repr)
        # AE_loss2 = AE_3D_2D_model(molecule_repr, molecule_2D_repr)
        # AE_loss = (AE_loss1 + AE_loss2)/2

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        # AE_loss_accum += AE_loss.detach().cpu().item()

        loss = 0
        loss += CL_loss 
        # loss += AE_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    temp_loss = CL_loss_accum + AE_loss_accum
    temp_loss = CL_loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
    print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tTime: {:.5f}'.format(
        CL_loss_accum, CL_acc_accum, time.time() - start_time))
    # print('CL Loss: {:.5f}\tCL Acc: {:.5f}\t\tAE Loss: {:.5f}\tTime: {:.5f}'.format(
    #     CL_loss_accum, CL_acc_accum, AE_loss_accum, time.time() - start_time))
    return


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)

    if 'GEOM' in args.dataset:
        data_root = '../datasets/{}/'.format(args.dataset) if args.input_data_dir == '' else '{}/{}/'.format(args.input_data_dir, args.dataset)
        dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio, type_view = args.type_view)
    else:
        raise Exception

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print('2D model:{} \t 3D model:{}'.format(args.net2d, args.net3d))
    # === 2D Encoder Building ===  
    molecule_model_2D = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.net2d).to(device)
    molecule_readout_func = global_mean_pool

    # === 3D Encoder Building ===
    # TODO: More 3D networks for molecules with equivariance
    if args.type_view == '3d':
        molecule_model_contrast = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

    # === Fingerprint Encoder Building ===
    elif args.type_view == 'fingerprint':
        molecule_model_contrast_fp = MLP(1024, args.emb_dim, args.emb_dim, args.mlp_layers).to(device)
        molecule_model_contrast_3d = SchNet(hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)

    # === SMILES Encoder Building ===
    elif args.type_view == 'smiles':
        # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        # configuration = RobertaConfig(hidden_size=300, vocab_size = tokenizer.vocab_size)
        molecule_model_contrast = roberta(args.emb_dim, args.emb_dim).to(device)

        # molecule_model_contrast = RobertaModel(configuration).to(device)

    # === VAE model to constrain the contrastive loss ===
    AE_2D_3D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device) 
    AE_3D_2D_model = VariationalAutoEncoder(
            emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device) 
    
    # === Multi-view Fusion via Attention ===
    Fusion_model = Multi_View_Fusion(args.emb_dim).to(device)


    # === Adam optimizer ===
    model_param_group = []
    model_param_group.append({'params': molecule_model_2D.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': molecule_model_contrast_fp.parameters(), 'lr': args.lr * args.mlp_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': molecule_model_contrast_3d.parameters(), 'lr': args.lr * args.mlp_lr_scale}) # TODO :lr_scale 
    model_param_group.append({'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.gnn_lr_scale})
    model_param_group.append({'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.mlp_lr_scale})
    model_param_group.append({'params': Fusion_model.parameters(), 'lr': args.lr})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = 1e10

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, molecule_model_2D, molecule_model_contrast_fp, args.type_view, device, loader, optimizer)

    save_model(save_best=False)
