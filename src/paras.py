import argparse

parser = argparse.ArgumentParser()

# Seed and basic info
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

# Dataset and dataloader
parser.add_argument('--input_data_dir', type=str, default='../datasets/GEOM/processed')
parser.add_argument('--dataset', type=str, default='bbbp')
parser.add_argument('--num_workers', type=int, default=1)

# Training strategies
parser.add_argument('--split', type=str, default='scaffold')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_scale', type=float, default=1)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--type_view', type=str, default='fuse', choices=['2d', '3d', 'fingerprint', 'smiles', 'fuse'])

# GNN for Molecules
parser.add_argument('--net2d', type=str, default='gin')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5) 
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')


# Fingerprint for Molecules
parser.add_argument('--mlp_layers', type=int, default=3)
parser.add_argument('--transformer_layer', type=int, default=1)

# SchNet
parser.add_argument('--net3d', type=str, default='schnet')
parser.add_argument('--model_3d', type=str, default='schnet',
                    choices=['schnet'])
parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean',
                    choices=['mean', 'add'])


# Contrastive CL
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='InfoNCE_dot_prod')
parser.add_argument('--T', type=float, default=0.3)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.set_defaults(normalize=False) 
parser.add_argument('--SSL_masking_ratio', type=float, default=0)

# Loading and saving roots
parser.add_argument('--input_model_file', type=str, default='../runs/Classification_models/')
parser.add_argument('--output_model_dir', type=str, default='../runs/')

# Learning rate for different networks
parser.add_argument('--gnn_lr_scale', type=float, default=1)
parser.add_argument('--schnet_lr_scale', type=float, default=0.1)
parser.add_argument('--fp_lr_scale', type=float, default=0.1)
parser.add_argument('--mlp_lr_scale', type=float, default=10)
parser.add_argument('--fuse_lr_scale', type=float, default=0.001)
parser.add_argument('--mode', type=str, default='mean')
parser.add_argument('--coeff_mode', type=str, default='final')


# If print out eval metric for training data
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=True)
# =======================

args = parser.parse_args()
print('arguments\t', args)

