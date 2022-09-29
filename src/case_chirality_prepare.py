import os
import json
import pdb
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from os.path import join
from itertools import repeat
from datasets.molecule_datasets import allowable_features
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch_geometric.data import Data, InMemoryDataset
from transformers import RobertaTokenizer


def mol_to_graph_data_obj_simple_3D(mol, type_view):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """

    # todo: more atom/bond features in the future
    # atoms, two features: atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    # every CREST conformer gets its own mol object,
    # every mol object has only one RDKit conformer
    # ref: https://github.com/learningmatter-mit/geom/blob/master/tutorials/
    if type_view == '3D':
        conformer = mol.GetConformers()[0]
        positions = conformer.GetPositions() # Positions are not pre-produced
        positions = torch.Tensor(positions)
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, positions=positions)

    elif type_view in ['smiles', 'fingerprint']:
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr)

    elif type_view == 'fuse':
        conformer = mol.GetConformers()[0]
        positions = conformer.GetPositions()
        positions = torch.Tensor(positions)
        data = Data(x=x, edge_index=edge_index,
                    edge_attr=edge_attr, positions=positions)

    return data


def summarise():
    """ summarise the stats of molecules and conformers """
    dir_name = '../datasets/GEOM/rdkit_folder'
    drugs_file = '{}/summary_drugs.json'.format(dir_name)

    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    # expected: 304,466 molecules
    print('number of items (SMILES): {}'.format(len(drugs_summary.items())))

    sum_list = []
    drugs_summary = list(drugs_summary.items())

    for smiles, sub_dic in tqdm(drugs_summary):
        #==== Path should match ====
        if sub_dic.get('pickle_path', '') == '':
            continue

        mol_path = join(dir_name, sub_dic['pickle_path'])
        with open(mol_path, 'rb') as f:
            mol_sum = {}
            mol_dic = pickle.load(f)
            conformer_list = mol_dic['conformers']
            conformer_dict = conformer_list[0]
            rdkit_mol = conformer_dict['rd_mol']
            data = mol_to_graph_data_obj_simple_3D(rdkit_mol)

            mol_sum['geom_id'] = conformer_dict['geom_id']
            mol_sum['num_edge'] = len(data.edge_attr)
            mol_sum['num_node'] = len(data.positions)
            mol_sum['num_conf'] = len(conformer_list)

            bw_ls = []
            for conf in conformer_list:
                bw_ls.append(conf['boltzmannweight'])
            mol_sum['boltzmann_weight'] = bw_ls
        sum_list.append(mol_sum)
    return sum_list


class MoleculeDataset(InMemoryDataset):

    def __init__(self, root, n_mol, n_conf, n_upper, type_view, radius=2, nBits=1024, transform=None, 
                 seed=777, pre_transform=None, pre_filter=None, empty=False):
        os.makedirs(root, exist_ok=True)
        os.makedirs(join(root, 'raw'), exist_ok=True)
        os.makedirs(join(root, 'processed'), exist_ok=True)

        self.root, self.seed = root, seed
        self.n_mol, self.n_conf, self.n_upper = n_mol, n_conf, n_upper
        self.pre_transform, self.pre_filter = pre_transform, pre_filter
        self.type_view = type_view
        if type_view in ['fingerprint', 'fuse']:
            self.radius = radius
            self.nBits = nBits

        super(MoleculeDataset, self).__init__(
            root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('root: {},\ndata: {},\nn_mol: {},\nn_conf: {}'.format(
            self.root, self.data, self.n_mol, self.n_conf))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx+1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return os.listdir(self.raw_dir)

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):
        data_list = []
        data_smiles_list = []

        dir_name = '../datasets/GEOM/rdkit_folder'
        drugs_file = '{}/summary_drugs.json'.format(dir_name)
        with open(drugs_file, 'r') as f:
            drugs_summary = json.load(f)
        drugs_summary = list(drugs_summary.items())
        print('# of SMILES: {}'.format(len(drugs_summary)))

        smiles_num = 0
        for smiles, _ in tqdm(drugs_summary):  
            if smiles_num >= self.n_mol: break  
            smiles_num += 1
            

        convert = {'S':[1], 'R':[-1]}        

        random.seed(self.seed)
        random.shuffle(drugs_summary)
        mol_idx, idx, notfound = 0, 0, 0
        cnt = {1:0, -1:0}
        for smiles, sub_dic in tqdm(drugs_summary):
            # === Path should match ===
            if sub_dic.get('pickle_path', '') == '':
                # === No pickle path available ===
                notfound += 1
                continue

            # === The conformers are pre-produced ===
            mol_path = join(dir_name, sub_dic['pickle_path'])
            with open(mol_path, 'rb') as f:
                mol_dic = pickle.load(f)
                conformer_list = mol_dic['conformers']

                # === count should match ===
                conf_n = len(conformer_list)
                if conf_n < self.n_conf or conf_n > self.n_upper:
                    notfound += 1
                    continue

                conf_list = [
                    Chem.MolToSmiles(
                        Chem.MolFromSmiles(
                            Chem.MolToSmiles(rd_mol['rd_mol'])))
                    for rd_mol in conformer_list[:self.n_conf]]

                conf_list_raw = [
                    Chem.MolToSmiles(rd_mol['rd_mol'])
                    for rd_mol in conformer_list[:self.n_conf]]
                # check that they're all the same
                same_confs = len(list(set(conf_list))) == 1
                same_confs_raw = len(list(set(conf_list_raw))) == 1
                

                if not same_confs:
                    if same_confs_raw is True:
                        print("Interesting")
                    notfound += 1
                    continue
                
                
                conformer_dict = conformer_list[0]
                rdkit_mol = conformer_dict['rd_mol']
                tags = Chem.FindMolChiralCenters(rdkit_mol)
                if len(tags) != 1: continue


                data = mol_to_graph_data_obj_simple_3D(rdkit_mol, self.type_view)
                fingerprint = np.array(GetMorganFingerprintAsBitVect(rdkit_mol, self.radius, nBits=self.nBits))
                data.fingerprint = torch.tensor(fingerprint, dtype = torch.float).unsqueeze(0)
                data.id = torch.tensor([idx])
                data.mol_id = torch.tensor([mol_idx])
                data.y = torch.tensor(convert[tags[0][1]])
                y = data.y.item()
                cnt[y] += 1
                if cnt[y] > 5000 and cnt[-1*y] > 5000: break
                if cnt[y] > 5000 and cnt[-1*y] < 5000: continue
          
                data_smiles_list.append(smiles)
                data_list.append(data)
                idx += 1



            # # select the first n_mol molecules
            # if mol_idx + 1 >= self.n_mol:
            #     break
            # if same_confs:
            #     mol_idx += 1
        

        
        print('mol id: [0, {}]\tlen of smiles: {}\tlen of set(smiles): {}'.format(
            mol_idx, len(data_smiles_list), len(set(data_smiles_list))))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = join(self.processed_dir, 'smiles.csv')
        print('saving to {}'.format(saver_path))
        data_smiles_series.to_csv(saver_path, index=False, header=False) 

        # tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
        # dicts = tokenizer(data_smiles_list, max_length=510, return_tensors="pt", padding='max_length', truncation=True)['input_ids']
        
        if self.type_view in ['smiles', 'fuse']:
            # inputs format : {input_ids: [], attention_mask:[]}
            # roberta_tensor = torch.load('./Roberta_tensor.pt')
            roberta_tensor = torch.load('./Case_tensor.pt')
            for i in range(len(data_list)):
                # data_list[i].roberta_tensor = dicts[i].unsqueeze(0)
                data_list[i].roberta_tensor = roberta_tensor[i].unsqueeze(0)

        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("%d molecules do not meet the requirements" % notfound)
        print("%d molecules have been processed" % mol_idx)
        print("%d conformers/fingerprint have been processed" % idx)
        return


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('--sum', type=bool, default=False, help='cal dataset stats')
    parser.add_argument('--n_mol', type=int, default=1000, help='number of unique smiles/molecules')
    parser.add_argument('--n_conf', type=int, default=1, help='number of conformers of each molecule')
    parser.add_argument('--n_upper', type=int, default=1000, help='upper bound for number of conformers')
    parser.add_argument('--type_view', type=str, default='fuse', help='type of processed dataset',
                        choices=['3D', 'fingerprint', 'smiles', 'fuse'])
    args = parser.parse_args()

    if args.sum:
        sum_list = summarise()
        with open('../datasets/summarise.json', 'w') as fout:
            json.dump(sum_list, fout)

    else:
        n_mol, n_conf, n_upper = args.n_mol, args.n_conf, args.n_upper
        if args.type_view == '3D':
        # n_mol, n_conf = 1000000, 5
            root = '../datasets/GEOM/processed/GEOM_FULL_nmol%d_nconf%d_nupper%d/' % (n_mol, n_conf, n_upper)
        elif args.type_view == 'fingerprint':
            root = '../datasets/GEOM/processed/GEOM_fp_nmol%d_nconf%d_nupper%d/' % (n_mol, n_conf, n_upper)
        elif args.type_view == 'smiles':
            root = '../datasets/GEOM/processed/GEOM_smiles_nmol%d/' % (args.n_mol)
        elif args.type_view == 'fuse':
            # root = '../datasets/GEOM/processed/Final_GEOM_FULL_nmol%d_nconf%d/' % (n_mol, n_conf)
            root = '../datasets/GEOM/processed/Case_Study_nmol_Roberta_%d/' % (n_mol)

        # ls = Descriptors.descList
        # funcs = []
        # for pair in ls:
        #     if pair[0][0:3] == 'fr_':
        #         funcs.append(pair[1])
        # Generate 3D Datasets (2D SMILES + 3D Conformer)
        MoleculeDataset(root=root, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper, type_view = 'fuse')
        # Generate Fingerprint Datasets (2D SMILES + Fingerprints)
        # MoleculeDataset(root=root_3d, n_mol=n_mol, n_conf=n_conf, n_upper=n_upper, type_view = args.type_view)



