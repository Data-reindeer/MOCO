import os
import pickle
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from transformers import RobertaModel, RobertaTokenizer
from torch.utils import data
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)


def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 3  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    AllChem.EmbedMolecule(mol)
    conformer_tuple = mol.GetConformers()            
    if len(conformer_tuple) > 0:
        conformer = conformer_tuple[0]
        positions = conformer.GetPositions()
    else:
        AllChem.Compute2DCoords(mol)
        conformer = mol.GetConformers()[0]
        positions = conformer.GetPositions() 
     # Positions are not pre-produced
    positions = torch.Tensor(positions)
    

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, positions=positions)

    return data


def graph_data_obj_to_nx_simple(data):
    """ torch geometric -> networkx
    NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: networkx object """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        temp_feature = atom_features[i]
        G.add_node(
            i,
            x0=temp_feature[0],
            x1=temp_feature[1],
            x2=temp_feature[2],
            x3=temp_feature[3],
            x4=temp_feature[4],
            x5=temp_feature[5],
            x6=temp_feature[6],
            x7=temp_feature[7],
            x8=temp_feature[8])
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        temp_feature= edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       e0=temp_feature[0],
                       e1=temp_feature[1],
                       e2=temp_feature[2])

    return G


def nx_to_graph_data_obj_simple(G):
    """ vice versa of graph_data_obj_to_nx_simple()
    Assume node indices are numbered from 0 to num_nodes - 1.
    NB: Uses simplified atom and bond features, and represent as indices.
    NB: possible issues with recapitulating relative stereochemistry
        since the edges in the nx object are unordered. """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['x0'], node['x1'], node['x2'], node['x3'], node['x4'], node['x5'], node['x6'], node['x7'], node['x8']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['e0'], edge['e1'], edge['e2']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def create_standardized_mol_id(smiles):
    """ smiles -> inchi """

    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol is not None:
            # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)\
            # c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles:  # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
    return


#todo: prune
class MoleculeDatasetComplete(InMemoryDataset):
    def __init__(self, root, roberta_tensor=None, dataset='zinc250k', transform=None,
                 pre_transform=None, pre_filter=None, empty=False):

        self.root = root
        self.dataset = dataset
        self.transform = transform
        self.pre_filter = pre_filter
        self.pre_transform = pre_transform
        self.roberta_tensor = roberta_tensor

        super(MoleculeDatasetComplete, self).__init__(root, transform, pre_transform, pre_filter)

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])
        print('Dataset: {}\nData: {}'.format(self.dataset, self.data))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        if self.dataset == 'davis':
            file_name_list = ['davis']
        elif self.dataset == 'kiba':
            file_name_list = ['kiba']
        else:
            file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        return

    def process(self):

        def shared_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list = []
            data_smiles_list = []
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                # sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

            return data_list, data_smiles_list

        def shared_single_extractor(smiles_list, rdkit_mol_objs, labels):
            data_list = []
            data_smiles_list = []
            roberta_tensor = self.roberta_tensor
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                fingerprint = np.array(GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=1024))
                data.fingerprint = torch.tensor(fingerprint, dtype = torch.float).unsqueeze(0)
                data.roberta_tensor = roberta_tensor[i].unsqueeze(0)
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i]).unsqueeze(0).float()
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

            return data_list, data_smiles_list

        if self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_single_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_single_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_single_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'malaria':
            smiles_list, rdkit_mol_objs, labels = \
                _load_malaria_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_single_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'cep':
            smiles_list, rdkit_mol_objs, labels = \
                _load_cep_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_single_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'pcba':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)
        
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'toxcast':
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            data_list, data_smiles_list = shared_extractor(
                smiles_list, rdkit_mol_objs, labels)

        elif self.dataset == 'geom':
            input_path = self.raw_paths[0]
            data_list, data_smiles_list = [], []
            input_df = pd.read_csv(input_path, sep=',', dtype='str')
            smiles_list = list(input_df['smiles'])
            for i in range(len(smiles_list)):
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol is not None:
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    data.id = torch.tensor([i])
                    data_list.append(data)
                    data_smiles_list.append(s)

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # For ESOL and FreeSOlv, there are molecules with single atoms and empty edges.
        valid_index = []
        neo_data_smiles_list, neo_data_list = [], []
        for i, (smiles, data) in enumerate(zip(data_smiles_list, data_list)):
            if data.edge_attr.size()[0] == 0:
                print('Invalid\t', smiles, data)
                continue
            valid_index.append(i)
            assert data.edge_attr.size()[1] == 3
            assert data.edge_index.size()[0] == 2
            assert data.x.size()[1] == 9
            assert data.id.size()[0] == 1
            assert data.y.size()[0] == 1
            neo_data_smiles_list.append(smiles)
            neo_data_list.append(data)

        old_N = len(data_smiles_list)
        neo_N = len(valid_index)
        print('{} invalid out of {}.'.format(old_N - neo_N, old_N))
        print(len(neo_data_smiles_list), '\t', len(neo_data_list))

        data_smiles_series = pd.Series(neo_data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(neo_data_list)
        torch.save((data, slices), self.processed_paths[0])

        return

def create_circular_fingerprint(mol, radius, size, chirality):
    """ :return: np array of morgan fingerprint """
    fp = GetMorganFingerprintAsBitVect(
        mol, radius, nBits=size, useChirality=chirality)
    return np.array(fp)


class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
            processed_fp dir. The raw dir should contain the SMILES files,
            and the processed_fp dir can either be empty
            or a previously processed file
        :param dataset: name of dataset. Currently only implemented for
            tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information """

        self.root = root
        self.size = size
        self.radius = radius
        self.dataset = dataset
        self.chirality = chirality

        self._load()

    def _process(self):

        data_list, data_smiles_list = [], []
        if self.dataset == 'chembl_with_labels':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
            print('processing')
            for i in range(len(rdkit_mol_objs)):
                # print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is not None:
                    # # convert aromatic bonds to double bonds
                    fp_arr = create_circular_fingerprint(
                        rdkit_mol, self.radius, self.size, self.chirality)
                    fp_arr = torch.tensor(fp_arr)
                    id = torch.tensor([i])
                    y = torch.tensor(labels[i, :])
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    data_list.append({'fp_arr': fp_arr,
                                      'fold':   fold,
                                      'id':     id,
                                      'y':      y})
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(os.path.join(self.root, 'raw/tox21.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                # print(i)
                rdkit_mol = rdkit_mol_objs[i]
                fp_arr = create_circular_fingerprint(
                    rdkit_mol, self.radius, self.size, self.chirality)
                fp_arr = torch.tensor(fp_arr)
                id = torch.tensor([i])
                y = torch.tensor(labels[i, :])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'hiv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(os.path.join(self.root, 'raw/HIV.csv'))
            print('processing')
            for i in range(len(smiles_list)):
                # print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(
                    rdkit_mol, self.radius, self.size, self.chirality)
                fp_arr = torch.tensor(fp_arr)
                id = torch.tensor([i])
                y = torch.tensor([labels[i]])
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        # save processed data objects and smiles
        processed_dir = os.path.join(self.root, 'processed_fp')
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False, header=False)
        with open(os.path.join(processed_dir, 'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        file_name_list = os.listdir(processed_dir)
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(
                    processed_dir, 'fingerprint_data_processed.pkl'),
                    'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps,
        # save, then reload
        else:
            self._process()
            self._load()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # if iterable class is passed, return dataset objection
        if hasattr(index, "__iter__"):
            dataset = MoleculeFingerprintDataset(self.root, self.dataset, self.radius, self.size,
                                                 chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            return dataset
        else:
            return self.data_list[index]


def _load_tox21_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_hiv_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['HIV_active']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_bace_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values


def _load_bbbp_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

def _load_esol_dataset(input_path):
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

def _load_freesolv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_malaria_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['activity']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_cep_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['PCE']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_muv_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
             'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
             'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values


def check_columns(df, tasks, N):
    bad_tasks = []
    total_missing_count = 0
    for task in tasks:
        value_list = df[task]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        total_missing_count += missing_count
        pos_ratio = 100. * pos_count / (pos_count + neg_count)
        missing_ratio = 100. * missing_count / N
        assert pos_count + neg_count + missing_count == N
        if missing_ratio >= 50:
            bad_tasks.append(task)
        print('task {}\t\tpos_ratio: {:.5f}\tmissing ratio: {:.5f}'.format(task, pos_ratio, missing_ratio))
    print('total missing ratio: {:.5f}'.format(100. * total_missing_count / len(tasks) / N))
    return bad_tasks


def check_rows(labels, N):
    from collections import defaultdict
    p, n, m = defaultdict(int), defaultdict(int), defaultdict(int)
    bad_count = 0
    for i in range(N):
        value_list = labels[i]
        pos_count = sum(value_list == 1)
        neg_count = sum(value_list == -1)
        missing_count = sum(value_list == 0)
        p[pos_count] += 1
        n[neg_count] += 1
        m[missing_count] += 1
        if pos_count + neg_count == 0:
            bad_count += 1
    print('bad_count\t', bad_count)
    
    print('pos\t', p)
    print('neg\t', n)
    print('missing\t', m)
    return


def _load_pcba_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    tasks = list(input_df.columns)[:-2]

    N = input_df.shape[0]
    temp_df = input_df[tasks]
    temp_df = temp_df.replace(0, -1)
    temp_df = temp_df.fillna(0)

    bad_tasks = check_columns(temp_df, tasks, N)
    for task in bad_tasks:
        tasks.remove(task)
    print('good tasks\t', len(tasks))

    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    labels = labels.fillna(0)
    labels = labels.values
    print(labels.shape)  # 439863, 92
    check_rows(labels, N)

    input_df.dropna(subset=tasks, how='all', inplace=True)
    # convert 0 to -1
    input_df = input_df.replace(0, -1)
    # convert nan to 0
    input_df = input_df.fillna(0)
    labels = input_df[tasks].values
    print(input_df.shape)  # 435685, 92
    N = input_df.shape[0]
    check_rows(labels, N)

    smiles_list = input_df['smiles'].tolist()
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_sider_dataset(input_path):

    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.value


def _load_toxcast_dataset(input_path):

    # NB: some examples have multiple species, some example smiles are invalid
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    preprocessed_rdkit_mol_objs_list = [m if m is not None else None
                                        for m in rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                                for m in preprocessed_rdkit_mol_objs_list]
    tasks = list(input_df.columns)[1:]
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, \
           preprocessed_rdkit_mol_objs_list, labels.values


def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of MLs methods for drug target prediction on ChEMBL'
    :param root_path: folder that contains the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f = open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds = pickle.load(f)
    f.close()

    f = open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat = pickle.load(f)
    sampleAnnInd = pickle.load(f)
    targetAnnInd = pickle.load(f)
    f.close()

    targetMat = targetMat
    targetMat = targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd = targetAnnInd
    targetAnnInd = targetAnnInd - targetAnnInd.min()

    folds = [np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed = targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall = np.array([np.sum(targetMatTransposed[x].data > 0.5)
                                for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall = np.array([np.sum(targetMatTransposed[x].data < -0.5)
                                for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData = targetMat.A  # possible values are {-1, 0, 1}

    # 2. load structures
    f = open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr = pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m is None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m is not None else None
                   for m in preprocessed_rdkitArr]
    # bc some empty mol in the rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData


# root_path = 'dataset/chembl_with_labels'
def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively """

    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list


def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one """

    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def create_all_datasets():

    downstream_dir = [
        # 'bace',
        # 'bbbp',
        # 'clintox',
        # 'esol',
        # 'freesolv',
        # 'hiv',
        # 'lipophilicity',
        # 'muv',
        # 'sider',
        # 'tox21',
        # 'toxcast',
        'cep',
    ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "../../datasets/molecule_datasets/" + dataset_name
        print('root\t', root)
        os.makedirs(root + "/processed", exist_ok=True)
        

        if dataset_name == 'esol':
            smiles_list, rdkit_mol_objs, labels = _load_esol_dataset(root+'/raw/'+dataset_name+'.csv')
        elif dataset_name == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = _load_lipophilicity_dataset(root+'/raw/'+dataset_name+'.csv')
        elif dataset_name == 'malaria':
            smiles_list, rdkit_mol_objs, labels = _load_malaria_dataset(root+'/raw/'+dataset_name+'.csv')
        elif dataset_name == 'cep':
            smiles_list, rdkit_mol_objs, labels = _load_cep_dataset(root+'/raw/'+dataset_name+'.csv')
        else:
            raise ValueError('Dataset not supported!')
        
        

        for i, string in enumerate(smiles_list):
            if not isinstance(string, str):print('===== {} ====='.format(i))
        smiles_list = list(filter(None, smiles_list)) 
        dicts = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=510)
        input_ids = dicts['input_ids']
        mask = dicts['attention_mask']
        roberta_data = TensorDataset(input_ids, mask)
        loader = DataLoader(roberta_data, batch_size=128, shuffle=False)
        smiles_repr = None

        for step, batch in enumerate(loader):
            ids = batch[0].to(device)
            att_mask = batch[1].to(device)

            with torch.no_grad():
                tensor_d = Geomberta(input_ids=ids, attention_mask=att_mask).pooler_output
            tensor_d = tensor_d.cpu()
            if step == 0:
                smiles_repr = tensor_d
            else:
                smiles_repr = torch.cat((smiles_repr, tensor_d), dim = 0)
                

        dataset = MoleculeDatasetComplete(root, dataset=dataset_name, roberta_tensor=smiles_repr)
        print(dataset)


# test MoleculeDataset object
if __name__ == "__main__":
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    Geomberta = RobertaModel.from_pretrained("../GeomBerta").to(device)
    create_all_datasets()
    

'''
task PCBA-1030		pos_ratio: 9.88062	missing ratio: 63.20854
task PCBA-1379		pos_ratio: 0.28422	missing ratio: 54.96621
task PCBA-1452		pos_ratio: 0.11968	missing ratio: 65.99737
task PCBA-1454		pos_ratio: 0.42874	missing ratio: 71.47203
task PCBA-1457		pos_ratio: 0.35595	missing ratio: 53.88678
task PCBA-1458		pos_ratio: 2.96564	missing ratio: 55.41498
task PCBA-1460		pos_ratio: 2.53771	missing ratio: 48.97184
task PCBA-1461		pos_ratio: 1.10730	missing ratio: 52.63434
task PCBA-1468		pos_ratio: 0.41446	missing ratio: 42.62395
task PCBA-1469		pos_ratio: 0.06284	missing ratio: 37.41142
task PCBA-1471		pos_ratio: 0.13301	missing ratio: 50.26224
task PCBA-1479		pos_ratio: 0.29088	missing ratio: 37.94363
task PCBA-1631		pos_ratio: 0.34316	missing ratio: 40.90569
task PCBA-1634		pos_ratio: 0.05875	missing ratio: 40.40349
task PCBA-1688		pos_ratio: 1.16394	missing ratio: 53.55236
task PCBA-1721		pos_ratio: 0.37387	missing ratio: 33.90215
task PCBA-2100		pos_ratio: 0.39686	missing ratio: 33.37676
task PCBA-2101		pos_ratio: 0.09378	missing ratio: 29.45190
task PCBA-2147		pos_ratio: 1.82161	missing ratio: 56.25638
task PCBA-2242		pos_ratio: 0.38838	missing ratio: 58.14651
task PCBA-2326		pos_ratio: 0.40984	missing ratio: 40.64629
task PCBA-2451		pos_ratio: 0.73574	missing ratio: 37.45871
task PCBA-2517		pos_ratio: 0.34169	missing ratio: 23.68374
task PCBA-2528		pos_ratio: 0.19193	missing ratio: 21.23093
task PCBA-2546		pos_ratio: 3.79670	missing ratio: 36.66573
task PCBA-2549		pos_ratio: 0.52730	missing ratio: 47.31382
task PCBA-2551		pos_ratio: 6.16759	missing ratio: 38.49744
task PCBA-2662		pos_ratio: 0.03855	missing ratio: 35.12344
task PCBA-2675		pos_ratio: 0.03977	missing ratio: 43.41329
task PCBA-2676		pos_ratio: 0.30185	missing ratio: 18.50758
task PCBA-411		pos_ratio: 2.21310	missing ratio: 83.94386
task PCBA-463254		pos_ratio: 0.01245	missing ratio: 25.15533
task PCBA-485281		pos_ratio: 0.08101	missing ratio: 28.43454
task PCBA-485290		pos_ratio: 0.28100	missing ratio: 22.89736
task PCBA-485294		pos_ratio: 0.04774	missing ratio: 29.52465
task PCBA-485297		pos_ratio: 2.94172	missing ratio: 29.42553
task PCBA-485313		pos_ratio: 2.42935	missing ratio: 29.12111
task PCBA-485314		pos_ratio: 1.41913	missing ratio: 27.87823
task PCBA-485341		pos_ratio: 0.52804	missing ratio: 25.55955
task PCBA-485349		pos_ratio: 0.19339	missing ratio: 27.23121
task PCBA-485353		pos_ratio: 0.18665	missing ratio: 26.55304
task PCBA-485360		pos_ratio: 0.68194	missing ratio: 50.32703
task PCBA-485364		pos_ratio: 3.12950	missing ratio: 22.19691
task PCBA-485367		pos_ratio: 0.17077	missing ratio: 25.84941
task PCBA-492947		pos_ratio: 0.02429	missing ratio: 25.11691
task PCBA-493208		pos_ratio: 0.82140	missing ratio: 90.53433
task PCBA-504327		pos_ratio: 0.20893	missing ratio: 15.45368
task PCBA-504332		pos_ratio: 10.31124	missing ratio: 32.34530
task PCBA-504333		pos_ratio: 4.81269	missing ratio: 25.88783
task PCBA-504339		pos_ratio: 4.74540	missing ratio: 19.11186
task PCBA-504444		pos_ratio: 2.55289	missing ratio: 33.86895
task PCBA-504466		pos_ratio: 1.34180	missing ratio: 29.31322
task PCBA-504467		pos_ratio: 3.14422	missing ratio: 44.69369
task PCBA-504706		pos_ratio: 0.06639	missing ratio: 31.16993
task PCBA-504842		pos_ratio: 0.03110	missing ratio: 26.17633
task PCBA-504845		pos_ratio: 0.03059	missing ratio: 15.26839
task PCBA-504847		pos_ratio: 0.93284	missing ratio: 13.28800
task PCBA-504891		pos_ratio: 0.00941	missing ratio: 17.83169
task PCBA-540276		pos_ratio: 2.24105	missing ratio: 53.22362
task PCBA-540317		pos_ratio: 0.57715	missing ratio: 15.86085
task PCBA-588342		pos_ratio: 7.66499	missing ratio: 25.66913
task PCBA-588453		pos_ratio: 1.06236	missing ratio: 15.34182
task PCBA-588456		pos_ratio: 0.01322	missing ratio: 12.31793
task PCBA-588579		pos_ratio: 0.51543	missing ratio: 11.21099
task PCBA-588590		pos_ratio: 1.10797	missing ratio: 18.51940
task PCBA-588591		pos_ratio: 1.26011	missing ratio: 14.73550
task PCBA-588795		pos_ratio: 0.34784	missing ratio: 13.85818
task PCBA-588855		pos_ratio: 1.40143	missing ratio: 19.74842
task PCBA-602179		pos_ratio: 0.09434	missing ratio: 12.28246
task PCBA-602233		pos_ratio: 0.04351	missing ratio: 13.78520
task PCBA-602310		pos_ratio: 0.07864	missing ratio: 10.38596
task PCBA-602313		pos_ratio: 0.20453	missing ratio: 15.18768
task PCBA-602332		pos_ratio: 0.01718	missing ratio: 6.03574
task PCBA-624170		pos_ratio: 0.20997	missing ratio: 9.37451
task PCBA-624171		pos_ratio: 0.31369	missing ratio: 9.98879
task PCBA-624173		pos_ratio: 0.12244	missing ratio: 8.83480
task PCBA-624202		pos_ratio: 1.08398	missing ratio: 16.67428
task PCBA-624246		pos_ratio: 0.02767	missing ratio: 17.00916
task PCBA-624287		pos_ratio: 0.13976	missing ratio: 31.19153
task PCBA-624288		pos_ratio: 0.41799	missing ratio: 26.24681
task PCBA-624291		pos_ratio: 0.06686	missing ratio: 24.51536
task PCBA-624296		pos_ratio: 3.37351	missing ratio: 33.51225
task PCBA-624297		pos_ratio: 2.01949	missing ratio: 29.91113
task PCBA-624417		pos_ratio: 1.97002	missing ratio: 25.92375
task PCBA-651635		pos_ratio: 1.09551	missing ratio: 21.05837
task PCBA-651644		pos_ratio: 0.21114	missing ratio: 19.35307
task PCBA-651768		pos_ratio: 0.46886	missing ratio: 18.68514
task PCBA-651965		pos_ratio: 1.96843	missing ratio: 25.79440
task PCBA-652025		pos_ratio: 0.06531	missing ratio: 17.15534
task PCBA-652104		pos_ratio: 1.89989	missing ratio: 14.54976
task PCBA-652105		pos_ratio: 1.26379	missing ratio: 26.69445
task PCBA-652106		pos_ratio: 0.13773	missing ratio: 17.46794
task PCBA-686970		pos_ratio: 1.76886	missing ratio: 23.37296
task PCBA-686978		pos_ratio: 20.82252	missing ratio: 31.07695
task PCBA-686979		pos_ratio: 15.85013	missing ratio: 29.53124
task PCBA-720504		pos_ratio: 2.90285	missing ratio: 20.30428
task PCBA-720532		pos_ratio: 7.74621	missing ratio: 97.04749
task PCBA-720542		pos_ratio: 0.20592	missing ratio: 18.85155
task PCBA-720551		pos_ratio: 0.36889	missing ratio: 22.03868
task PCBA-720553		pos_ratio: 0.96084	missing ratio: 22.86507
task PCBA-720579		pos_ratio: 0.67826	missing ratio: 35.47673
task PCBA-720580		pos_ratio: 0.49409	missing ratio: 30.29080
task PCBA-720707		pos_ratio: 0.07372	missing ratio: 17.35540
task PCBA-720708		pos_ratio: 0.18522	missing ratio: 18.74493
task PCBA-720709		pos_ratio: 0.14630	missing ratio: 19.66158
task PCBA-720711		pos_ratio: 0.07977	missing ratio: 17.35313
task PCBA-743255		pos_ratio: 0.24330	missing ratio: 15.71535
task PCBA-743266		pos_ratio: 0.07667	missing ratio: 9.26584
task PCBA-875		pos_ratio: 0.04468	missing ratio: 83.21023
task PCBA-881		pos_ratio: 0.56798	missing ratio: 76.26397
task PCBA-883		pos_ratio: 15.16303	missing ratio: 98.14533
task PCBA-884		pos_ratio: 32.69029	missing ratio: 97.61949
task PCBA-885		pos_ratio: 1.28445	missing ratio: 97.07955
task PCBA-887		pos_ratio: 1.47660	missing ratio: 84.20326
task PCBA-891		pos_ratio: 20.01775	missing ratio: 98.20671
task PCBA-899		pos_ratio: 22.25048	missing ratio: 98.12305
task PCBA-902		pos_ratio: 1.58359	missing ratio: 72.95294
task PCBA-903		pos_ratio: 0.64028	missing ratio: 87.99876
task PCBA-904		pos_ratio: 1.03611	missing ratio: 88.41457
task PCBA-912		pos_ratio: 0.80136	missing ratio: 87.06347
task PCBA-914		pos_ratio: 2.80935	missing ratio: 98.21967
task PCBA-915		pos_ratio: 5.47758	missing ratio: 98.16966
task PCBA-924		pos_ratio: 0.95768	missing ratio: 72.72378
task PCBA-925		pos_ratio: 0.06077	missing ratio: 85.40955
task PCBA-926		pos_ratio: 0.61732	missing ratio: 87.11030
task PCBA-927		pos_ratio: 0.10404	missing ratio: 86.67085
task PCBA-938		pos_ratio: 2.84961	missing ratio: 85.77512
task PCBA-995		pos_ratio: 1.06553	missing ratio: 85.04330
39.42927080420495
'''