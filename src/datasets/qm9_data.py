import os
import pickle
from itertools import chain, repeat
from pathlib import Path
from tqdm import tqdm
import pdb

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
from ase.units import Hartree, eV, Bohr, Ang
from torch.nn.utils.rnn import pad_sequence

have_position = True
conversions = [1., Bohr ** 3 / Ang ** 3,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Bohr ** 2 / Ang ** 2, Hartree / eV,
                   Hartree / eV, Hartree / eV, Hartree / eV,
                   Hartree / eV, 1.]
names = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve','U0', 'U', 'H', 'G', 'Cv']
allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def get_thermo_dict():
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.
    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    fpath = '/home/chendingshuo/MEMO/datasets/qm9/atomref.txt'


    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(fpath, 'r') as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    return therm_energy

def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}
    print(charge_counts.keys())

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts


def mol_to_graph_data_obj_simple(mol, mol_dict=None):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 1  # atom type
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = atom_to_feature_vector(atom)
        atom_feature[0] += 1
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

    
    
    # positions = []
    # for atom in mol_dict[1:mol_dict[0]+1]:
    #     positions.append([atom['position']['x'], atom['position']['y'], atom['position']['z']])
    # positions = torch.tensor(positions, dtype = torch.float) 
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


def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    """ Inverse of mol_to_graph_data_obj_simple() """
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx = atom_features[i]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        atom = Chem.Atom(atomic_num)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)

    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol


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
        atomic_num_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx,
                       bond_type_idx=bond_type_idx)

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
        atom_feature = [node['atom_num_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 1  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx']]
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
        if self.dataset == 'qm9':
            smiles_list, rdkit_mol_objs, labels, mol_list = \
                _load_qm9_dataset()
            data_list = []
            data_smiles_list = []
            roberta_tensor = self.roberta_tensor
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol, mol_list[i])
                fingerprint = np.array(GetMorganFingerprintAsBitVect(rdkit_mol, 2, nBits=1024))
                data.fingerprint = torch.tensor(fingerprint, dtype = torch.float).unsqueeze(0)
                data.roberta_tensor = roberta_tensor[i].unsqueeze(0)
                data.id = torch.tensor([i])
               
                # All the labels
                data.mu = torch.tensor(labels['mu'][i]).unsqueeze(0).float() 
                data.alpha = torch.tensor(labels['alpha'][i]).unsqueeze(0).float()
                data.homo = torch.tensor(labels['homo'][i]).unsqueeze(0).float() 
                data.lumo = torch.tensor(labels['lumo'][i]).unsqueeze(0).float()
                data.gap = torch.tensor(labels['gap'][i]).unsqueeze(0).float() 
                data.r2 = torch.tensor(labels['r2'][i]).unsqueeze(0).float()
                data.zpve = torch.tensor(labels['zpve'][i]).unsqueeze(0).float() 
                data.U0 = torch.tensor(labels['U0'][i]).unsqueeze(0).float() 
                data.U = torch.tensor(labels['U'][i]).unsqueeze(0).float()
                data.H = torch.tensor(labels['H'][i]).unsqueeze(0).float() 
                data.G = torch.tensor(labels['G'][i]).unsqueeze(0).float() 
                data.Cv = torch.tensor(labels['Cv'][i]).unsqueeze(0).float()
                
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        else:
            raise ValueError('Dataset {} not included.'.format(self.dataset))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_smiles_series = pd.Series(data_smiles_list)
        saver_path = os.path.join(self.processed_dir, 'smiles.csv')
        data_smiles_series.to_csv(saver_path, index=False, header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return


# ============= Load qm9 datasets and processing ============
def read_one_xyz(filename):

    '''
    Read raw QM9 dataset (*.xyz format) and store corresponding information into a list, format is as follow:
    mol[0]     : atom number (int)
    mol[1:-2]  : atom type and positions (dict)
    mol[-2:]   : Calculated properties (dict)
    mol[-1]    : SMILES string
    '''

    mol = []
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
    with open(filename, 'r') as f:
        content = f.read()       
        contact = content.split('\n')
        atom_nums = 0
        atom_charges = []
        
        for line in contact[2:-4]:
            atom = line.split()
            item = {'atom_type': atom[0], 'position': {"x": round(float(atom[1].replace('*^', 'e')),4), 
                    "y": round(float(atom[2].replace('*^', 'e')),4), "z": round(float(atom[3].replace('*^', 'e')),4)}}
            atom_nums += 1
            atom_charges.append(id2charge[atom[0]])
            mol.append(item)
        
        mol.insert(0, atom_nums)
        mol.append(atom_charges)

        properties = contact[1].split()[5:]
        values = [float(val.replace('*^', 'e')) for val in properties]
        labels = dict(zip(names, values)) 
        mol.append(labels)

        smiles = contact[-3].split()[0]
        mol.append(smiles)
        
    return mol

def get_needless_index():

    '''Get the index of uncharaterized molecules'''

    idx_list = []
    fname = '/home/chendingshuo/MEMO/datasets/qm9/uncharacterized.txt'
    with open(fname, 'r') as f:
        content = f.read().split('\n')
        for line in content:
            if line == '' or line.isdigit():
                continue
            else:
                index = int(line.split()[0])
                idx_list.append(index)
    assert len(idx_list) == 3054
    return idx_list

def _load_qm9_dataset():
    mol_list = []
    labels = {}
    idx_list = get_needless_index()
    dir = Path('/home/chendingshuo/MEMO/datasets/qm9/origin/')
    flist = list(dir.glob('*.xyz'))
    flist.sort()

    cnt = 0
    for p in tqdm(flist): 
        cnt += 1   
        if cnt in idx_list: continue
        else: 
            mol = read_one_xyz(p)
            mol_list.append(mol)
    assert len(mol_list) == 130831

    smiles_list = [m[-1] for m in mol_list]
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    rdkit_mol_list = [Chem.AddHs(mol) for mol in rdkit_mol_objs_list]
    list_charges = [torch.tensor(m[-3]) for m in mol_list]
    mol_charges = pad_sequence(list_charges, batch_first=True) # Pad to the same length with token '0'
    
     # Get the charge and number of charges
    charge_counts = get_unique_charges(mol_charges)
    
    # Now, loop over the targets with defined thermochemical energy       
    labels['mu'] = [m[-2]['mu'] for m in mol_list]
    labels['alpha'] = [m[-2]['alpha'] for m in mol_list]
    labels['homo'] = [m[-2]['homo'] for m in mol_list]
    labels['lumo'] = [m[-2]['lumo'] for m in mol_list]
    labels['gap'] = [m[-2]['gap'] for m in mol_list]
    labels['r2'] = [m[-2]['r2'] for m in mol_list]
    labels['zpve'] = [m[-2]['zpve'] for m in mol_list]
    labels['U0'] = [m[-2]['U0'] for m in mol_list]
    labels['U'] = [m[-2]['U'] for m in mol_list]
    labels['H'] = [m[-2]['H'] for m in mol_list]
    labels['G'] = [m[-2]['G'] for m in mol_list]
    labels['Cv'] = [m[-2]['Cv'] for m in mol_list]
    # pdb.set_trace()
    for target, target_therm in thermo_dict.items():
        thermo = np.zeros(len(labels[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:  # Skip the pad token '0'
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        labels[target] -= thermo

    # for key in labels.keys():
    #     if key in qm9_to_eV:
    #         labels[key] *= qm9_to_eV[key]

      
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels['mu'])
    
    return smiles_list, rdkit_mol_list, labels, mol_list
# ===========================================================



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
        'qm9'
    ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        if dataset_name is not 'qm9':
            root = "../../datasets/molecule_datasets/" + dataset_name
            print('root\t', root)
            os.makedirs(root + "/processed", exist_ok=True)
        elif dataset_name == 'qm9':
            root = "../../datasets/qm9/"
            print('root\t', root)
            os.makedirs(root + "/processed", exist_ok=True)
        
        if dataset_name == 'qm9':
            smiles_list, rdkit_mol_objs, labels, _ = _load_qm9_dataset() 
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

        for step, batch in enumerate(tqdm(loader)):
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
    thermo_dict = get_thermo_dict()
    create_all_datasets()
    