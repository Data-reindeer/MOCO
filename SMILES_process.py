import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import pandas as pd

from paras import args


def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]\n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    print('===start===\n')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)

    data_root = '../datasets/GEOM/processed/New_GEOM_FUSE_nmol50000_nconf5/'

    input_path = data_root + 'processed/smiles.csv'
    smiles = pd.read_csv(input_path, sep=',', header=None)[0].tolist()
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    roberta_dict = tokenizer(smiles,return_tensors="pt", padding='longest')

    dataset = TensorDataset(roberta_dict['input_ids'], roberta_dict['attention_mask'])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = RobertaModel.from_pretrained("./GeomBerta").to(device)
    smiles_repr = None
    

    l = tqdm(loader)
    for step, batch in enumerate(l):
        ids = batch[0].to(device)
        att_mask = batch[1].to(device)
        with torch.no_grad():
            reprs = model(input_ids=ids, attention_mask=att_mask).last_hidden_state
            molecule_repr = reprs[:,0,:]
        molecule_repr = molecule_repr.cpu()
        if step == 0:
            smiles_repr = molecule_repr
        else:
            smiles_repr = torch.cat((smiles_repr, molecule_repr), dim = 0)

    torch.save(smiles_repr, "./Roberta_tensor_New.pt")


    
