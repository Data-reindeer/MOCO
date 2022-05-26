import random
import numpy as np
import pandas as pd
import os
import json
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import RobertaForMaskedLM, RobertaTokenizer, RobertaConfig

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# === Building the DataLoader ===
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # store encodings internally
        self.encodings = encodings

    def __len__(self):
        # return the number of samples
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, i):
        # return dictionary of input_ids, attention_mask, and labels for index i
        return {key: tensor[i] for key, tensor in self.encodings.items()}



if __name__ == '__main__':
    mol_num = 300000
    device_num = 0
    seed = 42
    device = torch.device('cuda:' + str(device_num)) if torch.cuda.is_available() else torch.device('cpu')
    seed_all(seed)

    # === Prepare the smiles dataset and tokenize them ===
    dir_name = '../datasets/GEOM/rdkit_folder'
    drugs_file = '{}/summary_drugs.json'.format(dir_name)
    with open(drugs_file, 'r') as f:
        drugs_summary = json.load(f)
    drugs_summary = list(drugs_summary.items())
    print('# of SMILES: {}'.format(len(drugs_summary)))

    smiles_list = []
    smiles_num = 0
    for smiles, _ in tqdm(drugs_summary):
        if smiles_num+1 >= mol_num: break
        smiles_list.append(smiles)
        smiles_num += 1

    tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    dicts = tokenizer(smiles_list, return_tensors="pt", padding='longest')

    # === Mask partial(15%) tokens === 
    labels = dicts['input_ids']
    mask = dicts['attention_mask']

    # make copy of labels tensor, this will be input_ids
    input_ids = labels.detach().clone()
    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids > 4)
    # loop through each row in input_ids tensor (cannot do in parallel)
    for i in tqdm(range(input_ids.shape[0])):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        # mask input_ids
        input_ids[i, selection] = 4  # our custom [MASK] token == 4

    # === Build the dataloader ===
    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}
    dataset = Dataset(encodings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # === Set up the MLM ===
    config = RobertaConfig(
        vocab_size = tokenizer.vocab_size,
        hidden_act = "gelu",
        hidden_dropout_prob = 0.1,
        hidden_size = 768,
        initializer_range = 0.02,
        intermediate_size = 3072,
        layer_norm_eps = 1e-12,
        model_type = "roberta",
        num_attention_heads = 12,
        num_hidden_layers = 6,
        type_vocab_size = 1,
        )
    model = RobertaForMaskedLM(config)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 4

    for epoch in range(epochs):
        loop = tqdm(loader, leave=True)
        for batch in loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    # === Save the model ===
    os.mkdir('./GeomBerta')
    model.save_pretrained('./GeomBerta')

    
    