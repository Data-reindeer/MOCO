# Featurization Matters: A Multiview Contrastive Learning Approach for Molecular Pre-training

Authors:  Yanqiao Zhu, Dingshuo Chen, Yuanqi Du, Yingze Wang, Qiang Liu, Shu Wu

This repository provides the source code for the paper **Featurization Matters: A Multiview Contrastive Learning Approach for Molecular Pre-training**. Here we consider four kinds of views:

- 2D Graph 
- 3D Geometry
- Morgan Fingerprint
- SMILES String



## Environments

```bash
numpy				1.21.2
networkx 			2.6.3
scikit-learn        1.0.2
pandas              1.3.4
python 				3.7.11
torch 				1.10.2+cu113
torch-geometric		2.0.3
transformers		4.17.0
rdkit               2020.09.1.0
```



## Dataset Preprocessing

### Datasets

- Geometric Ensemble Of Molecules (GEOM)

```bash
mkdir datasets
mkdir -p GEOM/raw
mkdir -p GEOM/processed

wget https://dataverse.harvard.edu/api/access/datafile/4327252
mv 4327252 rdkit_folder.tar.gz
tar -xvf rdkit_folder.tar.gz
```

- Chem Datasets

```bash
wget http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip
unzip chem_dataset.zip
mv dataset molecule_datasets
```

- Other Chem Datasets
  - malaria
  - cep

```bash
wget -O malaria-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-03-malaria/malaria-processed.csv
mkdir -p ./molecule_datasets/malaria/raw
mv malaria-processed.csv ./molecule_datasets/malaria/raw/malaria.csv

wget -O cep-processed.csv https://raw.githubusercontent.com/HIPS/neural-fingerprint/master/data/2015-06-02-cep-pce/cep-processed.csv
mkdir -p ./molecule_datasets/cep/raw
mv cep-processed.csv ./molecule_datasets/cep/raw/cep.csv
```



### Preprocessing

Before preprocessing the datasets, please train the RoBERTa model first and store the corresponding SMILES embedding in order to save memory cost.

```bash
cd src
python SMILES_train.py
python SMILES_process.py
```

- **GEOM preprocessing**

```bash
python dataset_preparation.py --n_mol 50000 --n_conf 5 --n_upper 1000
```

- **Downstream preprocessing** (Classification)

```bash
python molecule_preparation.py
```

- **Downstream preprocessing** (Regression)

```bash
cd src/datasets
python regression_datasets.py
```



## Experiments

- **Pre-training for classification**

```bash
cd src
python pretrain.py --dataset=Final_GEOM_FULL_nmol50000_nconf5 
```

- **Fine-tune for classification**

```bash
python finetune_supervised.py --input_model_file = '../runs/Classification_models/'
```



- **Pre-training for regression**

```bash
cd src
python pretrain_regression.py --dataset=Final_GEOM_FULL_nmol50000_nconf5
```

- **Fine-tune for regression**

```bash
python finetune_regression.py --input_model_file = '../runs/Regression_models/'
```

