# Pre-training Experiment

## I Dataset

**Pre-training dataset**

- GEOM: Randomly select 50k qualified molecules from GEOM with both 2D and 3D structures for pre-training.



**Fine-tune dataset**

- Tox21, HIV, BBBP, Sider, ClinTox, Muv etc.



## II Model

**Pre-training**

- **2D GNN** : GIN as the backbone model.

​	Parameters and data dimension in data loader:

| Input Dimension | Embedding Dimension |
| --------------- | ------------------- |
| $N \times 2$    | $N \times 300$      |

- **3D GNN** : SchNet as the backbone model.

| Input Dimension                          | Embedding Dimension |
| ---------------------------------------- | ------------------- |
| $N \times 1$ and $positions \ N\times 3$ | $N \times 300$      |

```bash
cd src
python pretrain.py --dataset=GEOM_3D_nmol50000_nconf5_nupper1000
```



**Fine-Tune**

```bash
cd src
python finetune.py --dataset=tox21
```



