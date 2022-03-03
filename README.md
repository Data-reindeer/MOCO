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



## III Parameters 

Results for molecular property prediction tasks (Test based on best ROC-AUC model on the validation set.)

| Dataset | Embedding Dimension | Epochs | Learning rate | GIN layers | Dropout ratio | ROC-AUC               |
| ------- | ------------------- | ------ | ------------- | ---------- | ------------- | --------------------- |
| Tox21   | 300                 | 100    | 1e-3          | 5          | 0.5           | 0.6857                |
| Tox21   | 300                 | 200    | 1e-3          | 5          | 0.5           | 0.7215                |
| HIV     | 300                 | 300    | 1e-3          | 5          | 0.5           | 0.7619（best : 0.79） |
| BBBP    | 300                 | 200    | 1e-3          | 5          | 0.5           | 0.5775                |
| Sider   | 300                 | 200    | 1e-3          | 5          | 0.5           | 0.5842                |
| ClinTox | 300                 | 200    | 1e-3          | 5          | 0.5           | 0.5500                |
| MUV     | 300                 | 200    | 1e-3          | 5          | 0.5           | 0.6791（best：0.75）  |

- bbbp、ClinTox的OOD问题比较严重，train 200个epochs就可以到0.92+，bbbp的validation可以达到0.8+，但是test最好到0.65左右。
- Sider、Tox21还有提升空间，200轮还在稳步上升中。

bbbp和ClinTox小数据集的训练效果就很差。

