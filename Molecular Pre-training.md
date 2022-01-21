# **Molecular Pre-training**

A simple pre-training framework for molecular property prediction with GNN. Molecules are encoded by 2D network (without 3D information) and 3D network (only with 3D information). 

- **Dataset**

QM9 (50k for pre-training, 50k for fine-tuning, rest for testing) 

- **Framework Modules**

| 2D Network             | 3D Network | Contrastive Loss |
| ---------------------- | ---------- | ---------------- |
| vanilla GNN (7 layers) | EGNN       | GRACE            |

