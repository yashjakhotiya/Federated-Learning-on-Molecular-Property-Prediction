# Team 13 SysML Project Repository: Federated Learning on Molecular Property Prediction using GNNs and Transformers

Nigel Neo, Yash Jakhotiya, You Liang Tan, Zachary Minot

## Relevant paper

*insert pdf into repository when we get the chance*

## Directories

- `dataset` contains programming, documentation, and explanation of the heterogeneous fingerprint split for molecular chemical data
- `graphormer` contains the centralized running version of the Graphormer model we used for the project
- `qcf_bench` contains the quantum chemistry federated test bench with documentation to add custom datasets and models

## Data Splitting

Download and split HIV dataset
```bash
python3 dataset/pace_ice_split.py --name HIV --train_split 0.05 --method "fingerprint"
```

Here we select the "fingerprints" approach for data splitting, which yields better splitting result compared to "scaffold" approach.

For more details, checkout the readme in the dataset dir, [here](/dataset/)!

## Details of Qcf_bench

In `qcf_bench`, we use [FedML](https://github.com/FedML-AI/FedML/) framework to simulate federated learning training of GCN and Transformer with the `ogb-molhiv` dataset. In this benchmarking task, GCN model from OGB is chosen as the baseline. The proposed alternative model is a `Graphormer`. With this, we will evaluate the Graphormer with the GCN as the baseline in a federated learning setting.

Two GNN models for evaluation:
 - **OGB Baseline (GCN model)**
   - Original repo: [link](https://github.com/snap-stanford/ogb/tree/master/examples/graphproppred/mol)

 - **Graphormer**
   - Local repo: [graphormer](/graphormer/)
   - Original repo:  [link](https://github.com/ytchx1999/Graphormer)
   - Winner of quantum prediction track of OGB Large-Scale Challenge (KDD CUP 2021).

There are various knobs and configurations that can be tuned for a FL training. This can be done by changing default config of `fedml_hiv.yaml`. In this evaluation, these are some of the handy configs that we tune to benchmark both models.

```yaml
common_args:
  training_type: "simulation"   # Simulation mode
data_args:
  dataset: "ogbg-molhiv"        # MolHIV data
model_args:
  model: "graphormer"           # Choose between: ogb or graphormer
train_args:
  client_num_in_total: 1        # 1 client 
  client_num_per_round: 1       # run 1 client per round ( < client_num_in_total)
  comm_round: 50                # 50 rounds of communication
  epochs: 2                     # 2 epoch per comm round
  metric: "aucroc"              # Classification Task
tracking_args:
  enable_wandb: false           # We track the training progress with Wands
  wandb_key: b2607              # Self wandb key
```
