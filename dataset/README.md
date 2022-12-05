# Heterogeneous Datset Generation via ECFP4 Fingerprints and Latent Dirichlet Allocation

## How to use

The `splits` folder contains splits of some datasets that have been precomputed. To generate your own splits, you should use `pace_ice_split.py` which allows command-line arguments to specify parameters of the split. A Jupyter Notebook `Heterogeneous_dataset.ipynb` is provided to go into more detail about the methodology of the splitting procedure. 

Example downloading and splitting the PCQM4Mv2 dataset
```bash
python3 pace_ice_split.py --name PCQM4Mv2 --train_split 0.05 --method "scaffold"
```

## Background

Federated Learning for Chemistry uses Scaffold Splitting to sort molecules into clusters based on their carbon scaffolds. However, this method is limited to molecules with an extensive carbon backbone, and results in too many clusters when considering a database of millions of molecules.

Instead, the standard cheminformatics procedure would consist of vectorising molecules into fingerprints (ECFP4 fingerprints chosen here), as similar molecules would have similar fingerprints based on functional groups in addition to carbon backbones. This method also generalises to all molecules. 

With these fingerprints, molecules are assigned using a Latent Dirichlet Allocation method where the heterogeneity is scaled based on the alpha value (also known as document prior) with each cluster corresponding to a client. To ensure a minimum size for each client, molecules are greedily allocated starting from clusters with the smallest size to ensure that there are enough samples per client. 

