import json
import numpy as np
import torch

import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer
from dgllife.data import BACE, BBBP, ClinTox, ESOL, Lipophilicity, FreeSolv, SIDER, Tox21, HIV
from ogb.lsc import PCQM4Mv2Dataset
    
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.decomposition import LatentDirichletAllocation


def getDataset(datasetName, load_prev=True):
    if datasetName == "PCQM4Mv2":
        dataset = PCQM4Mv2Dataset(only_smiles = True)

    elif datasetName in ['BACE', 'BBBP', 'ClinTox', 'Esol', 'Freesolv',
                         'Lipophilicity', 'SIDER', 'Tox21', 'qm9', 'HIV']:

        node_featurizer = CanonicalAtomFeaturizer()

        if datasetName == 'BACE':  
            dataset = BACE(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'BBBP':  
            dataset = BBBP(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'ClinTox':  
            dataset = ClinTox(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'Esol':
            dataset = ESOL(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'Freesolv':
            dataset = FreeSolv(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'Lipophilicity':
            dataset = Lipophilicity(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'SIDER':  
            dataset = SIDER(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'Tox21':  
            dataset = Tox21(smiles_to_bigraph, node_featurizer, load=load_prev)
        elif datasetName == 'HIV':  
            dataset = HIV(smiles_to_bigraph, node_featurizer, load=load_prev)
    else:
        raise ValueError(f'Unexpected dataset: {datasetName}')

    return dataset

def get_fingerprints(smiles_list, nBits=1024):
    mol_list = [Chem.MolFromSmiles(x) for x in smiles_list]
    return [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits) for x in mol_list]

def split_data_LDA(vectors, num_grps, alpha, min_size=128):
    """
    Clusters vectors using Latent Dirichlet Allocation
    args:
        vectors:      List of fingerprint representation of molecules
        num_grps:     Number of clusters
        alpha:        (0,1]. Controls heterogeneity of dataset, lower values being more distinct
    return:
        cluster_idx:  N numpy array. Index of cluster for each row in vector
    """
    
    lda = LatentDirichletAllocation(n_components=num_grps, doc_topic_prior=alpha,
                                    learning_method="online", random_state=0)
    lda.fit(vectors)

    # Gives prob of each fingerprint in FPS belonging to the organisation
    group_logits = lda.transform(vectors)

    cluster_idx = np.argmax(group_logits, axis=1)

    net_dataidx_map = {}
    idxs = np.arange(len(vectors))
    sizes = []
    for i in range(num_grps):
        net_dataidx_map[i] = idxs[cluster_idx == i]
        np.random.shuffle(net_dataidx_map[i])
        net_dataidx_map[i] = net_dataidx_map[i].tolist()
        sizes.append((len(net_dataidx_map[i]), i))

    sizes.sort()

    if sizes[0][1] >= min_size:
        return net_dataidx_map

    net_dataidx_map = {}
    used_idxs = set()
    for curr_s in range(len(sizes)):
        idx_to_query = sizes[curr_s][1]
        if curr_s == len(sizes)-1:
            net_dataidx_map[idx_to_query] = list(set(range(len(vectors))).difference(used_idxs))
        else:
            indexes = idxs[cluster_idx == idx_to_query]
            remaining_indexes = list(set(indexes.tolist()).difference(used_idxs))
            if len(remaining_indexes) >= min_size:
                net_dataidx_map[idx_to_query] = remaining_indexes
            else:
                logit = group_logits[:, idx_to_query]
                all_idx_args = np.argsort(logit).tolist()
                net_dataidx_map[idx_to_query] = []
                temp = 0
                while len(net_dataidx_map[idx_to_query]) < min_size:
                    idx = all_idx_args[temp]
                    if idx not in used_idxs:
                        net_dataidx_map[idx_to_query].append(idx)
                    temp += 1

        used_idxs = used_idxs.union(set(net_dataidx_map[idx_to_query]))

    return net_dataidx_map
    
def scaffold_clustering(smiles_list):
    d = {}
    cluster_id = 0
    ans = []
    for idx, smiles in enumerate(smiles_list):
        scaffold = Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmilesFromSmiles(smiles)
        if scaffold not in d:
            d[scaffold] = cluster_id
            cluster_id += 1
        ans.append(d[scaffold])
    return np.asarray(ans)
    
def non_iid_partition_with_dirichlet_distribution(
    label_list, client_num, alpha, min_size, max_tries=100
):
    """
    Obtain sample index list for each client from the Dirichlet distribution.
    This LDA method is first proposed by :
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
    This can generate nonIIDness with unbalance sample number in each label.
    The Dirichlet distribution is a density over a K dimensional vector p whose K components are positive and sum to 1.
    Dirichlet can support the probabilities of a K-way categorical event.
    In FL, we can view K clients' sample number obeys the Dirichlet distribution.
    For more details of the Dirichlet distribution, please check https://en.wikipedia.org/wiki/Dirichlet_distribution

    Parameters
    ----------
        label_list : the label list from classification dataset
        client_num : number of clients
        alpha: a concentration parameter controlling the identicalness among clients.
        min_size: minimum dataset size for clients
        max_tries: maximum number of tries to generate datasets s.t. all datatsets have at least the min size
    Returns
    -------
        samples : ndarray,
            The drawn samples, of shape ``(size, k)``.
    """
    np.random.seed(0)
    net_dataidx_map = {}
    K = len(np.unique(label_list))
    N = label_list.shape[0]    

    # guarantee the minimum number of sample in each client
    current_min_size = 0
    num_tries = 0
    while current_min_size < min_size and num_tries < max_tries:
        idx_batch = [[] for _ in range(client_num)]

        # for each classification in the dataset
        for k in range(K):
            # get a list of batch indexes which are belong to label k
            idx_k = np.where(label_list == k)[0]
            idx_batch, current_min_size = partition_class_samples_with_dirichlet_distribution(
                    N, alpha, client_num, idx_batch, idx_k)
        num_tries += 1

    for i in range(client_num):
        np.random.shuffle(idx_batch[i])
        net_dataidx_map[i] = idx_batch[i]

    return net_dataidx_map


def partition_class_samples_with_dirichlet_distribution(
    N, alpha, client_num, idx_batch, idx_k
):
    np.random.shuffle(idx_k)
    # using dirichlet distribution to determine the unbalanced proportion for each client (client_num in total)
    # e.g., when client_num = 4, proportions = [0.29543505 0.38414498 0.31998781 0.00043216], sum(proportions) = 1
    proportions = np.random.dirichlet(np.repeat(alpha, client_num))

    # get the index in idx_k according to the dirichlet distribution
    proportions = np.array(
            [p * (len(idx_j) < N / client_num) for p, idx_j in zip(proportions, idx_batch)]
    )
    proportions = proportions / proportions.sum()
    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

    # generate the batch list for each client
    idx_batch = [ 
            idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
    ]
    min_size = min([len(idx_j) for idx_j in idx_batch])

    return idx_batch, min_size
    
def gen_hetero_split_fps(dataset_name, num_clients, alpha, min_size=128):
    all_smiles = [x[0] for x in getDataset(dataset_name)]
    fps_list = get_fingerprints(all_smiles)

    print("Fingerprints done")

    client_mapping = split_data_LDA(fps_list, num_grps=num_clients, alpha=alpha, min_size=min_size)

    with open(f"fps_{dataset_name}_clients_{num_clients}_alpha_{alpha}.json", "w") as f:
        json.dump(client_mapping, f, indent=4)

    return client_mapping
    
def gen_hetero_split_scaffold(dataset_name, num_clients, alpha, min_size=128):
    all_smiles = [x[0] for x in getDataset(dataset_name)]
    cluster_idx = scaffold_clustering(all_smiles)

    print("Scaffolding done")

    client_mapping = non_iid_partition_with_dirichlet_distribution(cluster_idx,
                                                                   client_num=num_clients,
                                                                   alpha=alpha, 
                                                                   min_size=min_size)
    
    with open(f"scaffold_{dataset_name}_clients_{num_clients}_alpha_{alpha}.json", "w") as f:
        json.dump(client_mapping, f, indent=4)

    return client_mapping
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name of dataset to use, from the following datasets:\n"+
                                                 "'BACE', 'BBBP', 'ClinTox', 'Esol', 'Freesolv', 'HIV', 'Lipophilicity', 'SIDER', 'Tox21', 'PCQM4Mv2'")
    parser.add_argument("--clients", type=int, help="Number of clients", default=4)
    parser.add_argument("--alpha", type=float, help="Alpha value of allocation", default=0.1)
    parser.add_argument("--method", type=str, help="Method of split: fps (fingerprint) or scaffold", default="fps")
    parser.add_argument("--min_size", type=int, help="Minimum size of client dataset", default=128)
    
    args = parser.parse_args()

    np.random.seed(0)

    if args.method == "fps":
        gen_hetero_split_fps(dataset_name=args.name,
                             num_clients=args.clients,
                             alpha=args.alpha,
                             min_size=args.min_size)
    elif args.method == "scaffold":
        gen_hetero_split_scaffold(dataset_name=args.name,
                                  num_clients=args.clients,
                                  alpha=args.alpha, 
                                  min_size=args.min_size)
    else:
        raise NotImplementedError
