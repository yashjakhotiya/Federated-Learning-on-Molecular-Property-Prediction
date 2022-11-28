import json
import logging
import pickle
import random

from torch_geometric.loader import DataLoader

from fedml.core import partition_class_samples_with_dirichlet_distribution

from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.graphproppred import PygGraphPropPredDataset


# NOTE: this is not being in used
# # For centralized training
# def get_dataloader(path, compact=True):
#     # dataset = PygPCQM4Mv2Dataset(root = '/home/jovyan/ogb_dataset')
#     dataset = PygGraphPropPredDataset(name="ogbg-molhiv")

#     split_idx = dataset.get_idx_split()

#     train_dataloader = DataLoader(
#         dataset[split_idx["train"]],
#         batch_size=args.batch_size,
#         shuffle=True, pin_memory=True
#     )
    
#     val_dataloader = DataLoader(
#         dataset[split_idx["valid"]],
#         batch_size=args.batch_size,
#         shuffle=True, pin_memory=True
#     )
    
#     test_dataloader = DataLoader(
#         dataset[split_idx["test-dev"]],
#         batch_size=args.batch_size,
#         shuffle=True, pin_memory=True
#     )

#     return train_dataloader, val_dataloader, test_dataloader


# Single process sequential
def load_partition_data(
    args,
    path,
    client_number,
    uniform=True
):

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()
    
    logging.info("Loading OGB Dataset...")
    
    # dataset = PygPCQM4Mv2Dataset(root = '/home/jovyan/ogb_dataset')
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
    
    logging.info(f"Loading previously generated splits with client num {client_number}...")

    ALPHA_VAL = args.partition_alpha
    with open(f"data/noniid/fps_HIV_clients_{client_number:02d}_alpha_{ALPHA_VAL}.json") as f:
        client_mapping = json.load(f)
    
    logging.info(f"Partitioning into train and val... with size {len(client_mapping)}")

    train_global = []
    val_global = []
    test_global = []

    train_data_num, val_data_num, test_data_num = 0, 0, 0
    TRAIN_SPLIT = 0.9
    VAL_SPLIT = 0.05
    for c_num in range(client_number):
        # NOTE: the index in the split is in str, will convert it to num
        c_idx = str(c_num)
        num_train = int(len(client_mapping[c_idx]) * TRAIN_SPLIT)
        num_val = int(len(client_mapping[c_idx]) * VAL_SPLIT)
        train_client = client_mapping[c_idx][:num_train]
        val_client = client_mapping[c_idx][num_train:num_train+num_val]
        test_client = client_mapping[c_idx][num_train+num_val:]

        data_local_num_dict[c_num] = len(train_client)
        
        train_data_local_dict[c_num] = DataLoader(
            dataset[train_client],
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True
        )
        
        val_data_local_dict[c_num] = DataLoader(
            dataset[val_client],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )
        
        test_data_local_dict[c_num] = DataLoader(
            dataset[test_client],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )

        logging.info(
            "Client idx = {}, local train sample number = {}".format(
                c_num, len(train_client)
            )
        )

        train_data_num += len(train_client)
        val_data_num += len(val_client)
        test_data_num += len(test_client)

        train_global.extend(train_client)
        val_global.extend(val_client)
        test_global.extend(test_client)

    train_data_global = DataLoader(
            dataset[train_global],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )
    val_data_global = DataLoader(
            dataset[val_global],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )
    test_data_global = DataLoader(
            dataset[test_global],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )

    return (
        train_data_num,
        val_data_num,
        test_data_num,
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    )
