import logging
import pickle
import random

from torch_geometric.loader import DataLoader

from fedml.core import partition_class_samples_with_dirichlet_distribution

from ogb.lsc import PygPCQM4Mv2Dataset

# For centralized training
def get_dataloader(path, compact=True):
    dataset = PygPCQM4Mv2Dataset(root = '/home/jovyan/ogb_dataset')

    split_idx = dataset.get_idx_split()

    train_dataloader = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    val_dataloader = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    test_dataloader = DataLoader(
        dataset[split_idx["test-dev"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )

    return train_dataloader, val_dataloader, test_dataloader


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
    
    dataset = PygPCQM4Mv2Dataset(root = '/home/jovyan/ogb_dataset')
    
    logging.info("Splitting OGB Dataset...")
    
    split_idx = dataset.get_idx_split()

    train_data_global = DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    val_data_global = DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False, pin_memory=True
    )
    
    test_data_global = DataLoader(
        dataset[split_idx["test-dev"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )

    train_data_num = len(dataset[split_idx["train"]])
    val_data_num = len(dataset[split_idx["valid"]])
    test_data_num = len(dataset[split_idx["test-dev"]])
    
    # spc = samples per client
    train_spc = train_data_num // client_number
    val_spc = val_data_num // client_number
    test_spc = test_data_num // client_number
    
    logging.info("Separating dataset to clients...")

    for client in range(client_number):
        
        train_client = dataset[split_idx["train"]][client * train_spc : (client + 1) * train_spc]
        val_client = dataset[split_idx["valid"]][client * val_spc : (client + 1) * val_spc]
        test_client = dataset[split_idx["test-dev"]][client * test_spc : (client + 1) * test_spc]

        data_local_num_dict[client] = len(train_client)
        
        train_data_local_dict[client] = DataLoader(
            train_client,
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True
        )
        
        val_data_local_dict[client] = DataLoader(
            val_client,
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )
        
        test_data_local_dict[client] = DataLoader(
            val_client, # no labels for test
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True
        )

        logging.info(
            "Client idx = {}, local train sample number = {}".format(
                client, len(train_client)
            )
        )

    return (
        train_data_num,
        val_data_num,
        val_data_num, # no labels for test
        train_data_global,
        val_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        val_data_local_dict,
        test_data_local_dict,
    )
