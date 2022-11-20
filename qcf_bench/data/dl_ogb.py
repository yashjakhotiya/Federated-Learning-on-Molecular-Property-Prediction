import logging
import pickle
import random

import torch.utils.data as data

from fedml.core import partition_class_samples_with_dirichlet_distribution

from ogb.lsc import PygPCQM4Mv2Dataset

# For centralized training
def get_dataloader(path, compact=True):
    dataset = PygPCQM4Mv2Dataset(root = 'dataset/')

    split_idx = dataset.get_idx_split()

    train_dataloader = data.DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    val_dataloader = data.DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    test_dataloader = data.DataLoader(
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
    global_data_dict, partition_dicts = partition_data_by_sample_size(
        args, path, client_number, uniform, compact=compact
    )

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    val_data_local_dict = dict()
    test_data_local_dict = dict()
    
    dataset = PygPCQM4Mv2Dataset(root = 'dataset/')
    
    split_idx = dataset.get_idx_split()
    
    valid_loader = DataLoader(num_workers = args.num_workers)

    train_data_global = data.DataLoader(
        dataset[split_idx["train"]],
        batch_size=args.batch_size,
        shuffle=True, pin_memory=True
    )
    
    val_data_global = data.DataLoader(
        dataset[split_idx["valid"]],
        batch_size=args.batch_size,
        shuffle=False, pin_memory=True
    )
    
    test_data_global = data.DataLoader(
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

    for client in range(client_number):
        
        train_client = dataset[split_idx["train"]][client * train_spc : (client + 1) * train_spc]
        val_client = dataset[split_idx["valid"]][client * val_spc : (client + 1) * val_spc]
        test_client = dataset[split_idx["test-dev"]][client * test_spc : (client + 1) * test_spc]

        data_local_num_dict[client] = len(train_client)
        
        train_data_local_dict[client] = data.DataLoader(
            train_client,
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True
        )
        
        val_data_local_dict[client] = data.DataLoader(
            val_client,
            batch_size=args.batch_size,
        shuffle=False, pin_memory=True
        )
        
        test_data_local_dict[client] = data.DataLoader(
                test_client,
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True
        )

        logging.info(
            "Client idx = {}, local train sample number = {}".format(
                client, len(train_dataset_client)
            )
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
