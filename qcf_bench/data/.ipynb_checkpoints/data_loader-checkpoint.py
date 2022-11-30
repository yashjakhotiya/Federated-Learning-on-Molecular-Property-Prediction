import json
import logging
import pickle
import random
from functools import partial

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PygDataLoader

from ogb.graphproppred import PygGraphPropPredDataset
from data.wrapper import MyGraphPropPredDataset
from data.collator import collator

# Single process sequential
def load_partition_data(
    args,
    path,
    client_number,
    uniform=True
):

    data_local_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    
    logging.info("Loading OGB Dataset...")

    if args.model == "graphormer":
        dataset = MyGraphPropPredDataset(name="ogbg-molhiv")
        DataLoader = TorchDataLoader
        collate_fn = partial(collator)
    else:
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv")
        DataLoader = PygDataLoader
        collate_fn = None
    
    logging.info(f"Loading previously generated splits with client num {client_number}...")

    ALPHA_VAL = args.partition_alpha
    with open(f"data/noniid/fps_HIV_clients_{client_number:02d}_alpha_{ALPHA_VAL}.json") as f:
        client_mapping = json.load(f)
    
    logging.info(f"Partitioning into train and val... with size {len(client_mapping)}")

    train_global = []
    test_global = []

    train_data_num, test_data_num = 0, 0
    TRAIN_SPLIT = 0.9
    for c_num in range(client_number):
        # NOTE: the index in the split is in str, will convert it to num
        c_idx = str(c_num)
        num_train = int(len(client_mapping[c_idx]) * TRAIN_SPLIT)
        train_client = client_mapping[c_idx][:num_train]
        test_client = client_mapping[c_idx][num_train:]

        data_local_num_dict[c_num] = len(train_client)
        
        train_data_local_dict[c_num] = DataLoader(
            dataset[train_client],
            batch_size=args.batch_size,
            shuffle=True, pin_memory=True,
            collate_fn=collate_fn
        )
        
        test_data_local_dict[c_num] = DataLoader(
            dataset[test_client],
            batch_size=args.batch_size,
            shuffle=False, pin_memory=True,
            collate_fn=collate_fn
        )

        logging.info(
            "Client idx = {}, local train sample number = {}".format(
                c_num, len(train_client)
            )
        )

        train_data_num += len(train_client)
        test_data_num += len(test_client)

        train_global.extend(train_client)
        test_global.extend(test_client)

    train_data_global = DataLoader(
        dataset[train_global],
        batch_size=args.batch_size,
        shuffle=False, pin_memory=True,
        collate_fn=collate_fn
    )

    test_data_global = DataLoader(
        dataset[test_global],
        batch_size=args.batch_size,
        shuffle=False, pin_memory=True,
        collate_fn=collate_fn
    )

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
    )
