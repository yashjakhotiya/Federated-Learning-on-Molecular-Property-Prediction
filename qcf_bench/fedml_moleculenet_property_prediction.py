import logging

import fedml
from data.data_loader import load_partition_data, get_data
from fedml import FedMLRunner
from model.gcn_readout import GcnMoleculeNet
from model.ogb_baseline import GNN
from trainer.gcn_aggregator_readout_regression import GcnMoleculeNetAggregator
from trainer.gcn_trainer_readout_regression import GcnMoleculeNetTrainer


def load_data(args, dataset_name):
    num_cats, feat_dim = 0, 0
    if dataset_name not in ["pcqm4mv2"]:
        raise Exception("no such dataset!")

    logging.info("load_data. dataset_name = %s" % dataset_name)
    unif = True if args.partition_method == "homo" else False

    args.metric = "mae"

    (
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
    ) = load_partition_data(
        args,
        args.data_cache_dir + args.dataset,
        args.client_num_in_total,
        uniform=unif
    )

    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        labels[0].shape[0],
    ]

    return dataset, labels[0].shape[0]


def create_model(args, model_name, num_cats, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    if model_name == "ogb":
        model = GNN(
            num_tasks = 1,
            num_layers = 5,
            emb_dim = 300, 
            gnn_type = 'gin',
            virtual_node = True,
            residual = False,
            drop_ratio = 0,
            JK = "last",
            graph_pooling = "sum"
        )
        trainer = OgbTrainer(model, args)
        aggregator = OgbAggregator(model, args)
    else:
        raise Exception("such model does not exist !")

    return model, trainer, aggregator


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)
    print(args)

    # load data
    dataset, num_cats = load_data(args, args.dataset)

    # load model
    model, trainer, aggregator = create_model(args, args.model, feat_dim, num_cats, output_dim=None)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()