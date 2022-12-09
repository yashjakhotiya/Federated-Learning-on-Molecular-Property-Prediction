# Note: this is a modification of the original code:
# https://github.com/FedML-AI/FedML/blob/master/python/app/fedgraphnn/moleculenet_graph_reg/fedml_moleculenet_property_prediction.py

import logging

import fedml
from data.data_loader import load_partition_data
from fedml import FedMLRunner
from model.ogb_baseline import GNN
from model.graphormer import Graphormer
from trainer.ogb_aggregator import OgbAggregator
from trainer.ogb_trainer import OgbTrainer
from trainer.graphormer_aggregator import GraphormerAggregator
from trainer.graphormer_trainer import GraphormerTrainer


def load_data(args, dataset_name):
    num_cats, feat_dim = 0, 0
    if dataset_name not in ["ogbg-molhiv"]:
        raise Exception("no such dataset!")

    logging.info("load_data. dataset_name = %s" % dataset_name)
    unif = True if args.partition_method == "homo" else False

    (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
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
        train_data_num + test_data_num
    ]

    return dataset


def create_model(args, model_name, num_tasks):
    print("modelname!! \n", model_name)
    logging.info("create_model. model_name = %s" % model_name)
    if model_name == "ogb":
        model = GNN(
            num_tasks = args.num_tasks,
            num_layer = args.ogb_num_layer,
            emb_dim = args.ogb_emb_dim, 
            gnn_type = args.ogb_gnn_type,
            virtual_node = args.virtual_node,
            residual = args.residual,
            drop_ratio = args.drop_ratio,
            JK = args.JK,
            graph_pooling = args.graph_pooling
        )
        trainer = OgbTrainer(model, args)
        aggregator = OgbAggregator(model, args)
    elif model_name == "graphormer":
        model = Graphormer(
            num_tasks = args.num_tasks,
            n_layers = args.graphormer_n_layers,
            num_heads = args.graphormer_num_heads,
            hidden_dim = args.graphormer_hidden_dim,
            dropout_rate = args.graphormer_dropout_rate,
            intput_dropout_rate = args.graphormer_intput_dropout_rate,
            weight_decay = args.graphormer_weight_decay,
            ffn_dim = args.graphormer_ffn_dim,
            warmup_updates = args.graphormer_warmup_updates,
            tot_updates = args.graphormer_tot_updates,
            peak_lr = args.graphormer_peak_lr,
            end_lr = args.graphormer_end_lr,
            edge_type = args.graphormer_edge_type,
            multi_hop_max_dist = args.graphormer_multi_hop_max_dist,
            attention_dropout_rate = args.graphormer_attention_dropout_rate
        )
        trainer = GraphormerTrainer(model, args)
        aggregator = GraphormerAggregator(model, args)
    else:
        raise Exception("such model does not exist !")

    return model, trainer, aggregator


if __name__ == "__main__":
    # init FedML framework
    args = fedml.init()

    # init device
    device = fedml.device.get_device(args)

    # load data
    dataset = load_data(args, args.dataset)

    # load model
    model, trainer, aggregator = create_model(args, args.model, 1)

    # start training
    fedml_runner = FedMLRunner(args, device, dataset, model, trainer, aggregator)
    fedml_runner.run()
