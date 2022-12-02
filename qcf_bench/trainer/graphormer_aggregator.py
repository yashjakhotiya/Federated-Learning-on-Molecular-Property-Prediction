import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error

from fedml.core import ServerAggregator

from ogb.graphproppred import Evaluator

class GraphormerAggregator(ServerAggregator):
    def __init__(self, *args):
        super().__init__()
        self.best_score = 0

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def test(self, test_data, device, args):
        logging.info("--------test--------")
        model = self.model
        model.eval()
        model.to(device)
        
        y_true = []
        y_pred = []

        evaluator = Evaluator("ogbg-molhiv")

        for step, batch in enumerate(test_data):
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            
        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)["rocauc"], model

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args) -> bool:
        logging.info("--------test_on_the_server--------")

        model_list, score_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, model = self.test(test_data, device, args)
            for idx in range(len(model_list)):
                self._compare_models(model, model_list[idx])
            model_list.append(model)
            score_list.append(score)
            logging.info("Client {}, Test {} = {}".format(client_idx, args.metric.upper(), score))
            if args.enable_wandb:
                wandb.log({"Client {} Test/{}".format(client_idx, args.metric.upper()): score})
        avg_score = np.mean(np.array(score_list))
        logging.info("Test {} score = {}".format(args.metric.upper(), avg_score))

        if self.best_score < avg_score:
            self.best_score = avg_score

        if args.enable_wandb:
            wandb.log({args.metric.upper(): avg_score, "best": self.best_score})

        return True

    def _compare_models(self, model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if key_item_1[0] == key_item_2[0]:
                    logging.info("Mismatch found at", key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            logging.info("Models match perfectly! :)")
