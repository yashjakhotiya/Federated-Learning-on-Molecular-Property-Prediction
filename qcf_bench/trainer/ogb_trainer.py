import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error

from fedml.core.alg_frame.client_trainer import ClientTrainer

class OgbTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        test_data = None
        try:
            test_data = self.test_data
        except:
            pass

        criterion = torch.nn.MAELoss()
        min_score = np.Inf
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        best_model_params = {}
        
        for epoch in range(args.epochs):
            avg_loss = 0
            count = 0
            for mol_idxs, (adj_matrix, feature_matrix, label, _) in enumerate(
                train_data
            ):
                optimizer.zero_grad()

                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)

                logits = model(adj_matrix, feature_matrix)
                loss = criterion(logits, label)
                loss.backward()
                optimizer.step()

                if test_data is not None:
                    test_score, _ = self.test(self.test_data, device, args)
                    print(
                        "Epoch = {}: Test {} = {}".format(
                            epoch, args.metric.upper(), test_score
                        )
                    )
                    if test_score < min_score:
                        min_score = test_score
                        best_model_params = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                    print(
                        "Current best {}= {}".format(args.metric.upper(), min_score)
                    )

        return min_score, best_model_params

    def test(self, test_data, device, args):
        logging.info("--------test--------")
        model = self.model
        model.eval()
        model.to(device)

        with torch.no_grad():
            y_pred = []
            y_true = []
            for mol_idx, (adj_matrix, feature_matrix, label, _) in enumerate(test_data):
                adj_matrix = adj_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                feature_matrix = feature_matrix.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                label = label.to(device=device, dtype=torch.float32, non_blocking=True)
                logits = model(adj_matrix, feature_matrix)
                y_pred.append(logits.cpu().numpy())
                y_true.append(label.cpu().numpy())

            score = mean_absolute_error(np.array(y_true), np.array(y_pred))
        return score, model
