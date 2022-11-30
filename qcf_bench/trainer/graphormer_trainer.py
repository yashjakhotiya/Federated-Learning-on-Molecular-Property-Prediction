import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error

from fedml.core.alg_frame.client_trainer import ClientTrainer

from ogb.graphproppred import Evaluator

class GraphormerTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model
        model.train()
        model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        test_data = None
        
        best_model_params = {}
        min_score = -np.Inf
        
        try:
            test_data = self.test_data
        except:
            pass

        for epoch in range(args.epochs):
            for step, batch in enumerate(train_data):
                if step % 10 == 0:
                    logging.info(f"Epoch {epoch}; Step {step}")
                
                batch = batch.to(device)

                pred = model(batch)
                optimizer.zero_grad()
                loss = criterion(pred, batch.y.float())
                loss.backward()
                optimizer.step()
                    
            train_score, _ = self.test(train_data, device, args)
            logging.info(
                "Epoch = {}: Train {} = {}".format(
                    epoch, args.metric.upper(), train_score
                )
            )

            if train_score > min_score:
                min_score = train_score
                best_model_params = {
                    k: v.cpu() for k, v in model.state_dict().items()
                }
            logging.info(
                "Current best {}= {}".format(args.metric.upper(), min_score)
            )
                    
        logging.info("Done training!")
        
        return best_model_params, min_score

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
