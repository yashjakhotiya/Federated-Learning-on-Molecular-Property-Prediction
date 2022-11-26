import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error

from fedml.core.alg_frame.client_trainer import ClientTrainer

from ogb.lsc import PCQM4Mv2Evaluator

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

        criterion = torch.nn.L1Loss()
        min_score = np.Inf
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        best_model_params = {}
        
        for epoch in range(args.epochs):
            avg_loss = 0
            count = 0
            for step, batch in enumerate(train_data):
                if step % 10 == 0:
                    logging.info(f"Epoch {epoch}; Step {step}")
                batch = batch.to(device)
                pred = model(batch).view(-1,)
                optimizer.zero_grad()
                loss = criterion(pred, batch.y)
                loss.backward()
                optimizer.step()

                if test_data is not None:
                    logging.info("Testing..")
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
                    
        logging.info("Done training!")

        return min_score, best_model_params

    def test(self, test_data, device, args):
        logging.info("--------test--------")
        model = self.model
        model.eval()
        model.to(device)
        evaluator = PCQM4Mv2Evaluator()

        with torch.no_grad():
            y_pred = []
            y_true = []
            for step, batch in enumerate(test_data):
                batch = batch.to(device)

                with torch.no_grad():
                    pred = model(batch).view(-1,)

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            
        y_true = torch.cat(y_true, dim = 0)
        y_pred = torch.cat(y_pred, dim = 0)

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        return evaluator.eval(input_dict)["mae"], model
