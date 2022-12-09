import logging

import numpy as np
import torch
import wandb
from sklearn.metrics import mean_absolute_error

from fedml.core.alg_frame.client_trainer import ClientTrainer

from ogb.graphproppred import Evaluator

class OgbTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    """
    This method is essentially just the standard 'train' script with a few added bonuses (logging, basic skeleton).
    
    This method doesn't need to return anything, but it can also return the best_model_params and min_score.
    At least, that's the way it looks like on FedML.
    """
    def train(self, train_data, device, args):
        model = self.model
        model.train()
        model.to(device)

        # Change this for a a different dataset
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        
        # Feel free to change your optimizer, or add an argument to allow changing it easily.
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        
        test_data = None
        
        best_model_params = {}
        min_score = -np.Inf
        
        try:
            test_data = self.test_data
        except:
            pass

        # Here is what is essentially your 'training script'
        # Change this to match your model's and if any dataset changes propagate here, change this as well
        for epoch in range(args.epochs):
            for step, batch in enumerate(train_data):
                if step % 10 == 0:
                    logging.info(f"Epoch {epoch}; Step {step}")
                
                batch = batch.to(device)

                if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
                    pass
                else:
                    pred = model(batch)
                    optimizer.zero_grad()
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    loss = criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss.backward()
                    optimizer.step()
                    
            # This simply logs the best train score per epoch (not to wandb)
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

    """
    Tests the model on the test data.
    
    Should return the score and the model, in that order.
    """
    def test(self, test_data, device, args):
        logging.info("--------test--------")
        model = self.model
        model.eval()
        model.to(device)
        
        y_true = []
        y_pred = []

        # Change the evaluator for a different dataset
        evaluator = Evaluator("ogbg-molhiv")

        for step, batch in enumerate(test_data):
            batch = batch.to(device)

            if batch.x.shape[0] == 1:
                pass
            else:
                with torch.no_grad():
                    pred = model(batch)

                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            
        y_true = torch.cat(y_true, dim = 0).numpy()
        y_pred = torch.cat(y_pred, dim = 0).numpy()

        input_dict = {"y_true": y_true, "y_pred": y_pred}

        # Change score creation for different dataset
        return evaluator.eval(input_dict)["rocauc"], model
