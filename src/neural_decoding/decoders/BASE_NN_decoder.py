from abc import ABC, abstractmethod
from neural_decoding.decoders.BASE_decoder import decoder
import torch
import numpy as np
import TrainingUtils

class NeuralNetwork(decoder):
    @abstractmethod
    def __init__(self, input_size, output_size, model_params) -> None:
        pass
    
    @abstractmethod
    def train_model(self, data_loader, loss_func, optimizer, training_params):
        """
        Simple training method for any neural network model using standard gradient descent. No extra features other than simple ones.
        Works with FNNs and RNNs. MODELS ARE UPDATED IN PLACE.

        Args:
            model:                          pytorch model
            data_loader:                    dict: contains training ("loader_train") and validation ("loader_val") loaders
            loss_func:                      loss func (nn.mseloss)
            optimizer:                      pytorch optimizer (Adam)
            training_params:                dict that Contains the below:
                                                device(str, optional):                    What device to train on (cpu/gpu). Defaults to cpu. 
                                                print_results (bool, optional)            Print updates. Defaults to True
                                                print_every (int, optional):              How often to print updates. Defaults to 10
                                                epochs (int, optional):                   Will stop after this amount of epochs. Defaults to 100

        Returns:
            [iter, valloss, corr]:   trained model,  validation loss, (train loss history, val loss history)
        """

        model = self
        loader_train = data_loader["loader_train"]
        loader_val = data_loader["loader_val"]
        print_results = training_params.get("print_results", True)
        print_every=training_params.get("print_every", 10)
        epochs=training_params.get("epochs", 100)
        device=training_params.get("device", "cpu")
        outer_iter, valloss = 0, 0
        loss_history_train, loss_history_val, corr_history = [], [], []

        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            inner_iter = 0

            for batch in loader_train:
                x = batch['chans']          # [batch_size x 96 x conv_size]
                y = batch['states']         # [batch_size x num_fings]
                x = x.to(device=device)
                y = y.to(device=device)
                model.train()

                # zero gradients + forward + backward + optimize
                optimizer.zero_grad()
                yhat = model.forward(x)   # normal forward pass

                if isinstance(yhat, tuple):
                    # RNNs return y, h. 
                    yhat = yhat[0]

                loss = loss_func(yhat, y)
                loss.backward()
                optimizer.step()

                # keep track of iteration and loss
                inner_iter += 1
                outer_iter += 1
                running_loss += loss.item()

            # occasionally check validation accuracy and plot
            if print_results and ((epoch % print_every == 0) or (epoch == epochs - 1)):
                    # get batch data 
                    running_val_loss = 0.0
                    all_predictions = []  # List to store all predictions (will be used to calculate corr)
                    all_targets = []      # List to store all targets (will be used to calculate corr)
                    for val_batch in loader_val:
                        with torch.no_grad():
                            x2 = val_batch['chans'].to(device=device)        # shape (num_samps, 96, seq_len)
                            y2 = val_batch['states'].to(device=device)       # shape (num_samps, num_outputs)
                            model.eval()

                            yhat2 = model.forward(x2)

                            if isinstance(yhat2, tuple):
                                # RNNs return y, h
                                yhat2 = yhat2[0]

                            all_predictions.append(yhat2.cpu().numpy())  # Convert to NumPy and append
                            all_targets.append(y2.cpu().numpy())         # Convert to NumPy and append
                            val_loss = loss_func(yhat2, y2).item()
                            running_val_loss += val_loss

                    # Concatenate all predictions and targets
                    all_predictions = np.concatenate(all_predictions, axis=0)
                    all_targets = np.concatenate(all_targets, axis=0)

                    # Calculate correlation
                    correlation = TrainingUtils.calc_corr(all_predictions, all_targets)

                    valloss = running_val_loss / len(loader_val)
                    print('Epoch [{}/{}], iter {} Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, epochs - 1, outer_iter, loss, valloss))
                    loss_history_val.append(valloss)
                    loss_history_train.append(running_loss / inner_iter)
                    corr_history.append(correlation)
                
                
        if print_results: print('*** final epoch is done ***') 
        return valloss, (loss_history_train, loss_history_val, corr_history)


    @abstractmethod
    def forward(self, input):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        pass




    