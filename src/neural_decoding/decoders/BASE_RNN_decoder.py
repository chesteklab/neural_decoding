from abc import ABC, abstractmethod
from neural_decoding.decoders.BASE_NN_decoder import NeuralNetwork
import torch
import torch.nn as nn
import utils

class RecurrentNeuralNetwork(NeuralNetwork):
    @abstractmethod
    def __init__(self, input_size, output_size, model_params) -> None:
        ''' 
        Initializes a general recurrent model that can use VanillaRNN/GRU/LSTM, with a linear layer to the output 

         Args:
            input_size:                  number of input features
            num_outputs:                 number of output features
            model_params:                dict containing the following model params:
                                            hidden_size:        size of hidden state in model 
                                            num_layers:         number of layers in model
                                            rnn_type:           specifies what type of recurrent model (gru, lstm, rnn)
                                            device:             optional, specifies what device to compute on. Default is cpu.
                                            hidden_noise_std:   optional, deviation of hidden noise to add to model
                                            dropout_input:      optional, drops out some input during forward pass
                                            drop_prob:          optional, specifies probability of layer of model being dropped
        Returns:
            None
        ''' 

        self.input_size = input_size
        self.num_outputs = output_size
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]
        self.device = model_params.get("device", "cpu") # default device is cpu
        self.hidden_noise_std = model_params.get("hidden_noise_std", None)

        dropout_input = model_params.get("dropout_input", None)
        self.dropout_input = nn.Dropout(dropout_input) if dropout_input else None

        self.rnn_type = model_params["rnn_type"].lower()
        drop_prob = model_params.get("drop_prob", 0)
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True, nonlinearity='relu')
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=drop_prob, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_outputs)
    
    @abstractmethod
    def train_model(self, data_loader, loss_func, optimizer, training_params):
        """
        Trains Recurrent Model. (See BASE_NN for details)
        """

        val_loss, (loss_history_train, loss_history_val) = super().train_model(data_loader, loss_func, optimizer, training_params)

        return val_loss, (loss_history_train, loss_history_val)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def save_model(self, filepath):
        pass
    
    @abstractmethod
    def load_model(self, filepath):
        pass




    