import torch
import torch.nn as nn
from neural_decoding.decoders.BASE_NN_decoder import NeuralNetwork

class RecurrentModel(nn.Module, NeuralNetwork):
    def __init__(self, input_size, num_outputs, model_params):
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
                                            dropout_input:      optional, specifies probability of layer dropping out
        Returns:
            None
        ''' 

        super().__init__()
        self.input_size = input_size
        self.num_outputs = num_outputs
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



    def forward(self, x, h=None, return_all_tsteps=False):
        """
        Runs forward pass of Recurrent Model

        Args:
            x:                  Neural data tensor of shape (batch_size, num_inputs, sequence_length)
            h:                  Hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
            return_all_steps:   If true, returns predictions from all timesteps in the sequence. If false, only returns the
                                last step in the sequence.
        Returns:
            out:                output/prediction from forward pass of shape (batch_size, seq_len^, num_outs)  ^if return_all_steps is true
            h:                  Hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
        """

        x = x.permute(0, 2, 1)  # put in format (batches, sequence length (history), features)

        if self.dropout_input and self.training:
            x = self.dropout_input(x)

        if h is None:
            h = self.init_hidden(x.shape[0]) # x.shape[0] is batch size

        out, h = self.rnn(x, h) # out shape:    (batch_size, seq_len, hidden_size) like (64, 20, 350)
                                # h shape:      (n_layers, batch_size, hidden_size) like (2, 64, 350)

        if return_all_tsteps:
            out = self.fc(out)  # out now has shape (batch_size, seq_len, num_outs) like (64, 20, 2)
        else:
            out = self.fc(out[:, -1])  # out now has shape (batch_size, num_outs) like (64, 2)

        return out, h
    

    def init_hidden(self, batch_size):
        """
        Initializes hidden state of Recurrent Model

        Args:
            batch_size:   integer describing current batch size

        Returns:
            hidden:       hidden state tensor of shape (n_layers, batch_size, hidden_size) [for LSTM, its a tuple of two of these, one for hidden state, one for cell state]
        """
        if self.rnn_type == 'lstm':
            # lstm - create a tuple of two hidden states
            if self.hidden_noise_std:
                hidden = (torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std).to(device=self.device),
                          torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                       std=self.hidden_noise_std).to(device=self.device))
            else:
                hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device),
                          torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device))
        else:
            # not an lstm - just a single hidden state vector
            if self.hidden_noise_std:
                hidden = torch.normal(mean=torch.zeros(self.num_layers, batch_size, self.hidden_size),
                                      std=self.hidden_noise_std).to(device=self.device)
            else:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=self.device)
        return hidden
    

    def train_model(self, data_loader, criterion, optimizer, training_params):
        """
        Trains model by calling the base NeuralNetwork training method with parameters and model specific to Recurrent Models

        Args:
            model:                          Recurrent Model
            data_loader:                    dict: contains training ("loader_train") and validation ("loader_val") loaders
            loss_func:                      loss func (nn.mseloss)
            optimizer:                      pytorch optimizer (Adam)
            training_params:                dict that Contains the below:
                                                device(str, optional):                    What device to train on (cpu/gpu). Defaults to cpu. 
                                                print_results (bool, optional)            Print updates. Defaults to True
                                                print_every (int, optional):              How often to print updates. Defaults to 10
                                                epochs (int, optional):                   Will stop after this amount of epochs. Defaults to 100

        Returns:
            [val_loss, (loss_history_train, loss_history_val)]:   trained model,  validation loss, and loss history in form of (train loss history, val loss history)
        """

        val_loss, (loss_history_train, loss_history_val) = super().train_model(data_loader, criterion, optimizer, training_params)

        return val_loss, (loss_history_train, loss_history_val)


    def save_model(self, filepath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in

        """
        checkpoint_dict = {
            "model_state_dict": self.state_dict(),
            "model_params": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
            },
            "model_type": self.rnn_type
        }
        torch.save(checkpoint_dict, filepath)


    def load_model(self, filepath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """
        
        checkpoint = torch.load(filepath)

        rnn_types = set("rnn", "lstm", "gru")
        if checkpoint["model_type"] not in rnn_types:
            raise Exception("Tried to load model that isn't a LSTM Instance")
        
        self.load_state_dict(checkpoint["model_state_dict"])

        model_params = checkpoint["model_params"]
        self.hidden_size = model_params["hidden_size"]
        self.num_layers = model_params["num_layers"]