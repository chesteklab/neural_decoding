from .base_decoder import decoder
import numpy as np

class ridge_regression(decoder):
    def __init__(self, input_size, output_size, model_config) -> None:
        """
        Constructs a ridge_regression decoder object. To construct just a linear regression, use a lambda of 0.
        
        Parameters:
            input_size (int) the size of the input data
            output_size (int) the size of the output
            model_params (dict) containing two keys, 'lbda' (float) which specifies the lambda hyperparameter,
            and 'intercept' (bool) which specifies whether or not to add a column of ones to calculate bias.
        """
        self.lbda = model_config['lbda']
        self.intercept = model_config['intercept']
        

        self.theta = np.zeros((input_size + 1 if self.intercept else input_size, output_size))

    def train(self, x, y):
        """
        Trains the theta matrix (model weights according to the given input and output training data). If intercept it true,
        a column of ones is added to calculate bias. 

        Parameters:
            x (ndarray) size [n, m], where n is the number of samples, and m is the number of input features.
            y (ndarray) size [n, k], where n is the number of samples, and k is the number of output features.

        """
        if self.intercept:
            x = np.concatenate((x,np.ones((x.shape[0],1))), axis=1)
        temp = np.linalg.lstsq(np.matmul(x.T, x) + self.lbda*np.eye(x.shape[1]), np.matmul(x.T, y))
        self.theta = temp[0]
    
    def forward(self, x):
        """
        Runs a forward pass, returning a prediction for all input datapoints. If intercept is true, a column of ones is added for bias.

        Parameters:
            x (ndarray) size [n, m], where n is the number of samples/data, and m is the number of input features
        
        Returns:
            yhat (ndarray) prediction of size [n, k], where n is the number of samples/data, and k is the number of output features
        """
        if self.intercept:
            x = np.concatenate((x,np.ones((x.shape[0],1))), axis=1)
        
        yhat = np.matmul(x, self.theta)
        return yhat
    
    def save_model(self, filepath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in
        """
         # todo
        pass

    def load_model(self, filepath):
       """
       Load model parameters from a specified location

       Parameters:
            filepath (path-like object) indicates the file path to load the model from
       """

       # todo
       pass
