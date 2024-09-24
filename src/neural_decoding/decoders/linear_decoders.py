from decoders.base_decoder import decoder
import numpy as np
import pickle
import torch
from tqdm import tqdm

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
        model_dict = {
            'theta': self.theta,
            'lbda': self.lbda,
            'intercept': self.intercept,
            'Model': "RR"
        }

        with open(filepath, "wb") as file:
            pickle.dump(model_dict, file)
        

    def load_model(self, filepath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """

        with open(filepath, "rb") as file:
           model_dict = pickle.load(file)

        
        if model_dict["Model"] != "RR":
            raise Exception("Tried to load model that isn't a Ridge Regression Instance")

        
        self.theta = model_dict['theta']
        self.intercept = model_dict['intercept']
        self.lbda = model_dict['lbda']




class KalmanFilter(decoder):
    def __init__(self, input_size, output_size, model_config):
        """
        Constructs a Kalman Filter decoder object (can use numpy arrays, torch arrays, or torch dataloaders). 
        
        Parameters:
            input_size (int); Not used (but inherits from decoder, so need it. Dummy variable)
            output_size (int): Not used (but inherits from decoder, so need it. Dummy variable)
            model_params (dict) containing four keys, 'append_ones_y' (bool) which specifies whether or not to add column of ones for bias, 
            'device' (bool) which specifies what device to train/run the model on, "start_y" (array) which specifies an initial [1, m] state, and "return_tensor" 
            (bool) which specifies whether a tensor should be returned or not.

            TODO: option to zero position uncertainty
            TODO: make it store the current state (currently resets every time)
            TODO: not tested on GPU
        """

        self.A, self.C, self.W, self.Q = None, None, None, None
        self.At, self.Ct = None, None
        self.append_ones_y = model_config["append_ones_y"]
        self.device = model_config["device"]
        self.start_y = model_config["start_y"]
        self.return_tensor = model_config["return_tensor"]

    def train(self, x, y):
        """
        Trains the matrices in the model. If append_ones_y is true, a column of ones is added to calculate bias. 

        Parameters:
            x (ndarray) size [n, m], where n is the number of samples, and m is the number of observation features.
            y (ndarray) size [n, k], where n is the number of samples, and k is the number of hidden state features.

        """
          
        if self.A is not None:
            raise ValueError("Tried to train a model that's already trained ")
        
        if self.append_ones_y:
            y = torch.cat((y, torch.ones([y.shape[0], 1])), dim=1)

        ytm1 = y[:-1, :]
        yt = y[1:, :]

        self.A = (yt.T @ ytm1) @ torch.pinverse(ytm1.T @ ytm1)                              # kinematic trajectory model
        self.W = (yt - (ytm1 @ self.A.T)).T @ (yt - (ytm1 @ self.A.T)) / (yt.shape[0] - 1)  # trajectory model noise
        self.C = (x.T @ y) @ torch.pinverse(y.T @ y)                                        # neural observation model
        self.Q = (x - (y @ self.C.T)).T @ (x - (y @ self.C.T)) / yt.shape[0]                # observation model noise

        self.At = self.A.T
        self.Ct = self.C.T

    def forward(self, input):
        """
        Runs a forward pass, by calling a predict method (torch or numpy).

        Parameters:
            input (ndarray) size [n, m], where n is the number of samples/data, and m is the number of observation features
        
        Returns:
            yhat (ndarray) prediction of size [n, k], where n is the number of samples/data, and k is the number of hidden state features
        """
        yhat = self.predict_numpy(input)
        return yhat

    # def predict(self, x, start_y=None):
    #     ''' Translated directly from Sam's matlab code kfPredict.m'''
    #     x = x.view((x.shape[0], -1))
    #     yhat = torch.zeros((x.shape[0], self.A.shape[1]))
    #     if start_y:
    #         yhat[0, :] = start_y

    #     Pt = self.W.clone()

    #     # for t in range(1, yhat.shape[0]):
    #     for t in tqdm(range(1, yhat.shape[0])):
    #         yt = yhat[t - 1, :] @ self.At                               # predict new state
    #         Pt = self.A @ Pt @ self.At + self.W                         # compute error covariance
    #         K = torch.linalg.lstsq((self.C @ Pt @ self.Ct + self.Q).T,
    #                                (Pt @ self.Ct).T, rcond=None)[0].T   # compute kalman gain, where B/A = (A'\B')'
    #         yhat[t, :] = yt.T + K @ (x[t, :].T - self.C @ yt.T)         # update state estimate
    #         Pt = (torch.eye(Pt.shape[0]) - K @ self.C) @ Pt                              # update error covariance

    #     if self.append_ones_y:
    #         yhat = yhat[:, :-1]
    #     return yhat

    def predict_numpy(self, x):
        """
        Runs a forward pass, returning a prediction for all input datapoints. If start_y is true, initial state is added.

        Parameters:
            input (ndarray) size [n, m], where n is the number of samples/data, and m is the number of observation features
        
        Returns:
            yhat (ndarray) prediction of size [n, k], where n is the number of samples/data, and k is the number of hidden state features
        """
        x = x.view((x.shape[0], -1))
        x = x.numpy()

        self.A = self.A.numpy()
        self.W = self.W.numpy()
        self.C = self.C.numpy()
        self.Q = self.Q.numpy()

        yhat = np.zeros((x.shape[0], self.A.shape[1]))
        if self.start_y:
            yhat[0, :] = self.start_y

        Pt = self.W.copy()

        for t in tqdm(range(1, yhat.shape[0])):
            yt = yhat[t-1, :] @ self.A.T                                # predict new state
            Pt = self.A @ Pt @ self.A.T + self.W                        # compute error covariance
            K = np.linalg.lstsq((self.C @ Pt @ self.C.T + self.Q).T,
                                (Pt @ self.C.T).T, rcond=None)[0].T     # compute kalman gain, where B/A = (A'\B')'
            yhat[t, :] = yt.T + K @ (x[t, :].T - self.C @ yt.T)	        # update state estimate
            Pt = (np.eye(Pt.shape[0]) - K @ self.C) @ Pt	            # update error covariance

        if self.return_tensor:
            yhat = torch.Tensor(yhat)
        if self.append_ones_y:
            yhat = yhat[:, :-1]
        return yhat

    def save_model(self, fpath):
        """
        Saves the model in its current state at the specified filepath

        Parameters:
            filepath (path-like object) indicates the file path to save the model in
        """
        model_dict = {
            "A": self.A,
            "C": self.C,
            "W": self.W,
            'Q': self.Q,
            "Model": "KF"
        }

        with open(fpath, "wb") as file:
            pickle.dump(model_dict, file)

    def load_model(self, fpath):
        """
        Load model parameters from a specified location

        Parameters:
            filepath (path-like object) indicates the file path to load the model from
        """

        with open(fpath, "rb") as file:
           model_dict = pickle.load(file)

        if model_dict["Model"] != "KF":
            raise Exception("Tried to load model that isn't a Kalman Filter Instance")
        
        self.A = model_dict['A']
        self.C = model_dict['C']
        self.W = model_dict['W']
        self.Q = model_dict['Q']

