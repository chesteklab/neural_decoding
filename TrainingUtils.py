import torch
from torch.utils.data import Dataset
import numpy as np

dtype = torch.float
ltype = torch.long


class FingerDataset(Dataset):
    """Torch Dataset for predicting finger position/velocity from neural data.
       Nearly identical to the older 'OnlineDatasets', but here we pass in previously-loaded data
    """

    def __init__(self,
                 device,
                 X_neural,
                 Y_fings,
                 transform=None,
                 zero_train=False,
                 Resamp=None,
                 predtype='v',
                 numfingers=2,
                 numdelays=3,
                 positioninput=False,
                 last_timestep_recent=True,
                 predict_stopgo=False,
                 stopgo_thresh=0.1,
                 ):
        """
        Args:
            X_neural (ndarray): Neural data, [n, neu], where neu is the number of channels
            Y_neural (ndarray): Behavioral data, [n, dim], where dim is the number of behavioral states
            transform (callable, optional): Optional transform to be applied on a sample.
            zero_train (bool, optional): If true, will zero Y values that are below a threshold (only works if resamp is none)
            Resamp (indices, optional): An np array of indices to resample the data. Applied after history is added.
            predtype ('v', 'p', or 'pv'): selects if position, velocity, or both are in the 'y' output
            numdelays (int, optional): Number of delay bins. Defaults to 3.
            positioninput (bool, optional): If True, the previous position is appended to the neural data as additional features
            last_timestep_recent (bool, optional): If the last timestep should be the most recent data (used in RNNs)
            predict_stopgo (bool):      If True, adds a binary signal corresponding to if the DOF is moving or not.
            stopgo_thresh (float):      The absolute threshold for movement, applied to velocity data.
        """
        Xtrain_temp = torch.tensor(X_neural).to(device).to(dtype)
        Ytrain_temp = torch.tensor(Y_fings).to(device).to(dtype)

        # (optional) append the previous position(s) as additional input features
        if positioninput:
            prevpos = torch.cat((torch.zeros((1, numfingers)).to(device).to(dtype),
                                 Ytrain_temp[:-1, 0:numfingers]), dim=0)
            Xtrain_temp = torch.cat((Xtrain_temp, prevpos), dim=1)

        # add time delays to input features
        Xtrain1 = torch.zeros((int(Xtrain_temp.shape[0]), int(Xtrain_temp.shape[1]), numdelays), device=device,dtype=dtype)
        Xtrain1[:, :, 0] = Xtrain_temp
        for k1 in range(numdelays - 1):
            k = k1 + 1
            Xtrain1[k:, :, k] = Xtrain_temp[0:-k, :]

        if last_timestep_recent:
            # for RNNs, we want the last timestep to be the most recent data
            Xtrain1 = torch.flip(Xtrain1, (2,))

        # choose position/velocity/both
        if predtype == 'v':
            Ytrain1 = Ytrain_temp[:, numfingers:numfingers*2]
        if predtype == 'p':
            Ytrain1 = Ytrain_temp[:, 0:numfingers]
        if predtype == 'pv':
            Ytrain1 = Ytrain_temp[:, 0:numfingers*2]
        else:
            RuntimeError("Must specify prediction type as Position 'p', Velocity 'v', or both 'pv'")

        # if predicting stop/go, threshold velocities and append to y-data
        if predict_stopgo:
            assert 'v' in predtype, "to use stopgo decoding the pred_type must contain velocity"
            stopgo = (np.abs(Ytrain_temp[:, numfingers:numfingers * 2]) > stopgo_thresh).float()
            Ytrain1 = np.hstack((Ytrain1, stopgo))
            # print('WARNING ****** only using stopgo ******')
            #
            # plt.figure(figsize=(15, 6), dpi=120)
            # n = 200
            # plt.plot(2*Ytrain_temp[500:500+n, 0+1])
            # plt.plot(Ytrain_temp[500:500+n, 2+1])
            # plt.plot(2*stopgo[500:500+n, 0+1])
            # plt.axhline(stopgo_thresh, color='k')
            # plt.axhline(-stopgo_thresh, color='k')
            # plt.show()

        # (optional) resample velocities to based on resamp (resamp should be a vector of indices)
        if Resamp is not None and isinstance(Resamp, np.ndarray):
            ind1 = Resamp.astype(int)
            Xtrain = Xtrain1[ind1, :, :]
            Ytrain = Ytrain1[ind1, :]

        elif zero_train:
            ind = torch.sqrt(torch.sum(Ytrain1 ** 2, 1)) < 0.001
            Xtrain = Xtrain1[ind, :, :]
            Ytrain = Ytrain1[ind, :]

        else:
            Xtrain = Xtrain1
            Ytrain = Ytrain1

        # store the processed X/Y data
        self.chan_states = (Xtrain, Ytrain)
        self.transform = transform

    def __len__(self):
        return len(self.chan_states[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        chans = self.chan_states[0][idx, :]
        states = self.chan_states[1][idx, :]

        sample = {'states': states, 'chans': chans}

        if self.transform:
            sample = self.transform(sample)

        return sample





def calc_corr(y1,y2):
    """Calculates the correlation between y1 and y2 (tensors)
    """
    corr = []
    for i in range(y1.shape[1]):
        corr.append(np.corrcoef(y1[:, i], y2[:, i])[1, 0])
    return corr











