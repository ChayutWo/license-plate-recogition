# Import module and packages
import torch
from torch.utils.data import Dataset
import pandas as pd
from scipy.io import loadmat

import os
import os.path
from utils.variables import *

# Dataloader class
class RadarDataset(Dataset):
    # Radar signal dataset

    def __init__(self, csv_file, root_dir, transform=None):
        """
        csv_file (string): path to csv file containing data information
        root_dir (string): path to the data folder containing all data samples
        transform (callable, optional): data transformation after loaded
        """

        # Load dataframe containing data information
        self.radar_df = pd.read_csv(csv_file, header=None)
        self.radar_df.columns = ['filename', 'folder', 'category']

        # Data folder location
        self.root_dir = root_dir

        # Transformation
        self.transform = transform

    def __len__(self):
        # Total number of data
        return len(self.radar_df.category)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # dataloc: path to the data file
        # join root with folder and filename
        dataloc = os.path.join(self.root_dir,
                               self.radar_df.iloc[idx, 1],
                               self.radar_df.iloc[idx, 0])

        # load .mat file and get the variable xc which is the data
        signal = loadmat(dataloc)['xc']
        target = self.radar_df.iloc[idx, 2]
        if target > 4:
            target = 4
        # perform transformation
        if self.transform:
            signal = self.transform(signal)

        return signal.float(), target