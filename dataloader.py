import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

import imageio
from tqdm import tqdm
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, name, type, root_dir):
        self.root_dir = root_dir
        self.name = name
        data_path = os.path.join(root_dir, f'{name}_{type}.npy')
        bounds_path = os.path.join(root_dir, f'{name}_bounds.npy')
        self.data = np.load(data_path)
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        self.data = (self.data - mean)/std
        self.bounds = None

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, idx):
        return self.data[idx]
