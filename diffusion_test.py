import torch
import os
import numpy as np

from dataloader import *
from helper_classes import *
from continuous_diffusion_toy import *
from utils import *

names = {
    1: 'correlated_gaussians',
    2: 'deformed_swiss_roll',
    3: 'gaussian_mixture',
    4: 'indep',
    5: 'indep_extreme',
    6: 'quadratic',
    7: 'symm_swiss_roll',
    8: 'x_gaussian',
    9: 'sinusoid',
    10: 'sinusoid_vert',
    11: 'symm_circ_sin',
    12: 'helix_3D',
    13: '3d_sin_5_5',
    14: '12_petals',
    15: '4_petals',
    16: '6_petals'
}

#####################################################
dim = 2
name_type = 11
schedule_type = 6
name = names[name_type]
test_samples = 5000
steps = 50
epochs = 2000
batch_size = 1000
#####################################################

train_data = CustomDataset(name=name, type='train', root_dir='./data')
test_data = CustomDataset(name=name, type='test', root_dir='./data')
bounds = train_data.bounds

network = NN([dim + 2, 500, 500, 500, 500, dim], 1e-4, 0)

lbeta, ubeta = 1e-5, 1.28e-2
print(f"lbeta: {lbeta}, ubeta: {ubeta}")
train_loader = DataLoader(train_data, batch_size, shuffle=True)

DM_vanilla = DiffusionModel(
    dim, 
    steps, 
    network, 
    lbeta = lbeta, 
    ubeta = ubeta
)

res = run_process(
    DM_vanilla, 
    train_data.data, 
    train_loader, 
    test_data.data, 
    test_samples, 
    bounds, name, steps, 
    epochs, 
    showevery=10,
    get_plots = True, 
    train_dm = True,
    message = 'Trying linear'
)
