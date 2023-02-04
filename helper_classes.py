import math
from tkinter import W
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

import imageio
from tqdm import tqdm
from PIL import Image
import os

from dataloader import *

class EMA(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

class NN(nn.Module):
  def __init__(self, layer_ls, lr, weight_decay):
    super(NN, self).__init__()
    od = OrderedDict()
    for i in range(1, len(layer_ls)):
      od["dense{}".format(i)] = nn.Linear(layer_ls[i - 1], layer_ls[i])
      if i < len(layer_ls) - 1:
        od["relu{}".format(i)] = nn.ReLU()

    self.neuralNet = nn.Sequential(od)
    self.optimizer = torch.optim.Adam(self.neuralNet.parameters(), lr = lr, weight_decay = weight_decay)
    self.loss = torch.nn.MSELoss()

  def forward(self, x):
    x = torch.Tensor(x)
    # x = torch.flatten(x)
    res = self.neuralNet(x)
    return res

  def train(self, x, y, batch_size, epochs):
    x, y = torch.Tensor(x), torch.Tensor(y)
    for i in range(epochs):
      for j in range(0, x.shape[0], batch_size):
        local_x, local_y = x[j: min(j + batch_size, x.shape[0] - 1), :], y[j: min(j + batch_size, x.shape[0] - 1), :]
        pred = self.neuralNet(local_x)
        loss = self.loss(pred, local_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      if i % (epochs // 10) == 0:
        print(f"{i / (epochs // 10) * 10}% done, loss = {loss}")

  def test(self, x, y, giveLoss = False):
    x, y = torch.Tensor(x), torch.Tensor(y)
    pred = self.neuralNet(x)
    if giveLoss:
      print(f"Test loss: {self.loss(pred, y)}")
    return pred
