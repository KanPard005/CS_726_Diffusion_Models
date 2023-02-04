import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
from tqdm import tqdm
from PIL import Image
import os

from dataloader import *
from helper_classes import *

class DiffusionModel:
  def __init__(self, dim, steps, network, lbeta = None, ubeta = None):
    self.dim = dim
    self.steps = steps
    self.network = network
    self.lbeta = lbeta
    self.ubeta = ubeta

    self.ema = EMA(0.75)
    self.ema.register(self.network)

    self.betas, self.alphas, self.alpha_bars = self.get_schedule()

  def get_schedule(self):
      betas = torch.linspace(start = self.lbeta, end = self.ubeta, steps = self.steps)
      alphas = 1 - betas
      alpha_bars = torch.cumprod(alphas, dim = 0)
      return betas, alphas, alpha_bars

  def q_sample(self, x0, t):
    x0 = torch.Tensor(x0)
    norm = torch.randn_like(x0)
    ab = self.alpha_bars[t]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * norm
    
  def p_sample(self, xt, t):
    t_tensor = t * torch.ones((xt.shape[0], 1))
    t_tensor = torch.cat((torch.sin(0.1 * t_tensor / self.steps), torch.cos(0.1 * t_tensor / self.steps)), dim = 1)
    xt_app = torch.cat((xt, t_tensor), dim = 1)

    beta = self.betas[t]
    alpha = self.alphas[t]
    alpha_bar = self.alpha_bars[t]

    mod_res = self.network(xt_app)
    term1 = beta * mod_res / (1 - alpha_bar).sqrt()
    term1 = (xt - term1) / alpha.sqrt()

    norm = beta.sqrt() * torch.randn_like(term1)
    term2 = norm if t > 0 else 0
    return term1 + term2

  def train(self, dataloader, epochs):
    loss_ls = []
    for i in tqdm(range(epochs), ascii = True):
      loss_epoch = 0
      num_s = 0
      for data in dataloader:
        data = data.float()
        n_samples = data.shape[0]
        t = np.random.randint(0, self.steps, (n_samples, ))
        t_tensor = torch.Tensor(t).reshape((-1, 1))
        t_tensor = torch.cat((torch.sin(0.1 * t_tensor / self.steps), torch.cos(0.1 * t_tensor / self.steps)), dim = 1)

        alpha_bar = self.alpha_bars[t].reshape((-1, 1))
        norm = torch.randn_like(data)
        inp = alpha_bar.sqrt() * data + (1 - alpha_bar).sqrt() * norm
        inp = torch.cat((inp, t_tensor), dim = 1)

        mod_res = self.network(inp)
        loss = self.network.loss(norm, mod_res)

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        self.ema.update(self.network)
        loss_epoch += (loss.cpu().detach().numpy())
        num_s += n_samples
      loss_epoch = loss_epoch/num_s
      loss_ls.append(loss_epoch)
      if i % (epochs // 10) == 0:
        print(f"{i / (epochs // 10) * 10}% done")
        print(f"Loss: {loss_epoch}")
        print("=" * 50)
    return loss_ls

  def sample(self, size, get_ls = False):
    xt = torch.tensor(np.random.randn(size, self.dim)).float()
    ls = [xt.cpu().detach().numpy()]
    for t in tqdm(range(self.steps), ascii = True):
        xt = self.p_sample(xt, self.steps - t - 1)
        if get_ls:
          ls.append(xt.cpu().detach().numpy())
    return xt.cpu().detach().numpy() if not get_ls else xt.cpu().detach().numpy(), ls
        
        
