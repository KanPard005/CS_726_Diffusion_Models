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

from config import *
from dataloader import *
from helper_classes import *
from schedules import *

class DiffusionModel:
  def __init__(self, dim, steps, network, alt = False, schedule=None, dim_order = None, data = None, lbeta = None, ubeta = None):
    self.dim = dim
    self.steps = steps
    self.network = network
    self.schedule = schedule
    self.lbeta = lbeta
    self.dim_order = dim_order
    self.ubeta = ubeta
    # betas: list of diagonal matrices containing variance schedule for variables. Can be non-identity as well
    self.alt = alt
    if self.alt:
      self.betas = self.alt_beta_schedule(data, lbeta, ubeta)
    else:
      if lbeta is not None and ubeta is not None:
        self.betas = self.get_beta_schedule(lbeta, ubeta)
      else:
        self.betas = self.get_beta_schedule()
    # alpha: list of diagonal matrices containing (1 - alpha) values
    self.alpha = self.get_alpha()
    # alpha_bar: list of alpha_bar diagonal matrices
    self.alpha_bar = self.get_alpha_bar()
    self.loss_fn = torch.nn.MSELoss()

  def get_coeff(self, cov_mat, beta, ubeta, step):
    var = np.diag(cov_mat)
    opt = 2
    if opt == 1:
      var_mat = np.sqrt(var).reshape((-1, 1)) @ np.sqrt(var).reshape((1, -1))
      covar = np.abs(cov_mat) - np.diag(var)
      covar = np.sum(covar / var_mat, axis  = 1) + 1e-6
      coeff = covar / var
      coeff = coeff / np.sum(coeff)
      coeff = beta * cov_mat.shape[0] * coeff
    elif opt == 2:
      covar = (np.triu(np.ones((self.dim, self.dim))) - np.eye(self.dim))* cov_mat
      mat = np.sqrt(var).reshape((-1, 1)) @ np.sqrt(var).reshape((1, -1))
      corr = np.abs(covar) / mat
      covar = np.sum(np.sqrt(corr), axis = 1)
      covar = np.exp(covar) / np.sum(np.exp(covar))
      var = var / np.max(var)
      var = np.exp(var) / np.sum(np.exp(var))
      w = step / self.steps
      coeff = w * covar + (1 - w) * var
      coeff = beta * cov_mat.shape[0] * coeff
    # elif opt == 3:
    #   covar = 
    mx = np.max(coeff)
    if mx >= 1:
      coeff = coeff * ubeta / mx
    return coeff

  def alt_beta_schedule(self, x, lbeta, ubeta):
    cov_mat = np.matmul(x.T, x) / x.shape[0]
    coeff = self.get_coeff(cov_mat, lbeta, ubeta, 0)
    betas = []
    betas.append(torch.Tensor(np.diag(coeff)))
    for i in range(1, self.steps):
      beta = lbeta + (ubeta - lbeta) * i / (self.steps - 1)
      coeff_mat = np.sqrt(1 - coeff).reshape((-1, 1)) @ np.sqrt(1 - coeff).reshape((1, -1))
      # coeff_mat = coeff_mat
      cov_mat = coeff_mat * cov_mat + np.diag(coeff)
      coeff = self.get_coeff(cov_mat, beta, ubeta, i)
      betas.append(torch.Tensor(np.diag(coeff)))
    print(f"Betas:")
    for beta in betas:
      print(torch.diag(beta), end = '|')
    print('')
    return betas
    
  def get_beta_schedule(self, lbeta = 0.1, ubeta = 0.9, show_fig=False, dir=None):
    if self.schedule == 'linear':
      betas = torch.linspace(lbeta, ubeta, self.steps)
      # print(betas.shape)
      # plt.figure()
      # plt.plot(betas)
      # plt.show()
      return [torch.eye(self.dim) * betas[step] for step in range(self.steps)]
    if self.schedule == 'sigmoid':
      sched = [schedule_sigmoid(low=lbeta, high=ubeta, steps=self.steps, jump=int(i*self.steps/(self.dim+1)), slope=10*(ubeta-lbeta)/self.steps) for i in range(self.dim,0,-1)]
      betas = torch.zeros((self.dim, self.steps))
      for i,sch in enumerate(sched):
        for j in range(self.steps):
          betas[i,j] = sch.call(j)
      return [torch.eye(self.dim) * betas[:,step] for step in range(self.steps)]

    if self.schedule == 'linear_diff':
      sched = []
      for i,dimensions in enumerate(self.dim_order):
        for d in dimensions:
          sched.append(schedule_linear_diff(dim = d, low=lbeta, high=ubeta, steps=self.steps, start_step=int(i*self.steps/len(self.dim_order)), end_step=int((i+1)*self.steps/len(self.dim_order))))
          # sched = [schedule_linear_diff(low=lbeta, high=ubeta, steps=self.steps, start_step=int(i*self.steps/self.dim), end_step=int((i+1)*self.steps/self.dim-1)) for i in range(self.dim)]
      betas = torch.zeros((self.dim, self.steps))
      for i,sch in enumerate(sched):
        for j in range(self.steps):
          betas[sch.dim,j] = sch.call(j)

      if show_fig:
        print('In show_fig')
        if dir is not None and not os.path.exists(os.path.join('results/', dir)):
          os.makedirs(os.path.join('results', dir))
        plt.figure()
        l = []
        for i in range(betas.shape[0]):
          plt.plot(betas[i])
          l.append(str(i))
        plt.legend(l)
        print('Making plot')
        plt.savefig(os.path.join("results", dir, f"noise.png"))


      return [torch.eye(self.dim) * betas[:,step] for step in range(self.steps)]

  def get_alpha(self):
    return [torch.eye(self.dim) - self.betas[step] for step in range(self.steps)]
    
  def get_alpha_bar(self):
    start = torch.eye(self.dim)
    start = torch.ones(self.dim)
    alpha_bar_ls = []
    for step in range(self.steps):
      start = start * torch.diag(self.alpha[step])
      alpha_bar_ls.append(torch.diag(start))
    return alpha_bar_ls
      
  def q_sample(self, x0, t):
    # mean = torch.mm(torch.sqrt(self.alpha_bar[t])), x0.view(-1, 1))
    # print("alpha_bar :", self.alpha_bar[t])
    x0 = torch.Tensor(x0)
    mean = torch.mm(x0, torch.sqrt(self.alpha_bar[t]))
    mean = mean.cpu().detach().numpy()
    var = (torch.eye(self.dim) - self.alpha_bar[t]).cpu().detach().numpy()
    res = np.random.multivariate_normal(np.zeros(self.dim), var, mean.shape[0])
    res = res + mean
    # for i in range(mean.shape[0]):
    #   mean[i, :] = np.random.multivariate_normal(mean[i, :], var)
    # return np.random.multivariate_normal(mean, var)
    return res

  def p_sample(self, xt, t):
    t_tensor = t * torch.ones((xt.shape[0], 1))

    # if from_middle:
    #   mid_ind = t // (self.steps // self.dim) * (self.steps // len(self.dim_order))


    #   alpha_bar = self.alpha_bar[t] / self.alpha_bar[mid_ind] * self.alpha[mid_ind]

    xt = torch.Tensor(xt.float())
    ####
    t_tensor = torch.cat((torch.sin(0.1 * t_tensor / self.steps), torch.cos(0.1 * t_tensor / self.steps)), dim = 1)
    bit_tensor = torch.zeros((xt.shape[0],1))
    bit_tensor[self.steps//2:self.steps] = 1.0
    # print(data.shape)
    # print(t_tensor.shape)
    # print(bit_tensor.shape)
    
    # print(t_tensor.shape)
    # mod_inp = torch.cat((mod_inp, t_tensor, bit_tensor), dim = 1).float()
    xt_app = torch.cat((xt, t_tensor, bit_tensor), dim = 1)
    ####
    # xt_app = torch.cat((xt, t_tensor), dim = 1)
    alph = self.alpha[t]
    alph_bar = self.alpha_bar[t]
    bet = self.betas[t]
    neuralNet = self.network

    # if from_middle:
    #   alph_bar = alpha_bar

    mod_res = neuralNet.forward(xt_app)
    coeff = torch.diag(torch.diag(bet) * (1 / torch.sqrt(1 - torch.diag(alph_bar))))
    term1 = torch.diag(torch.sqrt(1 / torch.diag(alph)))
    term2 = xt - torch.mm(mod_res, coeff)
    mean = torch.mm(term2, term1)
    mean = mean.cpu().detach().numpy()

    var = bet.cpu().detach().numpy()
    # print(f"Mean: {mean}")
    # print(f"Variance: {var}")
    res = torch.tensor(np.random.multivariate_normal(np.zeros(self.dim), var, mean.shape[0]))
    res = res + mean if t > 0 else torch.Tensor(mean)
    # for i in range(mean.shape[0]):
    #     mean[i, :] = torch.tensor(np.random.multivariate_normal(mean[i, :], var))
    # mean = mean
    return res 

  def train(self, dataloader, epochs):
    # x = torch.Tensor(x)
    # x = x[torch.randperm(x.size()[0])]
    loss_ls = []
    alpha_bar_tensor = torch.Tensor(np.array([np.array(torch.diag(self.alpha_bar[step])) for step in range(self.steps)]))
    beta_tensor = torch.Tensor(np.array([np.array(torch.diag(self.betas[step])) for step in range(self.steps)]))
    alpha_tensor = torch.Tensor(np.array([np.array(torch.diag(self.alpha[step])) for step in range(self.steps)]))
    loss_wt_tensor = beta_tensor / (alpha_tensor * (1 - alpha_bar_tensor))
    loss_wt_den = torch.sqrt(torch.sum(loss_wt_tensor * loss_wt_tensor, dim = 1))
    loss_wt_tensor = loss_wt_tensor / loss_wt_den.view(-1, 1)
    loss_wt_tensor = torch.sqrt(loss_wt_tensor)
    
    for i in tqdm(range(epochs)):
      loss_epoch = 0
      num_s = 0
      for data in dataloader:
        # local_x = x[j: min(j + batch_size, x.shape[0]), :]
        n_samples = data.shape[0]
        t = np.random.randint(0, self.steps, (n_samples, ))
        t_tensor = torch.Tensor(t).long()

        # if from_middle:
        #   mid_ind = torch.div(t_tensor, (self.steps // self.dim), rounding_mode = 'floor') * (self.steps // len(self.dim_order))
        #   alpha_bar1 = alpha_bar_tensor[t_tensor] / alpha_bar_tensor[mid_ind] * alpha_tensor[mid_ind]
        #   alpha_bar = alpha_bar_tensor[mid_ind] / alpha_tensor[mid_ind]
        # else:
        #   alpha_bar = alpha_bar_tensor[t_tensor]
        alpha_bar = alpha_bar_tensor[t_tensor]


        # for i in range(mid_ind.shape[0]):
        #     print(mid_ind[i], alpha_bar[i], alpha_bar1[i])

        neuralNet = self.network
        eps = torch.tensor(np.random.randn(data.shape[0], self.dim)).float()
        eps1 = torch.tensor(np.random.randn(data.shape[0], self.dim)).float()

        ###Finding weights for loss terms###
        loss_wts = loss_wt_tensor[t_tensor]
        ###Finding weights for loss terms###

        # if from_middle:
        #   term1 = data * torch.sqrt(alpha_bar)
        #   term2 = torch.sqrt(1 - alpha_bar) * eps1
        #   term3 = torch.sqrt(1 - alpha_bar1) * eps
        #   mod_inp = torch.sqrt(alpha_bar1) * (term1 + term2) + term3
        # else:
        #   term1 = data * torch.sqrt(alpha_bar)
        #   term2 = eps * torch.sqrt(1 - alpha_bar)
        #   mod_inp = term1 + term2

        term1 = data * torch.sqrt(alpha_bar)
        term2 = eps * torch.sqrt(1 - alpha_bar)
        mod_inp = term1 + term2

        t_tensor = torch.cat((torch.sin(t_tensor / self.steps).view(-1, 1), torch.cos(t_tensor / self.steps).view(-1, 1)), dim = 1)
        bit_tensor = torch.zeros((data.shape[0],1))
        bit_tensor[self.steps//2:self.steps] = 1.0

        mod_inp = torch.cat((mod_inp, t_tensor, bit_tensor), dim = 1).float()
        mod_res = neuralNet.forward(mod_inp)
        loss = neuralNet.loss(loss_wts * eps, loss_wts * mod_res)

        neuralNet.optimizer.zero_grad()
        loss.backward()
        neuralNet.optimizer.step()
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
    for t in tqdm(range(self.steps)):
        xt = self.p_sample(xt, self.steps - t - 1)
        if get_ls:
          ls.append(xt.cpu().detach().numpy())
    return xt if not get_ls else xt, ls
        
        