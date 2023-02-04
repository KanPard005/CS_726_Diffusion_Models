import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import imageio
from tqdm import tqdm
from PIL import Image
import os
import datetime
from parzen_window import *
from earth_mover_distance import *
from chamferdist import ChamferDistance
from scipy.stats import norm
from pyemd.emd import emd

def get_image(DM, xt, step, steps, data_mean, data_cov, bounds, sampled, q_sample):
  dim = xt.shape[1]
  if (dim == 2):
    fig = plt.figure(figsize = (10, 10))
    fig.tight_layout()
    subfigs = fig.subfigures(2, 2)
    axs = subfigs[0][0].subplots(1, 1)
    axs.scatter(xt[:, 0], xt[:, 1], s=2)
    if bounds is not None:
      axs.set_xlim(bounds[0, :])
      axs.set_ylim(bounds[1, :])
    else:
      axs.set_xlim(-7, 7)
      axs.set_ylim(-7, 7)

    axs = subfigs[0][1].subplots(2, 1)
    
    axs[0].set_title('Frame Number')
    axs[0].set_xlim(0, steps)
    axs[0].axes.yaxis.set_visible(False)
    axs[0].barh([0], step + 1)
    n = xt.shape[0]
    bins = int(n**0.3)*2
    im = np.zeros((bins, bins), dtype=int)
    for i in range(n):
      a = np.clip(((xt[i]+3)/(6/bins)).astype(int), 0, bins-1)
      im[bins - a[1] - 1, bins - a[0] - 1]+=1
    axs[1].imshow(im, cmap='viridis')

    axs = subfigs[1][0].subplots(1, 1)
    bins = np.linspace(-4, 4, 10)
    if not sampled:
      axs.set_title('x histogram')
      hist, edges = np.histogram(xt[:, 0], bins = np.linspace(-4, 4, 12))
      axs.set_ylim(-1, 1)
      axs.plot((edges[:-1] + edges[1:]) / 2, hist / xt.shape[0], c = 'orange', label = 'Current distribution')
      axs.plot((edges[:-1] + edges[1:]) / 2, norm.cdf(edges[1:]) - norm.cdf(edges[:-1]), c = 'blue', label = 'True Gaussian')
    else:
      axs.set_title('x difference')
      hist1, edges = np.histogram(xt[:, 0], bins = np.linspace(-4, 4, 12))
      hist2, edges = np.histogram(q_sample[:, 0], bins = np.linspace(-4, 4, 12))
      axs.set_ylim(-1, 1)
      axs.plot((edges[:-1] + edges[1:]) / 2, hist1 / xt.shape[0], c = 'blue', label = 'Proportion')
      axs.plot((edges[:-1] + edges[1:]) / 2, hist2 / xt.shape[0], c = 'orange', label = 'q_sample')

    axs = subfigs[1][1].subplots(1, 1)

    bins = np.linspace(-4, 4, 10)
    if not sampled:
      axs.set_title('y histogram')
      hist, edges = np.histogram(xt[:, 1], bins = np.linspace(-4, 4, 12))
      axs.set_ylim(-1, 1)
      axs.plot((edges[:-1] + edges[1:]) / 2, hist / xt.shape[0], c = 'orange', label = 'Current distribution')
      axs.plot((edges[:-1] + edges[1:]) / 2, norm.cdf(edges[1:]) - norm.cdf(edges[:-1]), c = 'blue', label = 'True Gaussian')
    else:
      axs.set_title('y difference')
      hist1, edges = np.histogram(xt[:, 1], bins = np.linspace(-4, 4, 12))
      hist2, edges = np.histogram(q_sample[:, 1], bins = np.linspace(-4, 4, 12))
      axs.set_ylim(-1, 1)
      axs.plot((edges[:-1] + edges[1:]) / 2, hist1 / xt.shape[0], c = 'blue', label = 'Proportion')
      axs.plot((edges[:-1] + edges[1:]) / 2, hist2 / xt.shape[0], c = 'orange', label = 'q_sample')
    fig.savefig(f"results/{step + 1}.png") 
    plt.close(fig)

  elif dim == 3:
    plt.clf()
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xt[:,0],xt[:,1],xt[:,2], c = 'blue', s = 2)
    plt.savefig(f"results/scat.png")
    plt.close()

    images = [Image.open(x) for x in ['results/scat.png']]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    total_height = sum(heights)
    new_im = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(f"results/{step + 1}.png")

def get_gif(DM, steps, showevery, dir, sampled):
  file = os.path.join('results/', dir, f'q_transform_{sampled}.gif')
  with imageio.get_writer(file, mode = 'I', duration = 0.5) as writer:
      image = imageio.imread(f'results/0.png')
      writer.append_data(image)
      for step in tqdm(list(range(1, steps, showevery)) + [steps], ascii = True):
          image = imageio.imread(f'results/{step}.png')
          writer.append_data(image)

def make_plots(DM, xt, steps, showevery=1, dir = None, bounds = None, sampled = False, q_samples = None):
  if dir is not None and not os.path.exists(os.path.join('results/', dir)):
      os.makedirs(os.path.join('results', dir))
  if not sampled:
    data_mean = np.mean(xt, axis = 0)
    data_cov = np.matmul(xt.T, xt) / xt.shape[0]
    get_image(DM, xt, -1, steps, data_mean, data_cov, bounds, sampled, xt)
  else:
    data_mean = [np.mean(x, axis = 0) for x in xt]
    data_cov = [np.matmul(x.T, x) / x.shape[0] for x in xt]
    get_image(DM, xt[-1], -1, steps, data_mean[0], data_cov[0], bounds, sampled, xt[-1])
  if q_samples is None:
    q_samples = []
  if not sampled:
    for step in list(range(0, steps, showevery)) + [steps - 1]:
      x = DM.q_sample(xt, step)
      x = x.cpu().detach().numpy()
      q_samples.append(x)
      get_image(DM, x, step, steps, data_mean, data_cov, bounds, sampled, q_samples[-1])
    get_gif(DM, steps, showevery, dir, sampled)
  else:
    i = 0
    for step in [steps - 1] + list(reversed(list(range(0, steps, showevery)))):
      x = xt[step]
      get_image(DM, x, step, steps, data_mean[step], data_cov[step + 1], bounds, sampled, q_samples[i])
      i += 1
    get_gif(DM, steps, showevery, dir, sampled)
  filtered = [file for file in os.listdir('results/') if file.endswith('.png')]
  rm = [os.remove(f'results/{file}') for file in filtered]
  return q_samples

def run_process(DM, train_data, train_loader, test_data, test_samples, bounds, name, steps, epochs, showevery=1, get_plots = True, train_dm = True, sampled = True, message=None):
    res = None
    q_samples = None
    ct = str(datetime.datetime.now())
    if get_plots:
        q_samples = make_plots(DM, train_data, steps, showevery=showevery, dir=os.path.join(name, ct), bounds = None)
    if train_dm:
        if not os.path.exists(os.path.join('results', name, ct)):
          os.makedirs(os.path.join('results', name, ct))
        loss_ls = DM.train(train_loader, epochs)
        plt.figure()
        plt.plot(range(len(loss_ls)), loss_ls)
        plt.savefig(os.path.join('results/', name, ct, f'loss.png'))

        res, res_ls = DM.sample(test_samples, get_ls = True)
        v = []
        for qs in q_samples:
          arr = np.arange(qs.shape[0])
          arr = np.random.choice(arr, test_samples, replace = False)
          v.append(qs[arr, :])
        q_samples = v
        qs = make_plots(DM, res_ls, steps, showevery=showevery, dir = os.path.join(name, ct), bounds = None, sampled = True, q_samples = q_samples)

        if DM.dim == 2:
          plt.figure()
          plt.scatter(res[:, 0], res[:, 1], s=2)
          plt.xlim(-6, 6)
          plt.ylim(-6, 6)
          plt.savefig(os.path.join('results/', name, ct, f'model_result.png'))
          plt.figure()
          plt.scatter(test_data[:, 0], test_data[:, 1], s=2)
          plt.xlim(-6, 6)
          plt.ylim(-6, 6)
          plt.savefig(os.path.join('results/', name, ct, 'test_data.png'))
        elif DM.dim == 3:
          fig = plt.figure()
          ax = Axes3D(fig)
          plot_graph = ax.scatter(res[:,0],res[:,1],res[:,2], c = 'blue', s = 2)
          ax.set_title('Helix dataset')
          plt.savefig(os.path.join('results/', name, ct, f'model_result.png'))

          fig = plt.figure()
          ax = Axes3D(fig)
          plot_graph = ax.scatter(test_data[:,0],test_data[:,1],test_data[:,2], c = 'blue', s = 2)
          ax.set_title('Helix dataset')
          plt.savefig(os.path.join('results/', name, ct, f'test_data.png'))

    if not os.path.exists(os.path.join('results', name, ct)):
      os.makedirs(os.path.join('results', name, ct))
    print('Writing log')
    with open(os.path.join('results/', name, ct, 'log.txt'), 'w') as f:
      if message is not None:
        f.write(message + "\n")
      f.write(f'Name of dataset: {name}\n')
      f.write(f'Dimension of data: {DM.dim} \n')
      f.write(f'Number of steps: {DM.steps} \n')
      f.write(f'Beta range: {DM.lbeta} to {DM.ubeta} \n')
      if train_dm:
        nll_bits_per_dim = get_nll_bits_per_dim(train_data, res)
        idx = np.random.choice(np.arange(res.shape[0]), size = min(500, res.shape[0]), replace = False)
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Negative log likelihood (bits per dimension): {nll_bits_per_dim}\n')

        f.write(f'Earth Mover Distance: {get_emd(train_data[idx], res[idx])}\n')

        source = torch.tensor(train_data).view((1, train_data.shape[0], train_data.shape[1])).float()
        target = torch.tensor(res).view((1, res.shape[0], res.shape[1])).float()
        chamferDist = ChamferDistance()
        dist_bidirectional = chamferDist(source, target, bidirectional=True)
        f.write(f'Chamfer distance: {dist_bidirectional}\n')

    print('Log written')
    return res
