import numpy as np
from tqdm import *

temp = 1e-1

def gaussian_kernel(x, x0):
  dim = x0.shape[1]
  x = x.reshape((1, -1))
  exp_term = np.sum((x - x0) ** 2, axis = 1)
  main_term = np.exp(exp_term / (2 * temp)) 
  coeff = 1 / np.sqrt(2 * np.pi * temp) ** dim
  prod = coeff * main_term
  return np.sum(prod) / x0.shape[0]

def get_likelihood(data, pred):
  lh = np.zeros(pred.shape[0])
  dim = pred.shape[1]
  for i in tqdm(range(pred.shape[0])):
    lh[i] = gaussian_kernel(pred[i, :], data)
  return np.mean(lh)

def get_ll(data, pred):
  return np.log(get_likelihood(data, pred))

def get_nll(data, pred):
  return -get_ll(data, pred)

def get_nll_bits_per_dim(data, pred):
  nll = get_nll(data, pred)
  dim = data.shape[0]
  return nll / (np.log(2) * dim)
    

if __name__ == '__main__':
  dim = 2
  rng = np.random.default_rng()
  train_samples, test_samples = 10_000, 5_000
  data = 1 + np.sqrt(0.5) * rng.normal(size = (train_samples, dim))
  pred = rng.normal(size = (test_samples, dim))
  print(get_nll_bits_per_dim(data, pred))