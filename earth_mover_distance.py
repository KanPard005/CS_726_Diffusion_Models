from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from pyemd import emd

def get_emd(d1, d2):
  # d = cdist(d1, d2)
  d_comb = np.concatenate((d1, d2), axis = 0)
  dist = np.linalg.norm((d_comb), axis = 1).reshape((-1, 1))
  d1 = np.concatenate((np.zeros((d1.shape[0], 1)), d1), axis = 1)
  d2 = np.concatenate((np.ones((d2.shape[0], 1)), d2), axis = 1)
  d_comb = np.concatenate((d1, d2), axis = 0)
  app = np.concatenate((dist, d_comb), axis = 1)
  app = app[app[:, 0].argsort()]
  d1_sig, d2_sig = 1 - app[:, 1], app[:, 1]
  dist_sorted = app[:, 2:]
  dist = cdist(dist_sorted, dist_sorted)
  d1_sig = d1_sig.copy(order = 'C')
  d2_sig = d2_sig.copy(order = 'C')
  dist = dist.copy(order = 'C')
  return emd(d1_sig, d2_sig, dist)

if __name__ == '__main__':
  samples = 8_00
  g1 = np.random.randn(samples, 2)
  g2 = np.random.randn(samples, 2)
  print(get_emd(g1, g2))
  