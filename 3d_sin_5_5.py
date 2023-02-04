import numpy as np
import os
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def save_data(name, train, test, bounds, overwrite = False):
    if not os.path.exists(os.path.join('data/', f'{name}_test.npy')) or overwrite:
        np.save(os.path.join('data/', f'{name}_train.npy'), train)
        np.save(os.path.join('data/', f'{name}_test.npy'), test)
        np.save(os.path.join('data/', f'{name}_bounds.npy'), bounds)

def make_plot(samples):
    plt.scatter(samples[:, 0], samples[:, 1], s = 2)


rotmat = np.array([
  [+np.sqrt(1/3), +np.sqrt(1/3), +np.sqrt(1/3)],
  [-np.sqrt(1/6), +np.sqrt(2/3), -np.sqrt(1/6)],
  [-np.sqrt(1/2), +np.sqrt(0/3), +np.sqrt(1/2)],
]) 

train_samples, test_samples = 10_000, 5_000
x = -1 + 2 * np.random.random(2 * train_samples)
y = -1 + 2 * np.random.random(2 * train_samples)
xy = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis = 1)
xy = xy[np.linalg.norm(xy, axis = 1) < 1]
a = 1
z = a * np.sin(5 * xy[:, 0]) * np.cos(5 * xy[:, 1]) * (1 - np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2))
z = z + 0.01 * np.random.randn(z.shape[0])
train = np.concatenate((xy, z.reshape((-1, 1))), axis = 1)
# train = train @ rotmat
train = train[:, [1, 2, 0]]

x = -1 + 2 * np.random.random(2 * test_samples)
y = -1 + 2 * np.random.random(2 * test_samples)
xy = np.concatenate((x.reshape((-1, 1)), y.reshape((-1, 1))), axis = 1)
xy = xy[np.linalg.norm(xy, axis = 1) < 1]
a = 1
z = a * np.sin(5 * xy[:, 0]) * np.cos(5 * xy[:, 1]) * (1 - np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2))
z = z + 0.01 * np.random.randn(z.shape[0])
test = np.concatenate((xy, z.reshape((-1, 1))), axis = 1)
test = test[:, [1, 2, 0]]

name = '3d_sin_5_5_y'
fig = plt.figure()
ax = Axes3D(fig)
bounds = np.array([[-1, 1]] * 3)
plot_graph = ax.scatter(train[:, 0], train[:, 1], train[:, 2], s = 2)
plt.show()
save_data(name, train, test, bounds)
