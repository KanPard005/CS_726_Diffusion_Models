# Diffusion Models for Toy Datasets

This code is inspired from [Denoising Diffusion Probabilistic Models](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf).

A basic installation (without CUDA) would involve the following libraries:
| Sr No | Library | Version Tested |
|---|---|---|
| 1 | Pytorch | 1.12.1 |
| 2 | NumPy | 1.21.5 |
| 3 | ImageIO | 2.19.3 |
| 4 | chamferdist | 1.0.0 | 
| 5 | emd | 2.0 | 
| 5 | pyemd | 0.5.1 | 
| 6 | MatPlotLib | 3.5.2 |
| 7 | Pillow | 9.2.0 |

As the code was created fairly recently, the latest versions of these libraries should work.

A brief description of the code files:
| Sr No | File | Remarks |
|---|---|---|
| 1 | `3d_sin_5_5.py` | Generating the 3d_sin_5_5 dataset | 
| 2 | `continuous_diffusion_toy.py` | Contains the main diffusion model |
| 3 | `data_gen.ipynb` | Generate datasets | 
| 4 | `dataloader.py` | Custom dataloader | 
| 5 | `diffusion_test.py` | Run diffusion model | 
| 6 | `earth_mover_distance.py` | Calculating earth mover distance (EMD) |
| 7 | `helper_classes.py` | Neural network and EMA | 
| 8 | `parzen_window.py` | Likelihood computation using Parzen Window Estimation |
| 9 | `utils.py` | Utilities (image and gif creation, log creation, etc) |
