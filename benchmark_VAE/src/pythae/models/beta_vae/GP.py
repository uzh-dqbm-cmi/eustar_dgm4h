import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as col

# progress visualization
from tqdm.notebook import tqdm

import torch
import gpytorch


class PriorGP:
    def __init__(self):
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # initialize the kernel
        hypers0 = {
            "base_kernel.lengthscale": torch.tensor(0.3),
            "outputscale": torch.tensor(1.2),
        }

        covar_module.initialize(**hypers0)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_precision(self, x):
        MVN = self.forward(x)

        return MVN.precision_matrix


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
