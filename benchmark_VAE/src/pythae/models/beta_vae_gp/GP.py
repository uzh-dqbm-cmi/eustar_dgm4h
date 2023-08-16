import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as col

# progress visualization
from tqdm.notebook import tqdm

import torch
import gpytorch


def dynamic_mean(x, const):
    return torch.ones(x.shape[0], layout=x.layout, device=x.device) * const


class PriorGP(gpytorch.models.GP):
    def __init__(self, lengthscale=1.0, noise_var=0.0, scale=1.0):
        super(PriorGP, self).__init__()

        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = dynamic_mean
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.noise_var = noise_var
        self.jitter = 1e-4

        # initialize the kernel
        hypers0 = {
            "base_kernel.lengthscale": torch.tensor(lengthscale),
            "outputscale": torch.tensor(scale),
        }

        self.covar_module.initialize(**hypers0)

        # fix the parameters!
        for param in self.covar_module.parameters():
            param.requires_grad = False

        #######
        # self.covar_module.requires_grad = False
        # self.mean_module.requires_grad = True
        # self.mean_module.constant = 0

    def forward(self, x, pers_mean):
        mean_x = self.mean_module(x, pers_mean)
        covar_x = (
            self.covar_module(x) + torch.eye(x.shape[0]).to(x.device) * self.jitter
        )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward_noisy(self, x, pers_mean):
        mean_x = self.mean_module(x, pers_mean)
        covar_x = self.covar_module(x) + torch.eye(x.shape[0]).to(x.device) * (
            self.jitter + self.noise_var
        )

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior_cholesky(self, x, mu, var):
        # mu is mu
        # var is log_var.exp

        pers_mean = mu.mean()

        if True:
            # K = self.Kxx(x)

            MVN = self.forward(x, pers_mean)
            K = MVN.covariance_matrix  # store it?
            prior_mean = MVN.loc

            MVN = self.forward_noisy(x, pers_mean)
            # precision of the noisy prior GP
            iP = MVN.precision_matrix

            H = torch.matmul(K, iP)
            m = prior_mean + torch.matmul(H, mu - prior_mean)

            # covariance of posterior
            V = K - torch.matmul(H, K)

            # covariance of noisy posterior
            # V = V + torch.eye(K.shape[0]).to(K.device)*self.noise_var
            # V = V + torch.diag(var)

            L = torch.linalg.cholesky(V)

            # compute 1 posterior sample
            eps = torch.randn_like(m)
            z = m + torch.matmul(L, eps)

            eps2 = torch.randn_like(m)
            # std = torch.sqrt(var)
            std = np.sqrt(self.noise_var)
            z = z + std * eps2

            # P_fixed_noise = K + torch.eye(K.shape[0]).to(K.device)*0.1
            # iP_fixed_noise = torch.inverse(P_fixed_noise) # precision of the noisy prior GP
            iP = torch.inverse(K)

        else:
            V = torch.diag(var)
            L = V
            m = mu
            z = mu + torch.sqrt(var) * torch.randn_like(mu)
            iP = V

        return m, V, L, z, iP

    def Kzx(self, z, x):
        return self.covar_module(z, x)

    # def Kxx(self, x):

    # return self.covar_module(x)

    def Hzx(self, z, x, iKxx=None, returnKzx=False):
        if iKxx is None:
            MVN = self.forward(x)
            iKxx = MVN.precision_matrix  ## should we store it?

        Kzx = self.covar_module(z, x)
        Hzx = torch.matmul(Kzx, iKxx)

        if returnKzx:
            return Hzx, Kzx
        else:
            return Hzx

    def Qzx(self, z, x, iKxx=None, returnHzx=False):
        Hzx, Kzx = self.Hzx(z, x, iKxx, returnKzx=True)

        Qzz = torch.matmul(Hzx, Kzx.T)

        if returnHzx:
            return Qzz, Hzx
        else:
            return Qzz

    # def get_precision(self, x):

    # MVN = self.forward(x)

    # return MVN.precision_matrix

    def get_KL_post_prior(self, x, mu, w, mode="cholesky"):
        """
        compute KL[ N(mu,W) || N(m0,Sig) ] with
        posterior N(mu,W) where W = diag(w1,...,wn) and
        prior N(m0,Sig)
        that is
        KL = 0.5[ tr(iSig W) + (m0-mu)^T iSig (m0-mu) - n + log|Sig| - log|W|]
        """

        # mu is mu
        # w is log_var.exp

        pers_mean = mu.mean()

        # x.shape = (n,d_time)
        # mu.shape = (n,)
        # w.shape = (n,)
        # print(x.shape, mu.shape, w.shape)

        # mode = 'independent'
        # mode = 'precision'
        # mode = 'cholesky'

        if mode == "precision":
            MVN = self.forward(x, pers_mean)
            precision = MVN.precision_matrix
            m0 = MVN.loc

            trace = torch.dot(torch.diag(precision), w)
            square = torch.dot(torch.matmul(precision, m0 - mu), m0 - mu)
            n = x.shape[0]
            logSig = -torch.logdet(precision)
            logW = w.log().sum()

        elif mode == "cholesky":
            MVN = self.forward(x, pers_mean)
            L = MVN.scale_tril
            m0 = MVN.loc

            g = torch.linalg.solve_triangular(L, (m0 - mu).reshape(-1, 1), upper=False)
            H = torch.linalg.solve_triangular(L, torch.diag(torch.sqrt(w)), upper=False)

            trace = (H**2).sum()  # torch.dot( torch.diag(precision), w)
            square = (g**2).sum()
            n = x.shape[0]
            logSig = torch.diag(L).log().sum() * 2
            logW = w.log().sum()

        elif mode == "independent":  # independenta and diagonal
            # m0 = pers_mean
            # m0????

            trace = torch.sum(w)
            square = torch.dot(m0 - mu, m0 - mu)
            n = x.shape[0]
            logW = w.log().sum()
            logSig = 0.0

        # print('trace', trace, 'square', square, 'logSig', logSig, 'logW', logW)

        return 0.5 * (trace + square - n + logSig - logW)

    def get_KL_post_prior2(
        self,
        x,
        mu,
        W,
        pers_mean,
        iP,
        mode="precision",
    ):
        """
        compute KL[ N(mu,W) || N(m0,Sig) ] with
        posterior N(mu,W)
        prior N(m0,Sig)
        that is
        KL = 0.5[ tr(iSig W) + (m0-mu)^T iSig (m0-mu) - n + log|Sig| - log|W|]
        # TODO: make it for general prior including mean!
        """

        ### mu is ms!!
        ### W is Vs

        ### pers_mean is mean of mu

        if mode == "precision":
            # MVN = self.forward(x, pers_mean)
            # MVN = self.forward_noiseless(x, pers_mean)

            # precision = MVN.precision_matrix

            precision = iP  # noisy prior precision

            m0 = self.mean_module(x, pers_mean)  # MVN.loc

            trace = torch.trace(torch.matmul(precision, W))
            square = torch.dot(torch.matmul(precision, m0 - mu), m0 - mu)
            n = x.shape[0]
            logSig = -torch.logdet(precision)  # already inverted
            logW = torch.logdet(W)  # cholesky already computed!!!!!!

        # elif mode == 'cholesky':
        #    MVN = self.forward(x)
        #   L = MVN.scale_tril

        #   g = torch.linalg.solve_triangular(L, mu.reshape(-1,1), upper=False)
        #  H = torch.linalg.solve_triangular(L, torch.diag(torch.sqrt(w)), upper=False )

        #  trace = (H**2).sum() # torch.dot( torch.diag(precision), w)
        #  square = (g**2).sum()
        #  n = x.shape[0]
        #  logSig = torch.diag( L ).log().sum()*2
        # logW = w.log().sum()
        else:
            print(mode, " not implemented")

        # print('trace', trace, 'square', square, 'logSig', logSig, 'logW', logW)

        return 0.5 * (trace + square - n + logSig - logW)

    def predict(self, xs, mu, w, x, noise=False):
        ## we should store iKxx!
        ## diag k?
        ######## only compute diag!!!!!!!!!!!!!!!!!!

        # mu is mu
        # w is log_var.exp

        pers_mean = mu.mean()

        MVN = self.forward(x, pers_mean)
        iKxx = MVN.precision_matrix

        prior_mean = MVN.loc

        # H = Hzx(self, z, x, iKxx=iKxx)
        Q, H = self.Qzx(xs, x, iKxx=iKxx, returnHzx=True)

        k = self.covar_module(xs).diagonal()
        # k = torch.diag(torch.tensor([self.covar_module.eval(xs_i) for xs_i in xs]))
        # print('diag(kxx)!!!!')

        prior_mean_xs = self.mean_module(xs, pers_mean)

        ms = prior_mean_xs + torch.matmul(H, mu - prior_mean)
        vs = k - Q + torch.matmul(H * w, H.T)  # including noise?????

        if noise:
            vs += torch.eye(vs.shape[0]) * self.noise_var

        return ms, vs

    def predict2(self, xs, mu, W, x, pers_mean, noise=False):
        ## we should store iKxx!
        ## diag k?
        ######## only compute diag!!!!!!!!!!!!!!!!!!

        ##### mu is ms!
        ##### W is Vs!

        MVN = self.forward(x, pers_mean)  #### why noisy???????
        iKxx = MVN.precision_matrix

        prior_mean = MVN.loc

        # H = Hzx(self, z, x, iKxx=iKxx)
        Q, H = self.Qzx(xs, x, iKxx=iKxx, returnHzx=True)

        k = self.covar_module(xs).diagonal()
        # k = torch.diag(torch.tensor([self.covar_module.eval(xs_i) for xs_i in xs]))
        # print('diag(kxx)!!!!')

        prior_mean_xs = self.mean_module(xs, pers_mean)

        ms = prior_mean_xs + torch.matmul(H, mu - prior_mean)
        vs = k - Q + torch.matmul(torch.matmul(H, W), H.T)  # including noise?????

        if noise:
            vs += torch.eye(vs.shape[0]) * self.noise_var

        return ms, vs


def predict_D(trained_model, xs, MU, W, x, noise=False):
    """
    joint prediction for 1 patient and D dimensions for GPprior
    """

    # MU is mu
    # W is diag og_var.exp

    ms_ = []
    vs_ = []
    for d, GP in enumerate(trained_model.priorGPs):
        ms, vs = GP.predict(xs, MU[:, d], W[:, d], x, noise=noise)

        ms_.append(ms)
        vs_.append(vs)

    return torch.stack(ms_), torch.stack(vs_)


def predict_D2(trained_model, xs, MU, W, x, pers_means, noise=False):
    """
    joint prediction for 1 patient and D dimensions for GPpost
    """

    # MU is ms, GPmean
    # W is Vs, full

    ms_ = []
    vs_ = []
    for d, GP in enumerate(trained_model.priorGPs):
        # print(d,'Wshape',W.shape)

        ms, vs = GP.predict2(xs, MU[:, d], W[:, :, d], x, pers_means[d], noise=noise)

        ms_.append(ms)
        vs_.append(vs)

    return torch.stack(ms_), torch.stack(vs_)
