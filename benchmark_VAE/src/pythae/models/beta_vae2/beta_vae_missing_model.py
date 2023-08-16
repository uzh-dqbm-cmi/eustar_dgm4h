import os
from typing import Optional

import torch
import torch.nn.functional as F

from ...data.datasets import BaseDataset, MissingDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder
from ..vae import VAE
from .beta_vae_missing_config import BetaVAEmissConfig


class BetaVAEmiss(VAE):
    r"""
    :math:`\beta`-VAE model.

    Args:
        model_config (BetaVAEmissConfig): The Variational Autoencoder configuration setting the main
            parameters of the model.

        encoder (BaseEncoder): An instance of BaseEncoder (inheriting from `torch.nn.Module` which
            plays the role of encoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

        decoder (BaseDecoder): An instance of BaseDecoder (inheriting from `torch.nn.Module` which
            plays the role of decoder. This argument allows you to use your own neural networks
            architectures if desired. If None is provided, a simple Multi Layer Preception
            (https://en.wikipedia.org/wiki/Multilayer_perceptron) is used. Default: None.

    .. note::
        For high dimensional data we advice you to provide you own network architectures. With the
        provided MLP you may end up with a ``MemoryError``.
    """

    def __init__(
        self,
        model_config: BetaVAEmissConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
    ):
        VAE.__init__(self, model_config=model_config, encoder=encoder, decoder=decoder)

        self.model_name = "BetaVAEmiss"
        self.beta = model_config.beta

        self.missing_cond = model_config.missing_cond
        self.missing_loss = model_config.missing_loss

    def forward(self, inputs: MissingDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]

        # extract missing matrix
        if self.missing_cond or self.missing_loss:
            missing = inputs["missing"].reshape(x.shape[0], -1)

        encoder_output = self.encoder(x)

        mu, log_var = encoder_output.embedding, encoder_output.log_covariance

        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        # concat the missing matrix with the latent state
        if self.missing_cond:
            z_m = torch.cat([z, missing], dim=1)
            mu_m = torch.cat([mu, missing], dim=1)
        else:
            z_m = z
            mu_m = mu

        # reconstruction of the sample z
        recon_x = self.decoder(z_m)["reconstruction"]
        # reconstruction of the mean
        recon_m = self.decoder(mu_m)["reconstruction"]

        # compute non-missing matrix
        if self.missing_loss:
            non_missing = 1 - missing
        else:
            non_missing = 1

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, non_missing)

        output = ModelOutput(
            reconstruction_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            recon_m=recon_m,
            z=z,
            mu=mu,
            log_var=log_var,
        )

        return output

    def loss_function(self, recon_x, x, mu, log_var, non_missing):
        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (
                non_missing
                * F.mse_loss(
                    recon_x.reshape(x.shape[0], -1),
                    x.reshape(x.shape[0], -1),
                    reduction="none",
                )
            ).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = F.binary_cross_entropy(
                recon_x.reshape(x.shape[0], -1),
                x.reshape(x.shape[0], -1),
                reduction="none",
            ).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (
            (recon_loss + self.beta * KLD).mean(dim=0),
            recon_loss.mean(dim=0),
            KLD.mean(dim=0),
        )

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps
