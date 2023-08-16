""" 
This is the heart of pythae! 
Here are implemented some of the most common (Variational) Autoencoders models.

By convention, each implemented model is stored in a folder located in :class:`pythae.models`
and named likewise the model. The following modules can be found in this folder:

- | *modelname_config.py*: Contains a :class:`ModelNameConfig` instance inheriting
    from either :class:`~pythae.models.base.AEConfig` for Autoencoder models or 
    :class:`~pythae.models.base.VAEConfig` for Variational Autoencoder models. 
- | *modelname_model.py*: An implementation of the model inheriting either from
    :class:`~pythae.models.AE` for Autoencoder models or 
    :class:`~pythae.models.base.VAE` for Variational Autoencoder models. 
- *modelname_utils.py* (optional): A module where utils methods are stored.
"""

from .ae import AE, AEConfig
from .auto_model import AutoModel
from .base import BaseAE, BaseAEConfig
from .beta_vae_gp import (
    BetaVAEgp,
    BetaVAEgpInd,
    BetaVAEgpPrior,
    BetaVAEgpPost,
    BetaVAEgpCondInd,
)

from .beta_vae_gp import (
    BetaVAEgpConfig,
    BetaVAEgpIndConfig,
    BetaVAEgpPriorConfig,
    BetaVAEgpPostConfig,
    BetaVAEgpCondIndConfig,
)

__all__ = [
    "AutoModel",
    "BaseAE",
    "BaseAEConfig",
    "AE",
    "AEConfig",
    "VAE",
    "VAEConfig",
    "BetaVAEgp",
    "BetaVAEgpConfig",
    "BetaVAEgpInd",
    "BetaVAEgpIndConfig",
    "BetaVAEgpPrior",
    "BetaVAEgpPriorConfig",
    "BetaVAEgpPost",
    "BetaVAEgpPostConfig",
    "BetaVAEgpCondInd",
    "BetaVAEgpCondIndConfig",
]
