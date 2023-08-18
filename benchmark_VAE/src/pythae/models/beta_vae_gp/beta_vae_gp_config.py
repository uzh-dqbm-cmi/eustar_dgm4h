from pydantic.dataclasses import dataclass
from dataclasses import field

# from benchmark_VAE.src.pythae.models.beta_vae_gp.classifier_config import (
#     ClassifierConfig,
#     PredictorConfig,
#     EncoderDecoderConfig
# )
from .classifier_config import (
    ClassifierConfig,
    PredictorConfig,
    EncoderDecoderConfig,
    PriorLatentConfig,
    DecoderConfig,
)
from ..vae import VAEConfig

from typing import List, Dict
import numpy as np


@dataclass
class BetaVAEgpConfig(VAEConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1
    """

    missing_loss: bool = False
    # hidden_dims: List[int] = field(default_factory=lambda: [100])
    # prediction strategy
    predict: bool = True
    # retrodiction strategy
    retrodiction: bool = False
    num_for_rec: str = "random"
    max_num_to_pred: int = 1
    # progression
    progression: bool = False
    device: str = "cuda:0"
    # KL weights
    beta: float = 1.0
    # classifier specific weight
    w_class: Dict = field(default_factory=lambda: {"heart": 0})
    # reconstr. weight
    w_recon: float = 1.0
    w_class_pred: Dict = field(default_factory=lambda: {"heart": 0})
    w_recon_pred: float = 1.0

    classifier_config: List[ClassifierConfig] = field(
        default_factory=lambda: [ClassifierConfig()]
    )
    classifier_dims: List[int] = field(default_factory=lambda: [10, 4])
    predictor_config: PredictorConfig = PredictorConfig()
    encoder_config: EncoderDecoderConfig = EncoderDecoderConfig()
    decoder_config: DecoderConfig = DecoderConfig()
    # prior_config: PriorLatentConfig = PriorLatentConfig()
    # non-reasonable default
    splits_x0: List[int] = field(
        default_factory=lambda: [1, 1]
    )  # non-reasonable default
    kinds_x0: List[str] = field(
        default_factory=lambda: ["", ""]
    )  # non-reasonable default
    splits_y0: List[int] = field(
        default_factory=lambda: [1, 1]
    )  # non-reasonable default
    kinds_y0: List[str] = field(
        default_factory=lambda: ["", ""]
    )  # non-reasonable default
    names_x0: List[str] = field(default_factory=lambda: ["", ""])
    # weighting for cross entropy
    weights_x0: List = field(default_factory=lambda: [1.0, 1.0])
    weights_y0: List = field(default_factory=lambda: [1.0, 1.0])
    to_reconstruct_x: List = field(default_factory=lambda: ["", int, bool])
    to_reconstruct_y: List = field(default_factory=lambda: ["", int, bool])

    # # only needed for the encoder/decoder
    # cond_dim_missing_latent: int = 0
    # cond_dim_time_input: int = 0
    # cond_dim_time_latent: int = 0


@dataclass
class BetaVAEgpIndConfig(BetaVAEgpConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1
    """

    latent_prior_noise_var: float = 1.0


@dataclass
class BetaVAEgpCondIndConfig(BetaVAEgpConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1
    """

    latent_prior_noise_var: float = 1.0
    cond_dim_time_input: int = 1
    # sample in latent space
    sample: bool = True


@dataclass
class BetaVAEgpPriorConfig(BetaVAEgpIndConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1
    """

    lengthscales: List[float] = field(default_factory=lambda: [1.0])
    scales: List[float] = field(default_factory=lambda: [1.0])


@dataclass
class BetaVAEgpPostConfig(BetaVAEgpPriorConfig):
    r"""
    :math:`\beta`-VAE model config config class

    Parameters:
        input_dim (tuple): The input_data dimension.
        latent_dim (int): The latent space dimension. Default: None.
        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'
        beta (float): The balancing factor. Default: 1
    """
