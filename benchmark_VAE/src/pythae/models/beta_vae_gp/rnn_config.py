from pythae.models.base import BaseAEConfig
from typing import List
from pydantic.dataclasses import dataclass
from dataclasses import field
from .classifier_config import EncoderDecoderConfig, DecoderConfig


@dataclass
class RNNMLPConfig(BaseAEConfig):
    name: str = "RNNMLPConfig"
    input_dim: int = 0
    missing_loss: bool = False
    latent_dim: int = 0
    device: str = "cuda:0"
    predict_y: bool = True
    encoder_config: EncoderDecoderConfig = EncoderDecoderConfig()
    decoder_config: DecoderConfig = DecoderConfig()
    decoder_y_config: EncoderDecoderConfig = EncoderDecoderConfig()
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
    to_reconstruct_x: List = field(default_factory=lambda: ["", int, bool])
    to_reconstruct_y: List = field(default_factory=lambda: ["", int, bool])
