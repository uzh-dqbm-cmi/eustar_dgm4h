from pythae.config import BaseConfig
from typing import List
from pydantic.dataclasses import dataclass
from dataclasses import field


@dataclass
class ClassifierConfig(BaseConfig):
    name: str = "ClassifierConfig"
    organ_name: str = ""
    y_names: List[str] = field(default_factory=lambda: [""])
    z_dims: List[int] = field(default_factory=lambda: [0])
    layers: List[int] = field(default_factory=lambda: [100])
    y_dims: List[int] = field(default_factory=lambda: [0])
    y_indices: List[int] = field(default_factory=lambda: [0])
    input_dim: int = 0
    output_dim: int = 0
    type: str = ""


@dataclass
class PredictorConfig(BaseConfig):
    name: str = "PredictorConfig"
    lstm_predictor: bool = False
    hidden_dims: List[int] = field(default_factory=lambda: [100])
    lstm_hidden_size: int = 30
    num_lstm_layers: int = 1
    input_dim: int = 0
    output_dim: int = 0
    device: str = "cpu"
    # for MLP predictor (number of previous embeddings to consider)
    n_previous_for_pred: int = 3


@dataclass
class EncoderDecoderConfig(BaseConfig):
    name: str = "EncoderDecoderConfig"
    input_dim: int = 0
    output_dim: int = 0
    latent_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [100])
    hidden_dims_emb: List[int] = field(default_factory=lambda: [100])
    hidden_dims_log_var: List[int] = field(default_factory=lambda: [100])
    hidden_dims_feat_spec: List[int] = field(default_factory=lambda: [100])
    splits: List[int] = field(default_factory=lambda: [0])
    lstm_: bool = False
    lstm_hidden_size: int = 30
    cond_dim_missing_latent: int = 0
    cond_dim_time_input: int = 0
    cond_dim_time_latent: int = 0
    cond_dim_static_latent: int = 0
    num_lstm_layers: int = 1
    device: str = "cpu"
    dropout: float = 0.0
    predict: bool = False


@dataclass
class DecoderConfig(BaseConfig):
    name: str = "DecoderConfig"
    input_dim: int = 0
    output_dim: int = 0
    latent_dim: int = 0
    device: str = "cpu"
    # whether to learn the variance or fix to 1
    fixed_variance: bool = False
    hidden_dims: List[int] = field(default_factory=lambda: [100])
    hidden_dims_emb: List[int] = field(default_factory=lambda: [100])
    hidden_dims_log_var: List[int] = field(default_factory=lambda: [100])
    hidden_dims_feat_spec: List[int] = field(default_factory=lambda: [100])
    splits: List[int] = field(default_factory=lambda: [0])
    lstm_: bool = False
    lstm_hidden_size: int = 30
    cond_dim_missing_latent: int = 0
    cond_dim_time_input: int = 0
    cond_dim_time_latent: int = 0
    cond_dim_static_latent: int = 0
    num_lstm_layers: int = 1
    device: str = "cpu"
    dropout: float = 0.0
    predict: bool = False


@dataclass
class PriorLatentConfig(BaseConfig):
    name: str = "PriorLatentConfig"
    latent_dim: int = 0
    input_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [100])
    device: str = "cpu"
    dropout: float = 0.0
