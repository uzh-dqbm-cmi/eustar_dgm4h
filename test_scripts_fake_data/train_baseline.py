import pickle
import sys

sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/src/")

from pythae.data.datasets import MissingDataset

from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import RNNMLP
from pythae.models import (
    RNNMLPConfig
)
from pythae.models.beta_vae_gp.GP import predict_D, predict_D2
from pythae.models.beta_vae_gp.encoder_decoder import (
    MLP_Decoder,
    LSTM_Encoder,
    Indep_MLP_Decoder,
)
from pythae.models import AutoModel
from pythae.models.beta_vae_gp.utils import (
    load_missing_data_train_test,
    get_classifier_config,
)
from pythae.models.beta_vae_gp.body import Body
from pythae.models.beta_vae_gp.classifier_config import (
    ClassifierConfig,
    PredictorConfig,
    EncoderDecoderConfig,
    PriorLatentConfig,
    DecoderConfig
)
from pythae.models.beta_vae_gp.prior_latent import PriorLatent
from pythae.config import BaseConfig
import numpy as np
import torch
import random
import os
from pythae.models.beta_vae_gp.plots import plot_losses
import pandas as pd
from pythae.models.beta_vae_gp.plots import plot_losses, plot_recon_losses

if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    local = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if local:
        data_path = "/home/cctrotte/krauthammer/eustar/fake_data/processed/"
    else:
        data_path = "/cluster/work/medinfmk/EUSTAR2/data/processed/ct/"

    #name = "_reduced"
    name = "_medium"
    #name = "_allcont"
    with open(data_path + "body_" + name + ".pkl", "rb") as file:
        body = pickle.load(file)
    with open(data_path + "cohort_" + name + ".pkl", "rb") as file:
        cohort = pickle.load(file)

    (
        data_train,
        data_test,
        varNames,
        varSplits,
        xyt0,
        xyt1,
    ) = load_missing_data_train_test(data_path, name=name)
    var_names0 = [var.name for var in (body.variables + body.labels)]

    names_x0 = [vN for i, vN in enumerate(var_names0) if xyt0[i] == "x"]
    names_y0 = [vN for i, vN in enumerate(var_names0) if xyt0[i] == "y"]

    kinds_x0 = [
        var.kind
        for var in (body.variables + body.labels)
        for nx in names_x0
        if var.name == nx
    ]
    kinds_y0 = [
        var.kind
        for var in (body.variables + body.labels)
        for nx in names_y0
        if var.name == nx
    ]
    splits_x0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "x"]
    splits_y0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "y"]
    splits_s0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "s"]
    # remove samples of length 0 or 1
    data_train.list_x = [elem for elem in data_train.list_x if len(elem) >1]
    data_train.list_y = [elem for elem in data_train.list_y if len(elem) >1]
    data_train.list_t = [elem for elem in data_train.list_t if len(elem) >1]
    data_train.list_s = [elem for elem in data_train.list_s if len(elem) >1]

    data_train.missing_x = [elem for elem in data_train.missing_x if len(elem)> 1]
    data_train.missing_y = [elem for elem in data_train.missing_y if len(elem)> 1]
    data_train.missing_t = [elem for elem in data_train.missing_t if len(elem)> 1]
    data_train.missing_s = [elem for elem in data_train.missing_s if len(elem)> 1]
    data_test.list_x = [elem for elem in data_test.list_x if len(elem) >1]
    data_test.list_y = [elem for elem in data_test.list_y if len(elem) >1]
    data_test.list_t = [elem for elem in data_test.list_t if len(elem) >1]
    data_test.list_s = [elem for elem in data_test.list_s if len(elem) >1]
    data_test.missing_x = [elem for elem in data_test.missing_x if len(elem)> 1]
    data_test.missing_y = [elem for elem in data_test.missing_y if len(elem)> 1]
    data_test.missing_t = [elem for elem in data_test.missing_t if len(elem)> 1]
    data_test.missing_s = [elem for elem in data_test.missing_s if len(elem)> 1]
    input_size = sum(splits_x0)
    y_input_size = sum(splits_y0)

    static_size = sum(splits_s0)
    latent_dim = 20
    model_name = "RNN-MLP"
    encoder_config = EncoderDecoderConfig.from_dict({
        "input_dim": input_size + static_size,
        "output_dim": latent_dim,
        "latent_dim": latent_dim,
        "hidden_dims": [100, 100],
        "cond_dim_time_input": 1,
        "lstm_": True,
        "lstm_hidden_size": 50,
        "num_lstm_layers": 3,
        "device": device,
        "dropout": 0.2,
        "predict": True,

    })
    decoder_config = DecoderConfig.from_dict({
        "latent_dim": latent_dim,
        "fixed_variance": True,
        "output_dim": input_size,
        "hidden_dims": [100],
        "cond_dim_time_latent": 1,
        "cond_dim_static_latent": static_size,
        "lstm_hidden_size": 50,
        "lstm_": False,
        "dropout": 0.2,
        "device": device,})
    
    decoder_y_config = EncoderDecoderConfig.from_dict({
        "latent_dim": latent_dim,
        "output_dim": y_input_size,
        "hidden_dims": [30],
        "hidden_dims_emb": [20],
        "hidden_dims_log_var": [30],
        "cond_dim_time_latent": 1,
        
        "lstm_hidden_size": 50,
        "lstm_": False,
        "dropout": 0.2,
        "device": device,})
    
    to_reconstruct_x = [(name, index, True) for index, name in enumerate(names_x0)]
    to_reconstruct_y = [(name, index, True) for index, name in enumerate(names_y0)]

    my_encoder = LSTM_Encoder(encoder_config)
    my_decoder = Indep_MLP_Decoder(decoder_config)
    my_y_decoder = MLP_Decoder(decoder_y_config)
    config = BaseTrainerConfig(
        output_dir="my_model",
        learning_rate=1e-3,
        batch_size=100,
        num_epochs=4,  
        customized=True, 
        optimizer =  torch.optim.SGD,
        rnn = True, # use the rnn trainer
    )

    model_config = RNNMLPConfig(
        input_dim=input_size + static_size +1,
        latent_dim=latent_dim,
        missing_loss=True,
        predict_y = True,
        encoder_config = encoder_config,
        decoder_config = decoder_config,
        decoder_y_config = decoder_y_config,
        splits_x0=splits_x0,
        kinds_x0=kinds_x0,
        splits_y0=splits_y0,
        kinds_y0=kinds_y0,
        names_x0=names_x0,
        to_reconstruct_x = to_reconstruct_x,
        to_reconstruct_y = to_reconstruct_y,
        device=device,)
    model = RNNMLP(
        model_config=model_config,
        encoder=my_encoder,
        decoder=my_decoder,
        decoder_y=my_y_decoder,
    )

    pipeline = TrainingPipeline(training_config=config, model=model)
    pipeline(train_data=data_train, eval_data=data_test)
    # names_x = [vN for i, vN in enumerate(varNames) if xyt1[i] == "x"]
    # plot_recon_losses(pipeline, data_train)
    # i = 3
    
    # sample_batch_test = data_test.get_ith_sample_batch_with_customDataLoader(i, 1)
    # out = model(sample_batch_test)

    print("End")