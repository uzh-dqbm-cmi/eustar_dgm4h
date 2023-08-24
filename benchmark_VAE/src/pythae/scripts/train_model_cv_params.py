import pickle
import sys
import itertools

sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/")
sys.path.append("/home/cctrotte/krauthammer/eustar_clean/benchmark_VAE/src/")

sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ml4h_ct/benchmark_VAE/")
sys.path.append("/cluster/work/medinfmk/EUSTAR2/code_ml4h_ct/benchmark_VAE/src/")

from pythae.trainers import BaseTrainerConfig
from pythae.pipelines.training import TrainingPipeline
from pythae.models import BetaVAEgp, BetaVAEgpCondInd
from pythae.models import (
    BetaVAEgpCondIndConfig,
)
from pythae.models.beta_vae_gp.encoder_decoder import (
    Indep_MLP_Decoder,
    Guidance_Classifier,
    LSTM_Encoder,
    LSTM_Retrodiction_Encoder,
    LSTM_Retrodiction_Decoder,
)

from pythae.ssc.utils import (
    load_cv,
    get_classifier_config,
    remove_short_samples,
)

from pythae.models.beta_vae_gp.classifier_config import (
    EncoderDecoderConfig,
    PriorLatentConfig,
    DecoderConfig,
)
from pythae.models.beta_vae_gp.prior_latent import PriorLatent

import numpy as np
import torch
import random

import pandas as pd


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    local = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if local:
        data_path = "/home/cctrotte/krauthammer/eustar_clean/fake_data/processed/"
    else:
        data_path = "/cluster/work/medinfmk/EUSTAR2/data/processed/ct/"

    # name = "_reduced"
    name = "_ml4h"
    # name = "_allcont"
    with open(data_path + "bodies_" + name + ".pkl", "rb") as file:
        bodies = pickle.load(file)
    with open(data_path + "cohorts_" + name + ".pkl", "rb") as file:
        cohorts = pickle.load(file)

    (
        data_train_folds,
        data_valid_folds,
        data_test_folds,
        varNames,
        varSplits,
        xyt0,
        xyt1,
    ) = load_cv(data_path, name=name)
    var_names0 = [var.name for var in (bodies[0].variables + bodies[0].labels)]
    var_weights0 = [
        var.class_weight_norm for var in (bodies[0].variables + bodies[0].labels)
    ]

    names_x0 = [vN for i, vN in enumerate(var_names0) if xyt0[i] == "x"]
    names_y0 = [vN for i, vN in enumerate(var_names0) if xyt0[i] == "y"]
    weights_x0 = [vW for i, vW in enumerate(var_weights0) if xyt0[i] == "x"]
    weights_y0 = [vW for i, vW in enumerate(var_weights0) if xyt0[i] == "y"]

    kinds_x0 = [
        var.kind
        for var in (bodies[0].variables + bodies[0].labels)
        for nx in names_x0
        if var.name == nx
    ]
    kinds_y0 = [
        var.kind
        for var in (bodies[0].variables + bodies[0].labels)
        for nx in names_y0
        if var.name == nx
    ]
    splits_x0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "x"]
    splits_y0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "y"]
    splits_s0 = [vN for i, vN in enumerate(varSplits) if xyt0[i] == "s"]
    # remove samples of length 0 or 1
    for i, (data_train, data_valid, data_test) in enumerate(
        zip(data_train_folds, data_valid_folds, data_test_folds)
    ):
        data_train, data_valid, data_test = remove_short_samples(
            data_train, data_valid, data_test
        )

        data_train_folds[i] = data_train
        data_valid_folds[i] = data_valid
        data_test_folds[i] = data_test

    input_size = sum(splits_x0)

    static_size = sum(splits_s0)
    latent_dim = 22
    model_name = "VAE"
    best_params = best = {
        "samp_true_fixed_var_true": (0.05, 100, 3, [100, 100], [100, 100], [20], [40]),
        "samp_true_fixed_var_false": (0.05, 100, 3, [100, 100], [100, 100], [20], [40]),
        "samp_false_fixed_var_true": (0.05, 100, 1, [100, 100], [100], [100], [40]),
        "samp_false_fixed_var_false": (0.1, 100, 1, [100, 100], [100], [100], [40]),
    }
    params = {
        "dropout": 0.05,
        "lstm_hidden_size": 100,
        "num_lstm_layers": 3,
        "hidden_dims_enc": [100, 100],
        "hidden_dims_emb_dec": [100, 100],
        "hidden_dims_log_var_dec": [20],
        "classif_layers": [40],
    }

    predict = True
    sample_ = True
    fixed_variance = False
    retrodiction = False
    # to create classifier configs. Specify each classifier name, variables to predict in y, z dimensions to use and architecture of the classifier
    classifier_config = {
        "lung_inv": {
            "y_names": ["LUNG_ILD_involvement_or"],
            "z_dims": np.arange(0, 7),
            "layers": params["classif_layers"],
            "type": "static",
        },
        "lung_stage": {
            "y_names": ["LUNG_ILD_stage_or"],
            "z_dims": np.arange(0, 7),
            "layers": params["classif_layers"],
            "type": "static",
        },
        "heart_inv": {
            "y_names": ["HEART_involvement_or"],
            "z_dims": np.arange(7, 15),
            "layers": params["classif_layers"],
            "type": "static",
        },
        "heart_stage": {
            "y_names": ["HEART_stage_or"],
            "z_dims": np.arange(7, 15),
            "layers": params["classif_layers"],
            "type": "static",
        },
        "arthritis_inv": {
            "y_names": ["ARTHRITIS_involvement_or"],
            "z_dims": np.arange(15, 22),
            "layers": params["classif_layers"],
            "type": "static",
        },
        "arthritis_stage": {
            "y_names": ["ARTHRITIS_stage_or"],
            "z_dims": np.arange(15, 22),
            "layers": params["classif_layers"],
            "type": "static",
        },
    }
    # weights for the different losses
    beta = 0.01
    # overall weight factor for the classifiers
    w_class = {
        "lung_inv": 0.2,
        "lung_stage": 0.2,
        "heart_inv": 0.2,
        "heart_stage": 0.2,
        "arthritis_inv": 0.2,
        "arthritis_stage": 0.2,
    }
    w_recon = 1

    # weights for the different losses

    # overall weight factor for the classifiers
    w_class_pred = {
        "lung_inv": 0.2,
        "lung_stage": 0.2,
        "heart_inv": 0.2,
        "heart_stage": 0.2,
        "arthritis_inv": 0.2,
        "arthritis_stage": 0.2,
    }
    # w_recon = max(0, 1 - beta - sum(w_class.values()))
    w_recon_pred = 1

    if w_recon == 0:
        print(f"reconstruction loss weight is 0")
    classifier_configs = get_classifier_config(names_y0, splits_y0, classifier_config)

    encoder_config = EncoderDecoderConfig.from_dict(
        {
            "input_dim": input_size + static_size,
            "output_dim": latent_dim,
            "latent_dim": latent_dim,
            "hidden_dims": params["hidden_dims_enc"],
            "cond_dim_time_input": 1,
            "lstm_": True,
            "lstm_hidden_size": params["lstm_hidden_size"],
            "num_lstm_layers": params["num_lstm_layers"],
            "device": device,
            "dropout": params["dropout"],
            "predict": predict,
        }
    )
    decoder_config = DecoderConfig.from_dict(
        {
            "latent_dim": latent_dim,
            "output_dim": input_size,
            "fixed_variance": fixed_variance,
            "hidden_dims": [],
            "hidden_dims_emb": params["hidden_dims_emb_dec"],
            "hidden_dims_log_var": params["hidden_dims_log_var_dec"],
            "cond_dim_time_latent": 1,
            "cond_dim_static_latent": static_size,
            "lstm_": False,
            "dropout": params["dropout"],
            "device": device,
        }
    )

    prior_config = PriorLatentConfig.from_dict(
        {
            "input_dim": 1 + static_size,
            "latent_dim": latent_dim,
            "hidden_dims": [50],
            "device": device,
            "dropout": params["dropout"],
        }
    )

    to_reconstruct_x = [(name, index, True) for index, name in enumerate(names_x0)]

    to_reconstruct_y = [(name, index, True) for index, name in enumerate(names_y0)]
    model_config = BetaVAEgpCondIndConfig(
        input_dim=(input_size,),
        sample=sample_,
        latent_dim=latent_dim,
        w_class=w_class,
        w_recon=w_recon,
        beta=beta,
        w_class_pred=w_class_pred,
        w_recon_pred=w_recon_pred,
        missing_loss=True,
        latent_prior_noise_var=1,
        classifier_config=classifier_configs,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        prior_config=prior_config,
        splits_x0=splits_x0,
        kinds_x0=kinds_x0,
        splits_y0=splits_y0,
        kinds_y0=kinds_y0,
        names_x0=names_x0,
        weights_x0=weights_x0,
        weights_y0=weights_y0,
        to_reconstruct_x=to_reconstruct_x,
        to_reconstruct_y=to_reconstruct_y,
        device=device,
        predict=predict,
        retrodiction=retrodiction,
        progression=False,
    )

    for k in range(len(data_train_folds)):
        print(f"Combination {i} fold {k}")

        output_dir = (
            "samp_"
            + str(sample_)
            + "pred_"
            + str(predict)
            + "var_fixed"
            + str(fixed_variance)
            + "_cv_final/"
        )
        config = BaseTrainerConfig(
            output_dir=output_dir + str(k),
            learning_rate=1e-3,
            batch_size=100,
            num_epochs=80,
            customized=True,  # if we use the cusomized data loader for different sized patients
        )
        if retrodiction:
            my_encoder = LSTM_Retrodiction_Encoder(encoder_config)
        else:
            my_encoder = LSTM_Encoder(encoder_config)
        # my_encoder = Indep_MLP_Encoder(model_config)
        if decoder_config.lstm_:
            my_decoder = LSTM_Retrodiction_Decoder(decoder_config)
        else:
            my_decoder = Indep_MLP_Decoder(decoder_config)

        if prior_config is not None:
            prior_latent = PriorLatent(prior_config)
        else:
            prior_latent = None

        my_classifiers = [
            Guidance_Classifier(config) for config in model_config.classifier_config
        ]
        # my_classifier = Guidance_Classifier(model_config)

        model = BetaVAEgpCondInd(
            model_config=model_config,
            encoder=my_encoder,
            decoder=my_decoder,
            classifiers=my_classifiers,
            prior_latent=prior_latent,
        )

        pipeline = TrainingPipeline(training_config=config, model=model)
        try:
            pipeline(train_data=data_train_folds[k], eval_data=data_valid_folds[k])
        except Exception as e:
            print(e)
            continue

    print("End")
