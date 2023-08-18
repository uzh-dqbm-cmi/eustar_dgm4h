import os
from typing import Optional

import torch
import torch.nn.functional as F
import math
import cloudpickle
import inspect
import sys
from copy import deepcopy
from ...data.datasets import BaseDataset, MissingDataset
from ..base.base_utils import ModelOutput
from ..nn import BaseDecoder, BaseEncoder

from .beta_vae_gp_config import BetaVAEgpConfig

from ..base.base_config import BaseAEConfig, EnvironmentConfig

from pythae.models.beta_vae_gp.beta_vae_gp_model import BetaVAEgp


class BetaVAEgpCondInd(BetaVAEgp):
    """ """

    def __init__(
        self,
        model_config: BetaVAEgpConfig,
        encoder: Optional[BaseEncoder] = None,
        decoder: Optional[BaseDecoder] = None,
        classifiers=None,
        mu_predictor=None,
        var_predictor=None,
        prior_latent=None,
    ):
        super().__init__(
            model_config=model_config,
            encoder=encoder,
            decoder=decoder,
            classifiers=classifiers,
            mu_predictor=mu_predictor,
            var_predictor=var_predictor,
        )

        self.model_name = "BetaVAEgpCondInd"

        self.sample_z = model_config.sample

        self.latent_prior_noise_var = model_config.latent_prior_noise_var
        self.prior_latent = prior_latent

    def forward(self, inputs: MissingDataset, **kwargs):
        """
        input, encoding, sampling, decoding, guidance, loss
        """

        ################################
        ### 1) INPUT
        ################################

        # input data (N_patients x T_i) x L
        data_x = inputs["data_x"]

        # static data
        data_s = inputs["data_s"]

        # non-missing entries in the input/output data
        if self.missing_loss:
            non_missing_x = 1 - inputs["missing_x"] * 1.0
        else:
            non_missing_x = 1

        splits = inputs["splits"]  # N_patients
        # times = inputs["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1
        times = inputs["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1

        if not self.classifiers is None:
            data_y = inputs["data_y"]  # N_patients x n_class
            non_missing_y = 1 - inputs["missing_y"] * 1.0  # N_patients x n_class
        else:
            data_y = 0
            non_missing_y = 0

        ################################
        ### 2) ENCODING
        ################################

        # encode the input into dimension D
        # mu: (N_patients x T_i) x D
        # log_var: (N_patients x T_i) x D

        if self.predict:
            data_x_splitted = torch.split(data_x, splits, dim=0)
            data_y_splitted = torch.split(data_y, splits, dim=0)
            data_s_splitted = torch.split(data_s, splits, dim=0)
            non_missing_x_splitted = torch.split(non_missing_x, splits, dim=0)
            non_missing_y_splitted = torch.split(non_missing_y, splits, dim=0)
            times_splitted = torch.split(times, splits, dim=0)
            data_x_recon = torch.cat(
                [
                    data_x_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            data_s_recon = torch.cat(
                [
                    data_s_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            data_y_recon = torch.cat(
                [
                    data_y_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            non_missing_x_recon = torch.cat(
                [
                    non_missing_x_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            non_missing_y_recon = torch.cat(
                [
                    non_missing_y_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            times_recon = torch.cat(
                [
                    times_splitted[pat].repeat(splits[pat] + 1, 1)
                    for pat in range(len(splits))
                ]
            )
            indices_recon = torch.cat(
                [
                    torch.cat(
                        [
                            torch.cat(
                                [
                                    torch.full((index, 1), True, device=self.device),
                                    torch.full(
                                        (splits[pat] - index, 1),
                                        False,
                                        device=self.device,
                                    ),
                                ],
                                dim=0,
                            )
                            for index in range(0, splits[pat] + 1)
                        ]
                    )
                    for pat in range(len(splits))
                ]
            ).flatten()
            encoder_input_x = self.reshape_input_for_lstm(data_x, data_s, times, splits)

        else:
            data_x_recon = data_x
            data_s_recon = data_s
            data_y_recon = data_y
            non_missing_x_recon = non_missing_x
            non_missing_y_recon = non_missing_y
            times_recon = times
            indices_recon = torch.full((data_x.shape[0], 1), True).flatten()

        encoder_input_x = self.reshape_input_for_lstm(data_x, data_s, times, splits)

        encoder_output = self.encoder(encoder_input_x, times, times)
        mu, log_var, delta_t = (
            encoder_output.embedding,
            encoder_output.log_covariance,
            encoder_output.delta_t,
        )

        # prior distribution
        if self.prior_latent is not None and self.sample_z:
            prior_output = self.prior_latent(
                torch.cat([times_recon, data_s_recon], dim=1)
            )
            mu_p, log_var_p = prior_output.prior_mu, prior_output.prior_log_var

        ################################
        ### 3) SAMPLING
        ################################

        # sample from the latent distribution
        # z: (N_patients x T_i) x D

        if self.sample_z:
            z, kwargs = self.sample(mu, log_var)

        ################################
        ### 4a) DECODING
        ################################

        if self.model_config.decoder_config.lstm_:
            decoder_input_z = self.reshape_input_for_lstm(
                z, data_s_recon, times_recon, splits
            )
            decoder_input_mu = self.reshape_input_for_lstm(
                mu, data_s_recon, times_recon, splits
            )
        else:
            if self.sample_z:
                decoder_input_z = torch.cat([z, times_recon, data_s_recon], axis=1)
            decoder_input_mu = torch.cat([mu, times_recon, data_s_recon], axis=1)

        # reconstruction of the sample z
        # recon: (N_patients x T_i) x L

        if self.sample_z:
            recon_x = self.decoder(decoder_input_z)["reconstruction"]  # sample
            recon_x_log_var = self.decoder(decoder_input_z)[
                "reconstruction_log_var"
            ]  # sample
        recon_m = self.decoder(decoder_input_mu)["reconstruction"]  # NN output
        recon_m_log_var = self.decoder(decoder_input_mu)[
            "reconstruction_log_var"
        ]  # NN output

        ################################
        ### 4b) GUIDANCE
        ################################
        # reconstruction labels
        y_recon_out, y_recon_out_m = torch.zeros_like(
            data_y_recon, device=self.device
        ), torch.zeros_like(data_y_recon, device=self.device)

        for classifier in self.classifiers:
            if classifier.type == "static":
                if self.sample_z:
                    y_recon_out[:, classifier.y_dims] = classifier.forward(
                        z[:, classifier.z_dims]
                    )
                y_recon_out_m[:, classifier.y_dims] = classifier.forward(
                    mu[:, classifier.z_dims]
                )

        if self.progression:
            raise NotImplementedError(
                "Progression classifier not implemented for this model"
            )

        ################################
        ### 5) LOSS
        ################################

        if self.sample_z:
            (
                loss_recon,
                loss_recon_stacked,
                loss_recon_nll,
                loss_recon_ce,
            ) = self.loss_reconstruct(
                recon_x,
                data_x_recon,
                non_missing_x_recon,
                output_log_var=recon_x_log_var,
            )
            # only for loss tracking
            (
                loss_recon_pred,
                loss_recon_stacked_pred,
                loss_recon_nll_pred,
                loss_recon_ce_pred,
            ) = self.loss_reconstruct(
                recon_x[~indices_recon],
                data_x_recon[~indices_recon],
                non_missing_x_recon[~indices_recon],
                output_log_var=recon_x_log_var[~indices_recon],
            )
        else:
            # forward mean
            (
                loss_recon,
                loss_recon_stacked,
                loss_recon_nll,
                loss_recon_ce,
            ) = self.loss_reconstruct(
                recon_m,
                data_x_recon,
                non_missing_x_recon,
                output_log_var=recon_m_log_var,
            )
            (
                loss_recon_pred,
                loss_recon_stacked_pred,
                loss_recon_nll_pred,
                loss_recon_ce_pred,
            ) = self.loss_reconstruct(
                recon_m[~indices_recon],
                data_x_recon[~indices_recon],
                non_missing_x_recon[~indices_recon],
                output_log_var=recon_m_log_var[~indices_recon],
            )

        if not self.sample_z:
            loss_kld = torch.zeros(1, device=self.device)
            recon_x = recon_m
            recon_x_log_var = recon_m_log_var
            z = torch.zeros_like(mu, device=self.device)
            kwargs_output = {}
        else:
            if self.prior_latent is not None:
                loss_kld, kwargs_output = self.loss_kld_with_gaussian_prior(
                    mu_p, log_var_p, mu, log_var
                )
            else:
                loss_kld, kwargs_output = self.loss_kld(mu, log_var, **kwargs)

        # different weights per classifier

        loss_class_weighted = torch.zeros(len(self.classifiers), device=self.device)
        loss_class = torch.zeros(len(self.classifiers), device=self.device)
        loss_class_weighted_pred = torch.zeros(
            len(self.classifiers), device=self.device
        )
        loss_class_pred = torch.zeros(len(self.classifiers), device=self.device)
        for i, classifier in enumerate(self.classifiers):
            if self.sample_z:
                loss_class[i] = self.loss_classifier(
                    y_recon_out[indices_recon, :][:, classifier.y_dims],
                    data_y_recon[indices_recon, :][:, classifier.y_dims],
                    non_missing_y_recon[indices_recon, :][:, classifier.y_dims],
                    [self.splits_y0[elem] for elem in classifier.y_indices],
                    [self.kinds_y0[elem] for elem in classifier.y_indices],
                    [self.weights_y0[elem] for elem in classifier.y_indices],
                )[0]
            else:
                loss_class[i] = self.loss_classifier(
                    y_recon_out_m[indices_recon, :][:, classifier.y_dims],
                    data_y_recon[indices_recon, :][:, classifier.y_dims],
                    non_missing_y_recon[indices_recon, :][:, classifier.y_dims],
                    [self.splits_y0[elem] for elem in classifier.y_indices],
                    [self.kinds_y0[elem] for elem in classifier.y_indices],
                    [self.weights_y0[elem] for elem in classifier.y_indices],
                )[0]
            loss_class_weighted[i] = self.w_class[classifier.name] * loss_class[i]
            if self.sample_z:
                loss_class_pred[i] = self.loss_classifier(
                    y_recon_out[~indices_recon, :][:, classifier.y_dims],
                    data_y_recon[~indices_recon, :][:, classifier.y_dims],
                    non_missing_y_recon[~indices_recon, :][:, classifier.y_dims],
                    [self.splits_y0[elem] for elem in classifier.y_indices],
                    [self.kinds_y0[elem] for elem in classifier.y_indices],
                    [self.weights_y0[elem] for elem in classifier.y_indices],
                )[0]
            else:
                loss_class_pred[i] = self.loss_classifier(
                    y_recon_out_m[~indices_recon, :][:, classifier.y_dims],
                    data_y_recon[~indices_recon, :][:, classifier.y_dims],
                    non_missing_y_recon[~indices_recon, :][:, classifier.y_dims],
                    [self.splits_y0[elem] for elem in classifier.y_indices],
                    [self.kinds_y0[elem] for elem in classifier.y_indices],
                    [self.weights_y0[elem] for elem in classifier.y_indices],
                )[0]
            loss_class_weighted_pred[i] = (
                self.w_class_pred[classifier.name] * loss_class_pred[i]
            )

        loss_recon_stacked_mean = loss_recon_stacked.mean()
        loss = (
            self.w_recon * loss_recon_stacked_mean
            + torch.sum(loss_class_weighted)
            + torch.sum(loss_class_weighted_pred)
        )

        if self.sample_z:
            loss += self.beta * loss_kld

        loss_pred_unw = loss_recon_pred + torch.sum(loss_class_pred)
        loss_pred = self.w_recon_pred * loss_recon_pred + torch.sum(
            loss_class_weighted_pred
        )

        losses_unweighted = torch.cat(
            [
                loss_recon.reshape(-1, 1),
                loss_kld.reshape(-1, 1),
                loss_class.reshape(-1, 1),
            ]
        )
        losses_weighted = torch.cat(
            [
                self.w_recon * loss_recon.reshape(-1, 1),
                self.beta * loss_kld.reshape(-1, 1),
                loss_class_weighted.reshape(-1, 1),
            ],
        )
        losses_unweighted_pred = torch.cat(
            [
                loss_recon_pred.reshape(-1, 1),
                loss_class_pred.reshape(-1, 1),
            ]
        )
        losses_weighted_pred = torch.cat(
            [
                self.w_recon_pred * loss_recon_pred.reshape(-1, 1),
                loss_class_weighted_pred.reshape(-1, 1),
            ],
        )

        ################################
        ### 6) OUTPUT
        ################################
        # keep track of objective loss we use in CV
        loss_cv = loss.clone()
        output = ModelOutput(
            loss_recon=self.w_recon * loss_recon,
            loss_kld=self.beta * loss_kld,
            loss_class=torch.sum(loss_class_weighted),
            losses_unweighted=losses_unweighted,
            losses_weighted=losses_weighted,
            losses_weighted_pred=losses_weighted_pred,
            losses_unweighted_pred=losses_unweighted_pred,
            loss_recon_stacked=loss_recon_stacked,
            loss_recon_ce=loss_recon_ce,
            loss_recon_nll=loss_recon_nll,
            loss_recon_nll_pred=loss_recon_nll_pred,
            loss_recon_ce_pred=loss_recon_ce_pred,
            loss=loss,
            loss_cv=loss_cv,
            loss_pred=loss_pred,
            loss_pred_unw=loss_pred_unw,
            loss_recon_stacked_pred=loss_recon_stacked_pred,
            recon_x=recon_x,
            recon_m=recon_m,
            recon_x_log_var=recon_x_log_var,
            recon_m_log_var=recon_m_log_var,
            z=z,
            z_recon=z[indices_recon],
            z_pred=z[~indices_recon],
            mu=mu,
            mu_pred=mu[~indices_recon],
            log_var_pred=log_var[~indices_recon],
            y_out_pred=y_recon_out[~indices_recon],
            y_out_m_pred=y_recon_out_m[~indices_recon],
            log_var=log_var,
            y_out_rec=y_recon_out,
            y_out_m_rec=y_recon_out_m,
            indices_recon=indices_recon,
            delta_t=delta_t,
        )

        # update output with subclass specific values
        for key, values in kwargs_output.items():
            output[key] = values

        return output

    def sample(self, mu, log_var, *args):
        # sample once from N(mu, var)
        # z: (N_patients x T_i) x D
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z, {}

    def loss_kld(self, mu, log_var, *args, **kwargs):
        # v = 1.
        # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        v = self.latent_prior_noise_var
        KLD = -0.5 * torch.sum(
            1 + log_var - math.log(v) - mu.pow(2) / v - log_var.exp() / v, dim=-1
        )
        return KLD.mean(dim=0), {}

    def loss_kld_with_gaussian_prior(self, mu_p, log_var_p, mu_q, log_var_q):
        """_summary_

        Args:
            mu_p (_type_): prior mean
            log_var_p (_type_): prior log variance
            mu_q (_type_): posterior mean
            log_var_q (_type_): posterior log variance

        """
        logDet = log_var_p - log_var_q
        trace = torch.exp(-logDet)
        square = (mu_q - mu_p).pow(2) / log_var_p.exp()
        KLD = -0.5 * torch.sum((1 - trace - logDet - square), dim=-1)

        return KLD.mean(dim=0), {}

    def reshape_input_for_lstm(self, data_x, data_s, times, splits):
        # concatenate with time

        data_x = torch.cat([data_x, times, data_s], axis=1)
        data_x_splitted = torch.split(data_x, splits, dim=0)
        data_x_padded = torch.nn.utils.rnn.pad_sequence(
            data_x_splitted, batch_first=True
        )
        data_x_padded = torch.nn.utils.rnn.pack_padded_sequence(
            data_x_padded, batch_first=True, lengths=splits, enforce_sorted=False
        )
        return data_x_padded

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        model_weights = cls._load_model_weights_from_folder(dir_path)

        if not model_config.uses_default_encoder:
            encoder = cls._load_custom_encoder_from_folder(dir_path)

        else:
            encoder = None

        if not model_config.uses_default_decoder:
            decoder = cls._load_custom_decoder_from_folder(dir_path)

        else:
            decoder = None
        my_classifiers = cls._load_custom_modules_from_folder(
            dir_path, "classifiers.pkl"
        )
        my_mu_predictor = cls._load_custom_modules_from_folder(
            dir_path, "mu_predictor.pkl"
        )
        my_var_predictor = cls._load_custom_modules_from_folder(
            dir_path, "var_predictor.pkl"
        )

        prior_latent = cls._load_custom_modules_from_folder(
            dir_path, "prior_latent.pkl"
        )
        model = cls(
            model_config,
            encoder=encoder,
            decoder=decoder,
            classifiers=my_classifiers,
            mu_predictor=my_mu_predictor,
            var_predictor=my_var_predictor,
            prior_latent=prior_latent,
        )
        model.load_state_dict(model_weights)

        return model

    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                cloudpickle.dump(self.encoder, fp)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                cloudpickle.dump(self.decoder, fp)

        for classif in self.classifiers:
            cloudpickle.register_pickle_by_value(inspect.getmodule(classif))
        with open(os.path.join(dir_path, "mu_predictor.pkl"), "wb") as fp:
            cloudpickle.dump(self.mu_predictor, fp)
        with open(os.path.join(dir_path, "var_predictor.pkl"), "wb") as fp:
            cloudpickle.dump(self.var_predictor, fp)

        with open(os.path.join(dir_path, "classifiers.pkl"), "wb") as fp:
            cloudpickle.dump(self.classifiers, fp)
        with open(os.path.join(dir_path, "prior_latent.pkl"), "wb") as fp:
            cloudpickle.register_pickle_by_value(inspect.getmodule(self.prior_latent))
            cloudpickle.dump(self.prior_latent, fp)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))
