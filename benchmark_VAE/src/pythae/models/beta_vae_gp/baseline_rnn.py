from ..base.base_model import BaseAE
from ...data.datasets import MissingDataset
import torch
import torch.nn.functional as F
from ..base.base_utils import ModelOutput


class RNNMLP(BaseAE):
    def __init__(self, model_config, encoder, decoder, decoder_y):
        super().__init__(model_config, encoder, decoder)
        self.model_name = "RNNMLP"
        self.set_encoder(encoder)
        self.decoder_y = decoder_y
        self.predict_y = model_config.predict_y

        self.latent_dim = model_config.latent_dim
        self.missing_loss = model_config.missing_loss
        self.model_config = model_config

        self.splits_x0 = model_config.splits_x0
        self.kinds_x0 = model_config.kinds_x0

        self.splits_y0 = model_config.splits_y0
        self.kinds_y0 = model_config.kinds_y0

        self.to_reconstruct_x = model_config.to_reconstruct_x
        self.to_reconstruct_y = model_config.to_reconstruct_y

        return

    def forward(self, inputs: MissingDataset, **kwargs):
        ################################
        ### 1) INPUT
        ################################

        # input data (N_patients x T_i) x L
        data_x = inputs["data_x"]

        # static data
        data_s = inputs["data_s"]
        # labels
        data_y = inputs["data_y"]  # N_patients x n_class
        # non-missing entries in the input/output data
        if self.missing_loss:
            non_missing_x = 1 - inputs["missing_x"] * 1.0
            non_missing_y = 1 - inputs["missing_y"] * 1.0  # N_patients x n_class

        else:
            non_missing_x = 1
            non_missing_y = 1

        splits = inputs["splits"]  # N_patients
        # times = inputs["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1
        times = inputs["data_t"][:, 0].reshape(-1, 1)  # N_patients x 1

        # data augmentation
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
        ################################
        ### 2) ENCODER
        ################################
        # prepare input for lstm encoder
        data_input_padded = self.reshape_input_for_lstm(data_x, data_s, times, splits)
        hidden_state = self.encoder(data_input_padded, times, times).embedding
        data_x_hat = self.decoder(
            torch.cat([hidden_state, times_recon, data_s_recon], dim=1)
        ).reconstruction
        data_y_hat = self.decoder_y(
            torch.cat([hidden_state, times_recon], dim=1)
        ).reconstruction
        # discard indices for reconstruction (ie only predict) [~indices_recon]

        (
            loss_recon,
            loss_recon_stacked,
            loss_recon_nll,
            loss_recon_ce,
        ) = self.loss_reconstruct(
            data_x_hat, data_x_recon, non_missing_x_recon, self.splits_x0, self.kinds_x0
        )
        loss_x = loss_recon_stacked.mean()
        loss_recon_y, loss_recon_stacked_y, _, _ = self.loss_reconstruct(
            data_y_hat, data_y_recon, non_missing_y_recon, self.splits_y0, self.kinds_y0
        )
        loss_y = loss_recon_stacked_y.mean()
        loss = loss_x
        if self.predict_y:
            loss += loss_y
        output = ModelOutput(
            loss=loss,
            recon_x=data_x_hat,
            recon_y=data_y_hat,
            loss_x=loss_x,
            loss_y=loss_y,
            loss_recon_ce=loss_recon_ce,
            loss_recon_nll=loss_recon_nll,
        )
        return output

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

    def loss_multioutput(
        self,
        output,
        target,
        non_missing,
        splits0,
        kinds0,
        to_reconstruct=[],
    ):
        """
        general multi-output loss with missing data
        """
        # if to_reconstruct is empty list, populate with true for all
        if len(to_reconstruct) == 0:
            raise ValueError(
                "to_reconstruct must be a list of tuples (var_name, var_index, bool)"
            )

        # split the inputs into list of each dimension/variables
        # (i.e. for categorical variables we get a tensor N x n_categories
        # and for binary and conitionsou N x 1)
        output_splitted = torch.split(output, splits0, dim=1)
        target_splitted = torch.split(target, splits0, dim=1)
        non_missing_splitted = torch.split(non_missing, splits0, dim=1)

        # loop over the output dimensions
        loss_state = []
        # save the cont and CE losses separately
        nll_loss_state = []
        ce_loss_state = []
        for i, out in enumerate(output_splitted):
            nll_loss = None
            ce_loss = None
            if (kinds0[i] == "continuous") or (kinds0[i] == "ordinal"):

                dimension_loss = F.mse_loss(out, target_splitted[i], reduction="none")
                nll_loss = dimension_loss

            elif kinds0[i] == "categorical":
                if to_reconstruct[i][2]:
                    dimension_loss = F.cross_entropy(
                        out, target_splitted[i], reduction="none"
                    ).unsqueeze(1)
                    ce_loss = dimension_loss

                else:
                    dimension_loss = torch.zeros_like(out, device=self.device)

                # dimension_loss = F.cross_entropy(
                #     out, target_splitted[i], reduction="none"
                # ).unsqueeze(1)

            elif kinds0[i] == "binary":
                if to_reconstruct[i][2]:
                    dimension_loss = F.binary_cross_entropy(
                        torch.sigmoid(out), target_splitted[i], reduction="none"
                    )
                    ce_loss = dimension_loss

                else:
                    dimension_loss = torch.zeros_like(out, device=self.device)

            else:
                print("loss not implemented")

            # take only the contribution where we have observations
            dimension_loss = dimension_loss[non_missing_splitted[i][:, 0] == 1.0, :]

            # average over all observed samples
            dimension_loss = dimension_loss.mean()  # dim=0)
            if ce_loss is not None:
                ce_loss = ce_loss[non_missing_splitted[i][:, 0] == 1.0, :]
                ce_loss = ce_loss.mean()
                if ce_loss.isnan():
                    ce_loss = torch.zeros_like(ce_loss)
                ce_loss_state.append(ce_loss)
            if nll_loss is not None:
                nll_loss = nll_loss[non_missing_splitted[i][:, 0] == 1.0, :]
                nll_loss = nll_loss.mean()
                if nll_loss.isnan():
                    nll_loss = torch.zeros_like(nll_loss)
                nll_loss_state.append(nll_loss)
            if (
                dimension_loss.isnan()
            ):  # if we have no observation, then the mean produces nan!
                dimension_loss = torch.zeros_like(dimension_loss)

            loss_state.append(dimension_loss)

        # stack it together and sum all dimensions up
        loss_state_stacked = torch.stack(loss_state, dim=0)
        nll_loss = torch.zeros(1).to(self.device)
        if len(nll_loss_state) > 0:
            nll_loss_state_stacked = torch.stack(nll_loss_state, dim=0)
            nll_loss = nll_loss_state_stacked.mean(dim=-1)
        ce_loss = torch.zeros(1).to(self.device)
        if len(ce_loss_state) > 0:
            ce_loss_state_stacked = torch.stack(ce_loss_state, dim=0)
            ce_loss = ce_loss_state_stacked.mean(dim=-1)
        # changed to mean
        loss_state = loss_state_stacked.mean(dim=-1)
        # print(ce_loss)
        return loss_state, loss_state_stacked, nll_loss, ce_loss

    def loss_reconstruct(self, recon_x, x, non_missing_x, splits0, kinds0):
        return self.loss_multioutput(
            recon_x,
            x,
            non_missing_x,
            splits0,
            kinds0,
            to_reconstruct=self.to_reconstruct_x,
        )
