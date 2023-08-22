from tqdm.notebook import tqdm
import torch
from torch import nn

from ..nn import BaseEncoder, BaseDecoder
from ..base.base_utils import ModelOutput


class Indep_MLP_Encoder(BaseEncoder):
    """
    independent MLP encoder D -> L
    input is each vector state of dimension D
    output is latent state of dimension L
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        hidden_dims = [args.input_dim[0] + args.cond_dim_time_input] + args.hidden_dims

        modules = [nn.Flatten()]

        # all dense layers
        for i in range(len(hidden_dims) - 1):
            modules += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                # nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
            ]

        # combine all modules
        self.layers = nn.Sequential(*modules)

        # mean and log_var in the last layer
        self.embedding = nn.Linear(hidden_dims[-1], args.latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], args.latent_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # forward input through all layers
        h1 = self.layers(x).reshape(x.shape[0], -1)

        # create model output with mean and log_var
        output = ModelOutput(
            embedding=self.embedding(h1), log_covariance=self.log_var(h1)
        )

        return output


class LSTM_Encoder(BaseEncoder):
    """
    LSTM encoder T x D -> T x L
    input is vector state of dimension T x D
    output is latent state of dimension T x L
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.device = args.device
        self.input_size = args.input_dim + args.cond_dim_time_input
        self.hidden_size = args.lstm_hidden_size
        self.num_lstm_layers = args.num_lstm_layers
        self.dropout_layer = nn.Dropout(args.dropout)
        self.linear_hidden_dims = [
            self.hidden_size * 2 + args.cond_dim_time_input + 1
        ] + args.hidden_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.predict = args.predict

        modules = []
        # hidden_dims = [args.input_dim[0] + args.cond_dim_time_input] + args.hidden_dims

        modules = [nn.Flatten()]
        for i in range(len(self.linear_hidden_dims) - 1):
            modules += [
                nn.Linear(self.linear_hidden_dims[i], self.linear_hidden_dims[i + 1]),
                nn.ReLU(),
                self.dropout_layer,
            ]
        self.layers = nn.Sequential(*modules)
        # mean and log_var in the last layer
        self.embedding = nn.Linear(self.linear_hidden_dims[-1], args.latent_dim)
        self.log_var = nn.Linear(self.linear_hidden_dims[-1], args.latent_dim)

    def forward(
        self, x: torch.Tensor, absolute_time: torch.Tensor, time_of_pred: torch.Tensor
    ) -> ModelOutput:
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_lstm_layers * 2,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_lstm_layers * 2,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        lstm_out = torch.cat([t[:length, :] for t, length in zip(lstm_out, lengths)])
        if self.predict:
            initial_state = torch.cat([h0.mean(dim=0), h0.mean(dim=0)], dim=1)

            absolute_time_splitted = torch.split(absolute_time, list(lengths))
            time_of_pred_splitted = torch.split(time_of_pred, list(lengths))
            lstm_out_splitted = torch.split(lstm_out, list(lengths))
            # data augmentation: e.g. fot a sequence [x1, x2, x3] we create [[x1, x1, x1], [x1, x2, x2], [x1, x2, x3]] (flattened)
            # for the times absolute times: [t1, t2, t3] -> [[t1, t1, t1], [t1, t2, t2], [t1, t2, t3]

            absolute_time = torch.cat(
                [
                    torch.cat(
                        [
                            absolute_time_splitted[pat][0].repeat(
                                len(absolute_time_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            absolute_time_splitted[pat][:index, :],
                                            absolute_time_splitted[pat][
                                                index, :
                                            ].repeat(
                                                len(absolute_time_splitted[pat])
                                                - index,
                                                1,
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(absolute_time_splitted[pat]))
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(absolute_time_splitted))
                ]
            )
            lstm_out = torch.cat(
                [
                    torch.cat(
                        [
                            initial_state[pat].repeat(
                                len(absolute_time_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            lstm_out_splitted[pat][:index, :],
                                            lstm_out_splitted[pat][index, :].repeat(
                                                len(lstm_out_splitted[pat]) - index, 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(lstm_out_splitted[pat]))
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(lstm_out_splitted))
                ]
            )
            time_of_pred = torch.cat(
                [
                    time_of_pred_splitted[pat].repeat(
                        len(time_of_pred_splitted[pat]) + 1, 1
                    )
                    for pat in range(len(time_of_pred_splitted))
                ]
            )

        delta_t = time_of_pred - absolute_time
        output = self.layers(torch.cat([lstm_out, absolute_time, delta_t], dim=1))
        # output = self.layers(torch.cat([hn[-1], time], dim=1))
        # create model output with mean and log_var
        output = ModelOutput(
            embedding=self.embedding(output),
            log_covariance=self.log_var(output),
            delta_t=delta_t,
        )

        return output


class LSTM_Retrodiction_Encoder(BaseEncoder):
    """
    LSTM encoder T x D -> T x L
    input is vector state of dimension T x D
    output is latent state of dimension T x L
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        self.device = args.device
        self.input_size = args.input_dim + args.cond_dim_time_input
        self.hidden_size = args.lstm_hidden_size
        self.num_lstm_layers = args.num_lstm_layers
        self.dropout_layer = nn.Dropout(args.dropout)
        self.linear_hidden_dims = [
            self.hidden_size + args.cond_dim_time_input + 1
        ] + args.hidden_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        self.predict = args.predict

        modules = []
        # hidden_dims = [args.input_dim[0] + args.cond_dim_time_input] + args.hidden_dims

        modules = [nn.Flatten()]
        for i in range(len(self.linear_hidden_dims) - 1):
            modules += [
                nn.Linear(self.linear_hidden_dims[i], self.linear_hidden_dims[i + 1]),
                nn.BatchNorm1d(self.linear_hidden_dims[i + 1]),
                nn.ReLU(),
                self.dropout_layer,
            ]
        self.layers = nn.Sequential(*modules)
        # mean and log_var in the last layer
        self.embedding = nn.Linear(self.linear_hidden_dims[-1], args.latent_dim)
        self.log_var = nn.Linear(self.linear_hidden_dims[-1], args.latent_dim)

    def forward(
        self, x: torch.Tensor, absolute_time: torch.Tensor, time_of_pred: torch.Tensor
    ) -> ModelOutput:
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out, lengths = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True
        )
        lstm_out = torch.cat([t[:length, :] for t, length in zip(lstm_out, lengths)])
        if self.predict:
            initial_state = h0.mean(dim=0)

            absolute_time_splitted = torch.split(absolute_time, list(lengths))
            time_of_pred_splitted = torch.split(time_of_pred, list(lengths))
            lstm_out_splitted = torch.split(lstm_out, list(lengths))
            # data augmentation: e.g. fot a sequence [x1, x2, x3] we create [[x1, x1, x1], [x1, x2, x2], [x1, x2, x3]] (flattened)
            # for the times absolute times: [t1, t2, t3] -> [[t1, t1, t1], [t1, t2, t2], [t1, t2, t3]

            # lstm_out = torch.cat(
            #     [
            #         torch.cat(
            #             [
            #                 initial_state[pat].repeat(
            #                     len(absolute_time_splitted[pat]), 1
            #                 ),
            #                 torch.cat(
            #                     [
            #                         torch.cat(
            #                             [
            #                                 lstm_out_splitted[pat][:index, :],
            #                                 lstm_out_splitted[pat][index, :].repeat(
            #                                     len(lstm_out_splitted[pat]) - index, 1
            #                                 ),
            #                             ],
            #                             dim=0,
            #                         )
            #                         for index in range(len(lstm_out_splitted[pat]))
            #                     ]
            #                 ),
            #             ]
            #         )
            #         for pat in range(len(lstm_out_splitted))
            #     ]
            # )

            lstm_out = torch.cat(
                [
                    torch.cat(
                        [
                            initial_state[pat].repeat(
                                len(absolute_time_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    torch.cat(
                                        [
                                            lstm_out_splitted[pat][index, :].repeat(
                                                len(lstm_out_splitted[pat]), 1
                                            ),
                                        ],
                                        dim=0,
                                    )
                                    for index in range(len(lstm_out_splitted[pat]))
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(lstm_out_splitted))
                ]
            )
            # lstm_out = torch.cat(
            #     [
            #         torch.cat(
            #             [
            #                 initial_state[pat].repeat(
            #                     len(absolute_time_splitted[pat]), 1
            #                 ),
            #                 torch.cat(
            #                     [
            #                         torch.cat(
            #                             [

            #                                 torch.mean(lstm_out_splitted[pat][:index +1, :], dim = 0).repeat(
            #                                     len(lstm_out_splitted[pat]), 1
            #                                 ),
            #                             ],
            #                             dim=0,
            #                         )
            #                         for index in range(len(lstm_out_splitted[pat]))
            #                     ]
            #                 ),
            #             ]
            #         )
            #         for pat in range(len(lstm_out_splitted))
            #     ]
            # )
            time_of_pred = torch.cat(
                [
                    time_of_pred_splitted[pat].repeat(
                        len(time_of_pred_splitted[pat]) + 1, 1
                    )
                    for pat in range(len(time_of_pred_splitted))
                ]
            )
            # absolute_time = torch.cat(
            #     [torch.cat([

            #         elem.repeat(
            #             len(time_of_pred_splitted[pat]) + 1, 1
            #         ) for elem in absolute_time_splitted[pat]])
            #         for pat in range(len(time_of_pred_splitted))
            #     ]
            # )
            absolute_time = torch.cat(
                [
                    torch.cat(
                        [
                            time_of_pred_splitted[pat][0].repeat(
                                len(time_of_pred_splitted[pat]), 1
                            ),
                            torch.cat(
                                [
                                    elem.repeat(len(time_of_pred_splitted[pat]), 1)
                                    for elem in time_of_pred_splitted[pat]
                                ]
                            ),
                        ]
                    )
                    for pat in range(len(time_of_pred_splitted))
                ]
            )

        delta_t = time_of_pred - absolute_time
        output = self.layers(torch.cat([lstm_out, absolute_time, delta_t], dim=1))
        # output = self.layers(torch.cat([hn[-1], time], dim=1))
        # create model output with mean and log_var
        output = ModelOutput(
            embedding=self.embedding(output),
            log_covariance=self.log_var(output),
            delta_t=delta_t,
        )

        return output


class LSTM_Decoder(BaseDecoder):
    """
    LSTM encoder T x D -> T x L
    input is vector state of dimension T x D
    output is latent state of dimension T x L
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)

        self.device = args.device
        self.input_size = (
            args.latent_dim + args.cond_dim_time_latent + args.cond_dim_static_latent
        )
        self.hidden_size = args.lstm_hidden_size
        self.num_lstm_layers = args.num_lstm_layers
        # self.linear_hidden_dims = [self.hidden_size + 1] + args.hidden_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )

        modules = []
        # hidden_dims = [args.input_dim[0] + args.cond_dim_time_input] + args.hidden_dims
        hidden_dims_reverse = [
            args.lstm_hidden_size + args.cond_dim_missing_latent
        ] + args.hidden_dims[::-1]
        modules = []
        for i in range(len(hidden_dims_reverse) - 1):
            if i == 0:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    nn.BatchNorm1d(hidden_dims_reverse[i + 1]),
                    nn.ReLU(),
                ]
            else:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    # nn.BatchNorm1d(hidden_dims_reverse[i+1]),
                    nn.ReLU(),
                ]
        self.layers = nn.Sequential(*modules)
        # mean and log_var in the last layer
        self.embedding = nn.Linear(hidden_dims_reverse[-1], args.output_dim)
        self.log_var = nn.Linear(hidden_dims_reverse[-1], args.output_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.cat(
            [t[:length, :] for t, length in zip(lstm_out[0], lstm_out[1])]
        )
        # output = self.layers(torch.cat([lstm_out, absolute_time], dim=1))
        output = self.layers(lstm_out)
        # output = self.layers(torch.cat([hn[-1], time], dim=1))
        # create model output with mean and log_var
        output = ModelOutput(
            reconstruction=self.embedding(output),
            reconstruction_log_var=self.log_var(output),
        )

        return output


class LSTM_Retrodiction_Decoder(BaseDecoder):
    """
    LSTM encoder T x D -> T x L
    input is vector state of dimension T x D
    output is latent state of dimension T x L
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseDecoder.__init__(self)

        self.device = args.device
        self.input_size = (
            args.latent_dim + args.cond_dim_time_latent + args.cond_dim_static_latent
        )
        self.hidden_size = args.lstm_hidden_size
        self.num_lstm_layers = args.num_lstm_layers
        # self.linear_hidden_dims = [self.hidden_size + 1] + args.hidden_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )

        modules = []
        # hidden_dims = [args.input_dim[0] + args.cond_dim_time_input] + args.hidden_dims
        hidden_dims_reverse = [
            args.lstm_hidden_size + args.cond_dim_missing_latent
        ] + args.hidden_dims[::-1]
        modules = []
        for i in range(len(hidden_dims_reverse) - 1):
            if i == 0:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    nn.BatchNorm1d(hidden_dims_reverse[i + 1]),
                    nn.ReLU(),
                ]
            else:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    # nn.BatchNorm1d(hidden_dims_reverse[i+1]),
                    nn.ReLU(),
                ]
        self.layers = nn.Sequential(*modules)
        # mean and log_var in the last layer
        self.embedding = nn.Linear(hidden_dims_reverse[-1], args.output_dim)
        self.log_var = nn.Linear(hidden_dims_reverse[-1], args.output_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = torch.cat(
            [t[:length, :] for t, length in zip(lstm_out[0], lstm_out[1])]
        )
        # output = self.layers(torch.cat([lstm_out, absolute_time], dim=1))
        output = self.layers(lstm_out)
        # output = self.layers(torch.cat([hn[-1], time], dim=1))
        # create model output with mean and log_var
        output = ModelOutput(
            reconstruction=self.embedding(output),
            reconstruction_log_var=self.log_var(output),
        )

        return output


class Indep_MLP_Decoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.fixed_variance = args.fixed_variance
        self.in_dim0 = args.input_dim
        self.out_dim = args.output_dim
        self.dropout_layer = nn.Dropout(args.dropout)
        self.device = args.device
        hidden_dims_reverse = [
            args.latent_dim
            + args.cond_dim_missing_latent
            + args.cond_dim_time_latent
            + args.cond_dim_static_latent
        ] + args.hidden_dims[::-1]
        modules = []
        for i in range(len(hidden_dims_reverse) - 1):
            if i == 0:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    nn.BatchNorm1d(hidden_dims_reverse[i + 1]),
                    nn.ReLU(),
                ]
            else:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    # nn.BatchNorm1d(hidden_dims_reverse[i+1]),
                    nn.ReLU(),
                    self.dropout_layer,
                ]

        # combine all modules
        self.layers = nn.Sequential(*modules)

        self.embedding_rec, self.log_var_rec = self.init_modules(
            args, hidden_dims_reverse
        )

        # mean and log_var in the last layer

    def forward(self, z: torch.Tensor) -> ModelOutput:
        # forward through all layers and reshape for reconstruction
        h1 = self.layers(z)

        if self.fixed_variance:
            output = ModelOutput(
                reconstruction=self.embedding_rec(h1),
                reconstruction_log_var=torch.zeros(
                    (len(z), self.out_dim), device=self.device
                ),
            )
        else:
            output = ModelOutput(
                reconstruction=self.embedding_rec(h1),
                reconstruction_log_var=self.log_var_rec(h1),
            )

        return output

    def init_modules(self, args, hidden_dims_reverse):

        hidden_dims_reverse_emb = [hidden_dims_reverse[-1]] + args.hidden_dims_emb[::-1]
        hidden_dims_reverse_log_var = [
            hidden_dims_reverse[-1]
        ] + args.hidden_dims_log_var[::-1]
        # all dense layers
        modules_emb = []
        modules_log_var = []
        for i in range(len(hidden_dims_reverse_emb) - 1):
            if i == 0:
                modules_emb += [
                    nn.Linear(
                        hidden_dims_reverse_emb[i], hidden_dims_reverse_emb[i + 1]
                    ),
                    nn.BatchNorm1d(hidden_dims_reverse_emb[i + 1]),
                    nn.ReLU(),
                ]

            else:
                modules_emb += [
                    nn.Linear(
                        hidden_dims_reverse_emb[i], hidden_dims_reverse_emb[i + 1]
                    ),
                    nn.ReLU(),
                    self.dropout_layer,
                ]
        for i in range(len(hidden_dims_reverse_log_var) - 1):
            if i == 0:
                modules_log_var += [
                    nn.Linear(
                        hidden_dims_reverse_log_var[i],
                        hidden_dims_reverse_log_var[i + 1],
                    ),
                    nn.BatchNorm1d(hidden_dims_reverse_log_var[i + 1]),
                    nn.ReLU(),
                ]

            else:
                modules_log_var += [
                    nn.Linear(
                        hidden_dims_reverse_log_var[i],
                        hidden_dims_reverse_log_var[i + 1],
                    ),
                    nn.ReLU(),
                    self.dropout_layer,
                ]

        # last layer for input reconstruction
        modules_emb.append(nn.Linear(hidden_dims_reverse_emb[-1], args.output_dim))
        modules_log_var.append(
            nn.Linear(hidden_dims_reverse_log_var[-1], args.output_dim)
        )

        return nn.Sequential(*modules_emb), nn.Sequential(*modules_log_var)


class MLP_Decoder(BaseDecoder):
    def __init__(self, args=None):
        BaseDecoder.__init__(self)
        self.device = args.device
        self.in_dim0 = args.input_dim
        self.out_dim = args.output_dim
        self.dropout_layer = nn.Dropout(args.dropout)
        hidden_dims_reverse = [
            args.latent_dim + args.cond_dim_time_latent + args.cond_dim_static_latent
        ] + args.hidden_dims[::-1]
        modules = []
        for i in range(len(hidden_dims_reverse) - 1):
            if i == 0:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    nn.BatchNorm1d(hidden_dims_reverse[i + 1]),
                    nn.ReLU(),
                ]
            else:
                modules += [
                    nn.Linear(hidden_dims_reverse[i], hidden_dims_reverse[i + 1]),
                    # nn.BatchNorm1d(hidden_dims_reverse[i+1]),
                    nn.ReLU(),
                    self.dropout_layer,
                ]
        modules.append(nn.Linear(hidden_dims_reverse[-1], args.output_dim))

        # combine all modules
        self.layers = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor) -> ModelOutput:
        # forward through all layers and reshape for reconstruction
        h1 = self.layers(z)

        output = ModelOutput(
            reconstruction=h1,
            reconstruction_log_var=torch.ones(
                (len(z), self.out_dim), device=self.device
            ),
        )

        return output


# class Guidance_Classifier(nn.Module):
#     def __init__(self, model_config):
#         """
#         dims contains [input_dim, h_dim1, ..., h_dimN, output_dim]
#         """

#         # super(Guidance_Classifier,self).__init__()
#         super().__init__()

#         dims = model_config.classifier_dims

#         modules = []

#         # all dense layers
#         for i in range(len(dims) - 2):
#             modules += [
#                 nn.Linear(dims[i], dims[i + 1]),
#                 # nn.BatchNorm1d(hidden_dims[i+1]),
#                 nn.ReLU(),
#             ]

#         # last layer always linear
#         modules.append(nn.Linear(dims[-2], dims[-1]))

#         # combine all modules
#         self.layers = nn.Sequential(*modules)

#     def forward(self, z: torch.Tensor):
#         return self.layers(z)


class Guidance_Classifier(nn.Module):
    def __init__(self, model_config):
        """
        dims contains [input_dim, h_dim1, ..., h_dimN, output_dim]
        """

        # super(Guidance_Classifier,self).__init__()
        super().__init__()
        self.type = model_config.type
        self.name = model_config.organ_name
        self.y_dims = model_config.y_dims
        self.z_dims = model_config.z_dims
        self.y_indices = model_config.y_indices

        self.input_dim = model_config.input_dim

        output_dim = model_config.output_dim

        hidden_dims = model_config.layers

        modules = [nn.Linear(self.input_dim, hidden_dims[0]), nn.ReLU()]

        # all dense layers
        for i in range(len(hidden_dims) - 1):
            modules += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                # nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
            ]

        # last layer always linear
        modules.append(nn.Linear(hidden_dims[-1], output_dim))

        # combine all modules
        self.layers = nn.Sequential(*modules)

    def forward(self, z: torch.Tensor):
        return self.layers(z)


class MLP_Predictor(BaseEncoder):
    """ """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        BaseEncoder.__init__(self)

        hidden_dims = [args.input_dim * args.n_previous_for_pred] + args.hidden_dims
        modules = [nn.Flatten()]

        # all dense layers
        for i in range(len(hidden_dims) - 1):
            modules += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                # nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
            ]

        # combine all modules
        self.layers = nn.Sequential(*modules)

        # output
        self.out = nn.Linear(hidden_dims[-1], args.output_dim)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        # forward input through all layers
        h1 = self.layers(x).reshape(x.shape[0], -1)

        # create model output with mean and log_var
        output = ModelOutput(out=self.out(h1))

        return output


class LSTM_Predictor(nn.Module):
    """
    lstm layers followed by fully connected layers
    """

    def __init__(self, args=None):  # Args is a ModelConfig instance
        nn.Module.__init__(self)
        self.device = args.device
        self.input_size = args.input_dim
        self.hidden_size = args.lstm_hidden_size
        self.num_lstm_layers = args.num_lstm_layers
        self.linear_hidden_dims = [self.hidden_size + 1] + args.hidden_dims
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )
        modules = []
        for i in range(len(self.linear_hidden_dims) - 1):
            modules += [
                nn.Linear(self.linear_hidden_dims[i], self.linear_hidden_dims[i + 1]),
                nn.ReLU(),
            ]
        self.layers = nn.Sequential(*modules)
        self.out = nn.Linear(self.linear_hidden_dims[-1], args.output_dim)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> ModelOutput:
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_lstm_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        # lstm_out = torch.stack(
        #     [t[length - 1, :] for t, length in zip(lstm_out[0], lstm_out[1])]
        # )
        # if not torch.equal(lstm_out, hn[-1]):
        #     print("lstm_out and hn[-1] are not equal")
        output = self.layers(torch.cat([hn[-1], time], dim=1))
        output = ModelOutput(out=self.out(output))
        return output


# class Indep_MLP_Decoder(BaseDecoder):
#     def __init__(self, args=None):
#         BaseDecoder.__init__(self)

#         hidden_dims_reverse_emb = [
#             args.latent_dim
#             + args.cond_dim_missing_latent
#             + args.cond_dim_time_latent
#             + args.cond_dim_static_latent
#         ] + args.hidden_dims_emb[::-1]
#         hidden_dims_reverse_log_var = [
#             args.latent_dim
#             + args.cond_dim_missing_latent
#             + args.cond_dim_time_latent
#             + args.cond_dim_static_latent
#         ] + args.hidden_dims_log_var[::-1]
#         self.splits = args.splits
#         self.hidden_dims_feat_spec = args.hidden_dims_feat_spec
#         self.in_dim0 = args.input_dim
#         self.dropout_layer = nn.Dropout(args.dropout)
#         self.device = args.device
#         # self.in_dim1 = args.input_dim[1]

#         # all dense layers
#         modules_emb = []
#         modules_log_var = []
#         for i in range(len(hidden_dims_reverse_emb) - 1):
#             if i == 0:
#                 modules_emb += [
#                     nn.Linear(hidden_dims_reverse_emb[i], hidden_dims_reverse_emb[i + 1]),
#                     nn.BatchNorm1d(hidden_dims_reverse_emb[i + 1]),
#                     nn.ReLU(),
#                 ]

#             else:
#                 modules_emb += [
#                     nn.Linear(hidden_dims_reverse_emb[i], hidden_dims_reverse_emb[i + 1]),
#                     nn.BatchNorm1d(hidden_dims_reverse_emb[i+1]),
#                     nn.ReLU(),
#                     self.dropout_layer,
#                 ]
#         for i in range(len(hidden_dims_reverse_log_var) - 1):
#             if i == 0:
#                 modules_log_var += [
#                     nn.Linear(hidden_dims_reverse_log_var[i], hidden_dims_reverse_log_var[i + 1]),
#                     nn.BatchNorm1d(hidden_dims_reverse_log_var[i + 1]),
#                     nn.ReLU(),
#                 ]

#             else:
#                 modules_log_var += [
#                     nn.Linear(hidden_dims_reverse_log_var[i], hidden_dims_reverse_log_var[i + 1]),
#                     nn.BatchNorm1d(hidden_dims_reverse_log_var[i+1]),
#                     nn.ReLU(),
#                     self.dropout_layer,
#                 ]

#         # feature specific modules
#         self.emb_modules = [[] for i in range(len(self.splits))]
#         self.log_var_modules = [[] for i in range(len(self.splits))]
#         for i, num in enumerate(self.splits):
#             self.emb_modules[i].append(nn.Linear(hidden_dims_reverse_emb[-1], self.hidden_dims_feat_spec[0]))
#             self.emb_modules[i].append(nn.BatchNorm1d(self.hidden_dims_feat_spec[0]))
#             self.emb_modules[i].append(nn.ReLU())
#             self.emb_modules[i].append(self.dropout_layer)
#             self.log_var_modules[i].append(nn.Linear(hidden_dims_reverse_log_var[-1], self.hidden_dims_feat_spec[0]))
#             self.log_var_modules[i].append(nn.BatchNorm1d(self.hidden_dims_feat_spec[0]))
#             self.log_var_modules[i].append(nn.ReLU())
#             self.log_var_modules[i].append(self.dropout_layer)
#             for j in range(len(self.hidden_dims_feat_spec) - 1):
#                 self.emb_modules[i].append(nn.Linear(self.hidden_dims_feat_spec[j], self.hidden_dims_feat_spec[j + 1]))
#                 self.emb_modules[i].append(nn.BatchNorm1d(self.hidden_dims_feat_spec[j + 1]))
#                 self.emb_modules[i].append(nn.ReLU())
#                 self.emb_modules[i].append(self.dropout_layer)
#                 self.log_var_modules[i].append(nn.Linear(self.hidden_dims_feat_spec[j], self.hidden_dims_feat_spec[j + 1]))
#                 self.log_var_modules[i].append(nn.BatchNorm1d(self.hidden_dims_feat_spec[j + 1]))
#                 self.log_var_modules[i].append(nn.ReLU())
#                 self.log_var_modules[i].append(self.dropout_layer)
#             self.emb_modules[i].append(nn.Linear(self.hidden_dims_feat_spec[-1], num))
#             self.log_var_modules[i].append(nn.Linear(self.hidden_dims_feat_spec[-1], num))
#             self.emb_modules[i] = nn.Sequential(*self.emb_modules[i]).to(self.device)
#             self.log_var_modules[i] = nn.Sequential(*self.log_var_modules[i]).to(self.device)
#         # # last layer for input reconstruction
#         # modules_emb.append(nn.Linear(hidden_dims_reverse_emb[-1], args.output_dim))
#         # modules_log_var.append(nn.Linear(hidden_dims_reverse_log_var[-1], args.output_dim))

#         # combine all modules

#         self.h1_embedding = nn.Sequential(*modules_emb)
#         self.h1_log_var = nn.Sequential(*modules_log_var)

#         # mean and log_var in the last layer

#         # self.embedding = nn.Linear(hidden_dims_reverse[-1], args.output_dim)
#         # self.log_var = nn.Linear(hidden_dims_reverse[-1], args.output_dim)

#     def forward(self, z: torch.Tensor) -> ModelOutput:
#         # forward through all layers and reshape for reconstruction
#         #h1 = self.layers(z)
#         h1_embedding = self.h1_embedding(z)
#         h1_log_var = self.h1_log_var(z)
#         embedding = torch.cat([self.emb_modules[i](h1_embedding) for i in range(len(self.splits))], dim = 1)
#         log_var = torch.cat([self.log_var_modules[i](h1_log_var) for i in range(len(self.splits))], dim = 1)
#         # model output
#         output = ModelOutput(
#             reconstruction=embedding, reconstruction_log_var=log_var
#         )

#         return output
