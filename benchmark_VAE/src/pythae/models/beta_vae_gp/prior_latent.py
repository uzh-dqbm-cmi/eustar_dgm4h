import torch
from ..base.base_utils import ModelOutput


class PriorLatent(torch.nn.Module):
    def __init__(self, args):
        super(PriorLatent, self).__init__()
        self.input_dim = args.input_dim
        self.hidden_dims = args.hidden_dims
        self.latent_dim = args.latent_dim
        self.device = args.device
        self.dropout = torch.nn.Dropout(args.dropout)

        modules = [
            torch.nn.Linear(self.input_dim, self.hidden_dims[0]),
            torch.nn.ReLU(),
        ]
        for i in range(len(self.hidden_dims) - 1):
            modules += [
                torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
                torch.nn.BatchNorm1d(self.hidden_dims[i + 1]),
                torch.nn.ReLU(),
                self.dropout,
            ]

        self.layers = torch.nn.Sequential(*modules)

        self.mu = torch.nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.log_var = torch.nn.Linear(self.hidden_dims[-1], self.latent_dim)

    def forward(self, x):
        x = self.layers(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        return ModelOutput(prior_mu=mu, prior_log_var=log_var)
