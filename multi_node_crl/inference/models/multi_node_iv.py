"""
Simon Bing, TU Berlin
2024
"""
import torch
from torch import nn


class MultiNodeIV(nn.Module):
    def __init__(self, seed, z_hat_dim=3, z_dim=3):
        """

        :param seed:
        :param z_hat_dim:
        :param z_dim:
        """
        super().__init__()

        # Random seed
        self.seed = seed
        torch.manual_seed(seed=self.seed)
        # General params
        self.z_hat_dim = z_hat_dim
        self.z_dim = z_dim

        self.var_loss = None
        self.constr_loss = None
        self.diag_loss = None
        self.norm_loss = None

        self._build()

    def _build(self):
        """
        Build model network.
        """
        self.mlp_z_hat_z = nn.Linear(self.z_hat_dim, self.z_dim, bias=False)

    def _get_loss(self, z):
        """
        Compute the variance norm loss per environment
        :param z: [batch_size, n_envs, d]
        """
        z_var = torch.var(z, dim=0) # shape = [n_envs, d]
        steep = 1000
        shift = 0
        var_loss = torch.sigmoid(steep * (z_var - shift)).sum()

        # Column wise loss to enforce our constraint
        z_var_per_env_1 = torch.sum(z_var, dim=1) # shape = [n_envs]
        constr_loss_1 = torch.sigmoid(2000 * (z_var_per_env_1 - shift)).sum()

        z_var_per_env_2 = torch.sum(z_var, dim=0)  # shape = [d]
        constr_loss_2 = torch.sigmoid(2000 * (z_var_per_env_2 - shift)).sum()
        constr_loss = constr_loss_1 + constr_loss_2

        # Norm loss on the weights of the linear layer
        norm_loss = torch.square(torch.norm(self.mlp_z_hat_z.weight) - 1)

        # diagonal group norm
        diags = []
        for i in range(self.z_hat_dim):
            diags.append(torch.diagonal(torch.cat((z_var, z_var), dim=1), offset=i))

        diag_loss_list = [torch.norm(diag) for diag in diags]
        diag_loss = torch.stack(diag_loss_list).sum()

        return var_loss, constr_loss, diag_loss, norm_loss

    def forward(self, x):
        z = self.mlp_z_hat_z(x)

        self.var_loss, self.constr_loss, self.diag_loss, self.norm_loss = self._get_loss(z)
