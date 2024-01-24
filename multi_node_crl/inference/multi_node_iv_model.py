"""
Simon Bing, TU Berlin
2024
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

from multi_node_crl.inference.models import MultiNodeIV
from multi_node_crl.utils import get_device


class MultiNodeIVModel(object):
    def __init__(self, seed, z_dim, epochs, batch_size, lr):
        self.seed = seed

        self.z_dim = z_dim
        self.model = self._build_model()

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.optimizer = self._get_optimizer()

        self.device = get_device()

    def _build_model(self):
        return MultiNodeIV(seed=self.seed, z_hat_dim=self.z_dim, z_dim=self.z_dim)

    def _get_optimizer(self):
        return torch.optim.AdamW(params=self.model.parameters(), lr=self.lr)

    def train_model(self, Z_hat_train, Z_hat_val):
        """
        :param Z_hat: input data    shape : [n, n_envs, d]
        :return:
        """
        # Move model to correct device
        self.model.to(self.device)

        # Build dataloaders
        train_dataloader = DataLoader(
            TensorDataset(torch.tensor(Z_hat_train, dtype=torch.float32)),
            self.batch_size, shuffle=True)
        val_dataloader = DataLoader(
            TensorDataset(torch.tensor(Z_hat_val, dtype=torch.float32)),
            self.batch_size, shuffle=False)

        # Train loop
        # Preparation
        torch.autograd.set_detect_anomaly(True)
        best_val_loss = np.inf

        for epoch in range(self.epochs):
            # Batch training
            self.model.train()

            train_loss_tot = 0
            train_loss_constr = 0
            train_loss_var = 0
            train_loss_norm = 0
            train_loss_diag = 0

            for train_batch in train_dataloader:
                x_train = train_batch[0]
                x_train = x_train.to(self.device)
                # Zero gradients
                self.optimizer.zero_grad()
                # Forward pass
                self.model(x_train)
                # Get loss
                var_loss, constr_loss, diag_loss, norm_loss = self.model.var_loss, self.model.constr_loss, self.model.diag_loss, self.model.norm_loss
                loss_tot = var_loss + 5 * norm_loss - constr_loss + 10 * diag_loss
                # Backward pass
                loss_tot.backward()
                # Update weights
                self.optimizer.step()

                # Sum losses
                train_loss_tot += loss_tot.item()
                train_loss_constr += constr_loss.item()
                train_loss_var += var_loss.item()
                train_loss_diag += diag_loss.item()
                train_loss_norm += norm_loss.item()
            # Log (normalized) losses
            wandb.log(
                {'train_loss_tot': train_loss_tot / len(train_dataloader),
                 'train_loss_constr': train_loss_constr / len(train_dataloader),
                 'train_loss_diag': train_loss_diag / len(train_dataloader),
                 'train_loss_var': train_loss_var / len(train_dataloader),
                 'train_loss_norm': train_loss_norm / len(train_dataloader),
                 'epoch': epoch})

    def transform(self, Z_hat):
        """
        Transform input data according to learned tf.
        :param Z_hat:
        :return:
        """
        # Get dataloader
        dataloader = DataLoader(
            TensorDataset(torch.tensor(Z_hat, dtype=torch.float32)),
            self.batch_size, shuffle=False)

        with torch.no_grad():
            z_tf_list = []
            for val_batch in dataloader:
                x_val = val_batch[0]
                x_val = x_val.to(self.device)
                # Apply learned transform
                z_val = self.model.mlp_z_hat_z(x_val)
                z_tf_list.append(z_val)
            Z_tf = torch.cat(z_tf_list).cpu().detach().numpy()

        return Z_tf
