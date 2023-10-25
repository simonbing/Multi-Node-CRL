"""
Code to reproduce experiments.
"""
import sys
import os

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import FastICA
from absl import flags, app
import wandb

from multi_node_crl.utils import get_rand_A, compute_MCC, interventional_CRL
from multi_node_crl.data import get_scm
from multi_node_crl.inference import MultiNodeIVModel

FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 0, 'Random seed.')
# Experiment tracking
flags.DEFINE_string('run_name', None, 'Optional run name, use to name runs by seed.')
flags.DEFINE_string('exp_group', None, 'Group to gather related experiments.')
flags.DEFINE_bool('save_results', False, 'Whether or not to save eval results.')
flags.DEFINE_string('out_path', '', 'Base directory where to save results.')
# Data
flags.DEFINE_string('scm_id', 'example', 'SCM model to use.')
flags.DEFINE_integer('d', 3, 'Latent dimension of the data.')
flags.DEFINE_float('connect_prob', 0.75, 'Probability of an edge in a random graph.')
flags.DEFINE_integer('n', int(1e5), 'Number of data points to sample.')
flags.DEFINE_float('test_size', 0.25, 'Fraction of data to use for testing.')
flags.DEFINE_integer('mixing_seed', 0, 'Random seed for mixing matrix L.')
# Models
flags.DEFINE_list('models', ['ours'], 'Models to train.')
# Training
flags.DEFINE_integer('epochs', 50, 'Training epochs.')
flags.DEFINE_integer('batch_size', 1024, 'Mini-batch size.')
flags.DEFINE_float('lr', 1e-3, 'Learning rate.')

class MainApplication(object):
    def __init__(self):
        self.seed = FLAGS.seed
        self.mixing_seed = FLAGS.mixing_seed

        self.scm_id = FLAGS.scm_id
        self.d = FLAGS.d
        self.n = FLAGS.n
        self.connect_prob = FLAGS.connect_prob
        self.test_size = FLAGS.test_size

        self.models = FLAGS.models
        self.epochs = FLAGS.epochs
        self.batch_size = FLAGS.batch_size
        self.lr = FLAGS.lr

        self.out_path = FLAGS.out_path

    def _get_data(self):
        """
        Get data model and sample. Return ground truth and transformed data.

        Returns:
            Z
            Z_hat
        """
        # Get scm
        self.scm_model, interventions, _ = get_scm(scm_id=self.scm_id, seed=self.seed)

        # Sample from interventional distributions
        Z_list = [self.scm_model.intervent_sample(iv=intervention, n=self.n)
                  for intervention in interventions]
        Z = np.stack(Z_list).transpose(1, 0, 2) # shape = [n, n_envs, d]

        # Sample mixing matrix
        self.L = get_rand_A(len(self.scm_model.variables), seed=self.mixing_seed)

        # Transform ground truth data
        Z_hat_list = [np.matmul(Z_i, self.L) for Z_i in Z_list]
        Z_hat = np.stack(Z_hat_list).transpose(1, 0, 2)  # shape = [n, n_envs, d]

        # Split train and val data
        Z_hat_train, Z_hat_val, Z_gt_train, Z_gt_val = train_test_split(Z_hat,
                                                                        Z,
                                                                        test_size=self.test_size)

        return Z_hat_train, Z_hat_val, Z_gt_train, Z_gt_val

    def run(self):
        """
        Main method that contains code to run during call.
        """
        # Training prep: make folders according to config

        # Get data
        Z_hat_train, Z_hat_val, Z_gt_train, Z_gt_val = self._get_data()
        # For eval later
        Z_gt_val_re = Z_gt_val.reshape([-1, self.d], order='C')

        results_dict = {}

        # Get and train models (output their respective estimates)
        if 'ours' in self.models:
            multi_node = MultiNodeIVModel(seed=self.seed, z_dim=self.d, epochs=self.epochs,
                                          batch_size=self.batch_size, lr=self.lr)
            multi_node.train_model(Z_hat_train, Z_hat_val)
            Z_tf_multi = multi_node.transform(Z_hat_val)
            Z_tf_multi_re = Z_tf_multi.reshape([-1, self.d], order='C')

            # Eval
            mcc_multi = compute_MCC(Z_tf_multi_re, Z_gt_val_re, batch_size=5000)
            wandb.run.summary.update({'MCC_multi': np.mean(mcc_multi)})
            results_dict['MCC_multi'] = np.mean(mcc_multi)
        if 'icrl' in self.models:
            Gamma = interventional_CRL(Z_hat_train)
            # Transform data
            Z_tf_icrl = np.matmul(Z_hat_val.reshape([-1, self.d], order='C'), Gamma.T)
            mcc_icrl = compute_MCC(Z_tf_icrl, Z_gt_val_re, batch_size=5000)
            wandb.run.summary.update({'MCC_icrl': np.mean(mcc_icrl)})
            results_dict['MCC_icrl'] = np.mean(mcc_icrl)
        if 'ica' in self.models:
            # ICA gets the observational distribution, i.e. no interventions
            Z_ica = self.scm_model.sample(n=self.n)
            Z_hat_ica = np.matmul(Z_ica, self.L)
            Z_ica_train, Z_ica_val, Z_hat_ica_train, Z_hat_ica_val = train_test_split(Z_ica, Z_hat_ica,
                                                      test_size=self.test_size)
            ica_tf = FastICA()
            ica_tf.fit(Z_hat_ica_train)
            Z_tf_ica = ica_tf.transform(Z_hat_ica_val)
            mcc_ica = compute_MCC(Z_tf_ica, Z_ica_val, batch_size=5000)
            wandb.run.summary.update({'MCC_ica': np.mean(mcc_ica)})
            results_dict['MCC_ica'] = np.mean(mcc_ica)

        if FLAGS.save_results:
            save_path = os.path.join(self.out_path, F'{self.scm_id}',
                                     F'mixing_seed_{self.mixing_seed}',
                                     F'seed_{self.seed}')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            np.savez(os.path.join(save_path, 'results.npz'), **results_dict)


def main(argv):
    wandb_config = {
        'experiment_name': F'{FLAGS.scm_id}_mixing_seed_{FLAGS.mixing_seed}',
        'seed': FLAGS.seed,
        'inf_model': 'linear_var',
        'init_lr': FLAGS.lr,
        'batch_size': FLAGS.batch_size,
        'epochs': FLAGS.epochs,
        'dataset': FLAGS.scm_id,
        'gt_dim': FLAGS.d,
        'z_hat_dim': FLAGS.d,
        'eval_metric': 'mcc'
    }

    # Check if we are in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        mode = 'disabled'
    else:
        mode = 'online'

    wandb.init(project='multi_node_crl',
               config=wandb_config,
               group='linear_var',
               name= F'seed_{FLAGS.seed}',
               mode=mode)

    application = MainApplication()
    application.run()


if __name__ == '__main__':
    app.run(main)
