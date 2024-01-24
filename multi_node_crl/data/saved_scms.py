"""
Simon Bing, TU Berlin
2024
"""
import numpy as np
from scipy.stats import norm, uniform

from multi_node_crl.data.SCM.SCM_gen import SCM, CausalVar, Intervention
from multi_node_crl.utils import rand_weight_matrix


def get_scm(scm_id, seed, connect_prob=0.75):
    """
    Defines the SCM to generate data with, as well as possible interventions.

    Returns:
        SCM: SCM
            Structural causal model object defining the causal variables
            and their functional relationships.

        interventions: list of Interventions
            Intervention objects defining which interventions are applied to
            the SCM during sampling.
    """

    # Pass the index of which variable is the target. If there is none, set to
    # None. This can then also be used later for evaluation purposes.

    # Default value is None if scm doesn't specify a target
    obs_target = None

    if scm_id == 'toy':
        e1 = norm(0, 1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, 0.8)
        f2 = lambda x1, e: 0.3 * x1 ** 2 + 0.6 * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        variables = (z1, z2)
        interventions = None

    elif scm_id == 'toy_2':
        e1 = norm(0, 1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, 1)
        f2 = lambda x1, e: x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        ey = norm(0, 0.1)
        fy = lambda x1, x2, e: 0.3 * x1 + 0.8 * x2 + e
        y = CausalVar(parents=[z1, z2], noise_distr=ey, mechanism_fct=fy)

        variables = (z1, z2, y)
        obs_target = y

        interv1 = Intervention(targets=0, new_mechanism=2)
        interv2 = Intervention(targets=1, new_mechanism=2)
        # Held out intervention, only for testing
        interv3 = Intervention(targets=0, new_mechanism=500)

        # Don't need holdout intervention atm
        interventions = (interv1, interv2, interv3)

    elif scm_id == 'toy_3':
        e1 = norm(0, 1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, 1)
        f2 = lambda x1, e: x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        ey = norm(0, 0.1)
        fy = lambda x1, x2, e: 0.3 * x1 + 0.8 * x2 + e
        y = CausalVar(parents=[z1, z2], noise_distr=ey, mechanism_fct=fy)

        e3 = norm(0, 1)
        f3 = lambda x1, e: x1 + e
        z3 = CausalVar(parents=[y], noise_distr=e3, mechanism_fct=f3)

        variables = (z1, z2, y, z3)
        obs_target = y

        interv1 = Intervention(targets=0, new_mechanism=2)
        interv2 = Intervention(targets=1, new_mechanism=2)
        interv3 = Intervention(targets=3, new_mechanism=2)
        # Held out intervention, only for testing
        interv4 = Intervention(targets=3, new_mechanism=500)

        # Don't need holdout intervention atm
        interventions = (interv1, interv2, interv3, interv4)

    elif scm_id == 'example':
        # Dummy SCM with 3 nodes
        e1 = norm(0, .1)
        e1 = uniform()
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        a = 0
        e2 = norm(0, .1)
        e3 = uniform()
        f2 = lambda x1, e: a * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        b = 0
        c = 0
        e3 = norm(0, .1)
        e3 = uniform()
        f3 = lambda x1, x2, e: b * x1 + c * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        variables = (z1, z2, z3)

        # Interventions
        iv1 = Intervention(targets=[0, 1], new_mechanism=[1, 1])
        iv2 = Intervention(targets=[0, 2], new_mechanism=[1, 2])
        iv3 = Intervention(targets=[1, 2], new_mechanism=[1, 3])

        interventions = (iv1, iv2, iv3)

    elif scm_id == 'rand_d_3':
        # Randomly sampled graph with 3 nodes
        W = rand_weight_matrix(seed=seed, nodes=3, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        variables = (z1, z2, z3)
        # Interventions
        iv1 = Intervention(targets=[0, 1], new_mechanism=[1, 1])
        iv2 = Intervention(targets=[0, 2], new_mechanism=[1, 1])
        iv3 = Intervention(targets=[1, 2], new_mechanism=[1, 1])

        interventions = (iv1, iv2, iv3)

    elif scm_id == 'rand_d_6':
        # Randomly sampled graph with 6 nodes
        W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0,3] * x1 + W[1,3] * x2 + W[2,3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0,4] * x1 + W[1,4] * x2 + W[2,4] * x3 \
                                       + W[3,4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0,5] * x1 + W[1,5] * x2 \
                                           + W[2,5] * x3 + W[3,5] * x4 \
                                           + W[4,5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0, 1, 2, 3, 4], new_mechanism=[1, 1, 1, 1, 1])
        iv2 = Intervention(targets=[0, 1, 2, 3, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv3 = Intervention(targets=[0, 1, 2, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv4 = Intervention(targets=[0, 1, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv5 = Intervention(targets=[0, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv6 = Intervention(targets=[1, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6)

    elif scm_id == 'rand_d_6_cp':
        # Randomly sampled graph with 6 nodes and changing connection probability
        W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=connect_prob)
        # assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0,3] * x1 + W[1,3] * x2 + W[2,3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0,4] * x1 + W[1,4] * x2 + W[2,4] * x3 \
                                       + W[3,4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0,5] * x1 + W[1,5] * x2 \
                                           + W[2,5] * x3 + W[3,5] * x4 \
                                           + W[4,5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0, 1, 2, 3, 4], new_mechanism=[1, 1, 1, 1, 1])
        iv2 = Intervention(targets=[0, 1, 2, 3, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv3 = Intervention(targets=[0, 1, 2, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv4 = Intervention(targets=[0, 1, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv5 = Intervention(targets=[0, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv6 = Intervention(targets=[1, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6)

    elif scm_id == 'rand_d_6_atomic':
        # Randomly sampled graph with 6 nodes and ATOMIC interventions
        W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0,3] * x1 + W[1,3] * x2 + W[2,3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0,4] * x1 + W[1,4] * x2 + W[2,4] * x3 \
                                       + W[3,4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0,5] * x1 + W[1,5] * x2 \
                                           + W[2,5] * x3 + W[3,5] * x4 \
                                           + W[4,5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0], new_mechanism=[1])
        iv2 = Intervention(targets=[1], new_mechanism=[1])
        iv3 = Intervention(targets=[2], new_mechanism=[1])
        iv4 = Intervention(targets=[3], new_mechanism=[1])
        iv5 = Intervention(targets=[4], new_mechanism=[1])
        iv6 = Intervention(targets=[5], new_mechanism=[1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6)

    elif scm_id == 'rand_d_6_custom':
        # Randomly sampled graph with 6 nodes and ATOMIC interventions
        W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0,3] * x1 + W[1,3] * x2 + W[2,3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0,4] * x1 + W[1,4] * x2 + W[2,4] * x3 \
                                       + W[3,4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0,5] * x1 + W[1,5] * x2 \
                                           + W[2,5] * x3 + W[3,5] * x4 \
                                           + W[4,5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0,1,2,3,4], new_mechanism=[1,1,1,1,1])
        iv2 = Intervention(targets=[0,1,2,5], new_mechanism=[1,1,1,1])
        iv3 = Intervention(targets=[4,5], new_mechanism=[1,1])
        iv4 = Intervention(targets=[3,5], new_mechanism=[1,1])
        iv5 = Intervention(targets=[0,1], new_mechanism=[1,1])
        iv6 = Intervention(targets=[0,2], new_mechanism=[1,1])
        iv7 = Intervention(targets=[1,2], new_mechanism=[1,1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6, iv7)

    elif scm_id == 'rand_d_10':
        # Randomly sampled graph with 10 nodes
        W = rand_weight_matrix(seed=seed, nodes=10, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0,1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0,2] * x1 + W[1,2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0,3] * x1 + W[1,3] * x2 + W[2,3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0,4] * x1 + W[1,4] * x2 + W[2,4] * x3 \
                                       + W[3,4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5,
                       mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0,5] * x1 + W[1,5] * x2 \
                                           + W[2,5] * x3 + W[3,5] * x4 \
                                           + W[4,5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6,
                       mechanism_fct=f6)

        e7 = norm(0, .1)
        f7 = lambda x1, x2, x3, x4, x5, x6, e: W[0, 6] * x1 + W[1, 6] * x2 \
                                               + W[2, 6] * x3 + W[3, 6] * x4 \
                                               + W[4, 6] * x5 + W[5, 6] * x6 + e
        z7 = CausalVar(parents=[z1, z2, z3, z4, z5, z6], noise_distr=e7,
                       mechanism_fct=f7)

        e8 = norm(0, .1)
        f8 = lambda x1, x2, x3, x4, x5, x6, x7, e: W[0, 7] * x1 + W[1, 7] * x2 \
                                                   + W[2, 7] * x3 + W[3, 7] * x4 \
                                                   + W[4, 7] * x5 + W[5, 7] * x6 \
                                                   + W[6, 7] * x7 + e
        z8 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7], noise_distr=e8,
                       mechanism_fct=f8)

        e9 = norm(0, .1)
        f9 = lambda x1, x2, x3, x4, x5, x6, x7, x8, e: W[0, 8] * x1 \
                                                       + W[1, 8] * x2 \
                                                       + W[2, 8] * x3 \
                                                       + W[3, 8] * x4 \
                                                       + W[4, 8] * x5 \
                                                       + W[5, 8] * x6 \
                                                       + W[6, 8] * x7 \
                                                       + W[7, 8] * x8 + e
        z9 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8], noise_distr=e9,
                       mechanism_fct=f9)

        e10 = norm(0, .1)
        f10 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, e: W[0, 9] * x1 \
                                                            + W[1, 9] * x2 \
                                                            + W[2, 9] * x3 \
                                                            + W[3, 9] * x4 \
                                                            + W[4, 9] * x5 \
                                                            + W[5, 9] * x6 \
                                                            + W[6, 9] * x7 \
                                                            + W[7, 9] * x8 \
                                                            + W[8, 9] * x9 + e
        z10 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9],
                        noise_distr=e10, mechanism_fct=f10)

        variables = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10)
        # Interventions
        iv1 = Intervention(targets=[0, 1, 2, 3, 4, 5, 6, 7, 8], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv2 = Intervention(targets=[0, 1, 2, 3, 4, 5, 6, 7, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv3 = Intervention(targets=[0, 1, 2, 3, 4, 5, 6, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv4 = Intervention(targets=[0, 1, 2, 3, 4, 5, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv5 = Intervention(targets=[0, 1, 2, 3, 4, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv6 = Intervention(targets=[0, 1, 2, 3, 5, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv7 = Intervention(targets=[0, 1, 2, 4, 5, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv8 = Intervention(targets=[0, 1, 3, 4, 5, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv9 = Intervention(targets=[0, 2, 3, 4, 5, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        iv10 = Intervention(targets=[1, 2, 3, 4, 5, 6, 7, 8, 9], new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6, iv7, iv8, iv9, iv10)

    elif scm_id == 'nonlin1_d_6':
        # Nonlinear SCM with 6 nodes
        # W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=0.75)
        # assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: x1**2 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: x1**2 + x2**2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: x1**2 + x2**2 + x3**2 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x3, x4, e: x1**2 + x3**2 + x4**2 + e
        z5 = CausalVar(parents=[z1, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x2, x3, x4, x5, e: x2**2 + x3**2 + x4**2 + x5**2 + e
        z6 = CausalVar(parents=[z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0, 1, 2, 3, 4], new_mechanism=[1, 1, 1, 1, 1])
        iv2 = Intervention(targets=[0, 1, 2, 3, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv3 = Intervention(targets=[0, 1, 2, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv4 = Intervention(targets=[0, 1, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv5 = Intervention(targets=[0, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv6 = Intervention(targets=[1, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6)

    elif scm_id == 'nonlin2_d_6':
        # Nonlinear SCM with 6 nodes
        # W = rand_weight_matrix(seed=seed, nodes=6, connect_prob=0.75)
        # assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: np.sin(x1) + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: (x1 + x2)**0.5 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: np.log(x1**2 + x2) + x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x3, x4, e: np.cos(x1) * x3 + np.arctan(x4) + e
        z5 = CausalVar(parents=[z1, z3, z4], noise_distr=e5, mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x2, x3, x4, x5, e: x2 * x3 * np.exp(x4**2 / x5) + e
        z6 = CausalVar(parents=[z2, z3, z4, z5], noise_distr=e6, mechanism_fct=f6)

        variables = (z1, z2, z3, z4, z5, z6)
        # Interventions
        iv1 = Intervention(targets=[0, 1, 2, 3, 4], new_mechanism=[1, 1, 1, 1, 1])
        iv2 = Intervention(targets=[0, 1, 2, 3, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv3 = Intervention(targets=[0, 1, 2, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv4 = Intervention(targets=[0, 1, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv5 = Intervention(targets=[0, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])
        iv6 = Intervention(targets=[1, 2, 3, 4, 5], new_mechanism=[1, 1, 1, 1, 1])

        interventions = (iv1, iv2, iv3, iv4, iv5, iv6)

    elif scm_id == 'rand_d_30':
        # Randomly sampled graph with 30 nodes
        W = rand_weight_matrix(seed=seed, nodes=30, connect_prob=0.75)
        assert np.sum(W) != 0, F'Graph is fully unconnected with this seed ({seed})'
        e1 = norm(0, .1)
        f1 = lambda e: e
        z1 = CausalVar(parents=None, noise_distr=e1, mechanism_fct=f1)

        e2 = norm(0, .1)
        f2 = lambda x1, e: W[0, 1] * x1 + e
        z2 = CausalVar(parents=[z1], noise_distr=e2, mechanism_fct=f2)

        e3 = norm(0, .1)
        f3 = lambda x1, x2, e: W[0, 2] * x1 + W[1, 2] * x2 + e
        z3 = CausalVar(parents=[z1, z2], noise_distr=e3, mechanism_fct=f3)

        e4 = norm(0, .1)
        f4 = lambda x1, x2, x3, e: W[0, 3] * x1 + W[1, 3] * x2 + W[
            2, 3] * x3 + e
        z4 = CausalVar(parents=[z1, z2, z3], noise_distr=e4, mechanism_fct=f4)

        e5 = norm(0, .1)
        f5 = lambda x1, x2, x3, x4, e: W[0, 4] * x1 + W[1, 4] * x2 + W[
            2, 4] * x3 \
                                       + W[3, 4] * x4 + e
        z5 = CausalVar(parents=[z1, z2, z3, z4], noise_distr=e5,
                       mechanism_fct=f5)

        e6 = norm(0, .1)
        f6 = lambda x1, x2, x3, x4, x5, e: W[0, 5] * x1 + W[1, 5] * x2 \
                                           + W[2, 5] * x3 + W[3, 5] * x4 \
                                           + W[4, 5] * x5 + e
        z6 = CausalVar(parents=[z1, z2, z3, z4, z5], noise_distr=e6,
                       mechanism_fct=f6)

        e7 = norm(0, .1)
        f7 = lambda x1, x2, x3, x4, x5, x6, e: W[0, 6] * x1 + W[1, 6] * x2 \
                                               + W[2, 6] * x3 + W[3, 6] * x4 \
                                               + W[4, 6] * x5 + W[5, 6] * x6 + e
        z7 = CausalVar(parents=[z1, z2, z3, z4, z5, z6], noise_distr=e7,
                       mechanism_fct=f7)

        e8 = norm(0, .1)
        f8 = lambda x1, x2, x3, x4, x5, x6, x7, e: W[0, 7] * x1 + W[1, 7] * x2 \
                                                   + W[2, 7] * x3 + W[3, 7] * x4 \
                                                   + W[4, 7] * x5 + W[5, 7] * x6 \
                                                   + W[6, 7] * x7 + e
        z8 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7], noise_distr=e8,
                       mechanism_fct=f8)

        e9 = norm(0, .1)
        f9 = lambda x1, x2, x3, x4, x5, x6, x7, x8, e: W[0, 8] * x1 \
                                                       + W[1, 8] * x2 \
                                                       + W[2, 8] * x3 \
                                                       + W[3, 8] * x4 \
                                                       + W[4, 8] * x5 \
                                                       + W[5, 8] * x6 \
                                                       + W[6, 8] * x7 \
                                                       + W[7, 8] * x8 + e
        z9 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8], noise_distr=e9,
                       mechanism_fct=f9)

        e10 = norm(0, .1)
        f10 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, e: W[0, 9] * x1 \
                                                            + W[1, 9] * x2 \
                                                            + W[2, 9] * x3 \
                                                            + W[3, 9] * x4 \
                                                            + W[4, 9] * x5 \
                                                            + W[5, 9] * x6 \
                                                            + W[6, 9] * x7 \
                                                            + W[7, 9] * x8 \
                                                            + W[8, 9] * x9 + e
        z10 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9],
                        noise_distr=e10, mechanism_fct=f10)

        e11 = norm(0, .1)
        f11 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, e: W[0, 10] * x1 \
                                                            + W[1, 10] * x2 \
                                                            + W[2, 10] * x3 \
                                                            + W[3, 10] * x4 \
                                                            + W[4, 10] * x5 \
                                                            + W[5, 10] * x6 \
                                                            + W[6, 10] * x7 \
                                                            + W[7, 10] * x8 \
                                                            + W[8, 10] * x9 \
                                                            + W[9, 10] * x10 + e
        z11 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10],
                        noise_distr=e11, mechanism_fct=f11)

        e12 = norm(0, .1)
        f12 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, e: \
            W[0, 11] * x1 \
            + W[1, 11] * x2 \
            + W[2, 11] * x3 \
            + W[3, 11] * x4 \
            + W[4, 11] * x5 \
            + W[5, 11] * x6 \
            + W[6, 11] * x7 \
            + W[7, 11] * x8 \
            + W[8, 11] * x9 \
            + W[9, 11] * x10 \
            + W[10, 11] * x11 + e
        z12 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11],
                        noise_distr=e12, mechanism_fct=f12)

        e13 = norm(0, .1)
        f13 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, e: \
            W[0, 12] * x1 \
            + W[1, 12] * x2 \
            + W[2, 12] * x3 \
            + W[3, 12] * x4 \
            + W[4, 12] * x5 \
            + W[5, 12] * x6 \
            + W[6, 12] * x7 \
            + W[7, 12] * x8 \
            + W[8, 12] * x9 \
            + W[9, 12] * x10 \
            + W[10, 12] * x11 \
            + W[11, 12] * x12 + e
        z13 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12],
                        noise_distr=e13, mechanism_fct=f13)

        e14 = norm(0, .1)
        f14 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, e: \
            W[0, 13] * x1 \
            + W[1, 13] * x2 \
            + W[2, 13] * x3 \
            + W[3, 13] * x4 \
            + W[4, 13] * x5 \
            + W[5, 13] * x6 \
            + W[6, 13] * x7 \
            + W[7, 13] * x8 \
            + W[8, 13] * x9 \
            + W[9, 13] * x10 \
            + W[10, 13] * x11 \
            + W[11, 13] * x12 \
            + W[12, 13] * x13 + e
        z14 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13],
                        noise_distr=e14, mechanism_fct=f14)

        e15 = norm(0, .1)
        f15 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, e: \
            W[0, 14] * x1 \
            + W[1, 14] * x2 \
            + W[2, 14] * x3 \
            + W[3, 14] * x4 \
            + W[4, 14] * x5 \
            + W[5, 14] * x6 \
            + W[6, 14] * x7 \
            + W[7, 14] * x8 \
            + W[8, 14] * x9 \
            + W[9, 14] * x10 \
            + W[10, 14] * x11 \
            + W[11, 14] * x12 \
            + W[12, 14] * x13 \
            + W[13, 14] * x14 + e
        z15 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14],
                        noise_distr=e15, mechanism_fct=f15)

        e16 = norm(0, .1)
        f16 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, e: \
            W[0, 15] * x1 \
            + W[1, 15] * x2 \
            + W[2, 15] * x3 \
            + W[3, 15] * x4 \
            + W[4, 15] * x5 \
            + W[5, 15] * x6 \
            + W[6, 15] * x7 \
            + W[7, 15] * x8 \
            + W[8, 15] * x9 \
            + W[9, 15] * x10 \
            + W[10, 15] * x11 \
            + W[11, 15] * x12 \
            + W[12, 15] * x13 \
            + W[13, 15] * x14 \
            + W[14, 15] * x15 + e
        z16 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15],
                        noise_distr=e16, mechanism_fct=f16)

        e17 = norm(0, .1)
        f17 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, e: \
            W[0, 16] * x1 \
            + W[1, 16] * x2 \
            + W[2, 16] * x3 \
            + W[3, 16] * x4 \
            + W[4, 16] * x5 \
            + W[5, 16] * x6 \
            + W[6, 16] * x7 \
            + W[7, 16] * x8 \
            + W[8, 16] * x9 \
            + W[9, 16] * x10 \
            + W[10, 16] * x11 \
            + W[11, 16] * x12 \
            + W[12, 16] * x13 \
            + W[13, 16] * x14 \
            + W[14, 16] * x15 \
            + W[15, 16] * x16 + e
        z17 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16],
                        noise_distr=e17, mechanism_fct=f17)

        e18 = norm(0, .1)
        f18 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, e: \
            W[0, 17] * x1 \
            + W[1, 17] * x2 \
            + W[2, 17] * x3 \
            + W[3, 17] * x4 \
            + W[4, 17] * x5 \
            + W[5, 17] * x6 \
            + W[6, 17] * x7 \
            + W[7, 17] * x8 \
            + W[8, 17] * x9 \
            + W[9, 17] * x10 \
            + W[10, 17] * x11 \
            + W[11, 17] * x12 \
            + W[12, 17] * x13 \
            + W[13, 17] * x14 \
            + W[14, 17] * x15 \
            + W[15, 17] * x16 \
            + W[16, 17] * x17 + e
        z18 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17],
                        noise_distr=e18, mechanism_fct=f18)

        e19 = norm(0, .1)
        f19 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, e: \
            W[0, 18] * x1 \
            + W[1, 18] * x2 \
            + W[2, 18] * x3 \
            + W[3, 18] * x4 \
            + W[4, 18] * x5 \
            + W[5, 18] * x6 \
            + W[6, 18] * x7 \
            + W[7, 18] * x8 \
            + W[8, 18] * x9 \
            + W[9, 18] * x10 \
            + W[10, 18] * x11 \
            + W[11, 18] * x12 \
            + W[12, 18] * x13 \
            + W[13, 18] * x14 \
            + W[14, 18] * x15 \
            + W[15, 18] * x16 \
            + W[16, 18] * x17 \
            + W[17, 18] * x18 + e
        z19 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18],
                        noise_distr=e19, mechanism_fct=f19)

        e20 = norm(0, .1)
        f20 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, e: \
                      W[0, 19] * x1 \
                    + W[1, 19] * x2 \
                    + W[2, 19] * x3 \
                    + W[3, 19] * x4 \
                    + W[4, 19] * x5 \
                    + W[5, 19] * x6 \
                    + W[6, 19] * x7 \
                    + W[7, 19] * x8 \
                    + W[8, 19] * x9 \
                    + W[9, 19] * x10 \
                    + W[10, 19] * x11 \
                    + W[11, 19] * x12 \
                    + W[12, 19] * x13 \
                    + W[13, 19] * x14 \
                    + W[14, 19] * x15 \
                    + W[15, 19] * x16 \
                    + W[16, 19] * x17 \
                    + W[17, 19] * x18 \
                    + W[18, 19] * x19 + e

        z20 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19],
                        noise_distr=e20, mechanism_fct=f20)

        e21 = norm(0, .1)
        f21 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, e: \
            W[0, 20] * x1 \
            + W[1, 20] * x2 \
            + W[2, 20] * x3 \
            + W[3, 20] * x4 \
            + W[4, 20] * x5 \
            + W[5, 20] * x6 \
            + W[6, 20] * x7 \
            + W[7, 20] * x8 \
            + W[8, 20] * x9 \
            + W[9, 20] * x10 \
            + W[10, 20] * x11 \
            + W[11, 20] * x12 \
            + W[12, 20] * x13 \
            + W[13, 20] * x14 \
            + W[14, 20] * x15 \
            + W[15, 20] * x16 \
            + W[16, 20] * x17 \
            + W[17, 20] * x18 \
            + W[18, 20] * x19 \
            + W[19, 20] * x20 + e
        z21 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20],
                        noise_distr=e21, mechanism_fct=f21)

        e22 = norm(0, .1)
        f22 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, e: \
            W[0, 21] * x1 \
            + W[1, 21] * x2 \
            + W[2, 21] * x3 \
            + W[3, 21] * x4 \
            + W[4, 21] * x5 \
            + W[5, 21] * x6 \
            + W[6, 21] * x7 \
            + W[7, 29] * x8 \
            + W[8, 21] * x9 \
            + W[9, 21] * x10 \
            + W[10, 21] * x11 \
            + W[11, 21] * x12 \
            + W[12, 21] * x13 \
            + W[13, 21] * x14 \
            + W[14, 21] * x15 \
            + W[15, 21] * x16 \
            + W[16, 21] * x17 \
            + W[17, 21] * x18 \
            + W[18, 21] * x19 \
            + W[19, 21] * x20 \
            + W[20, 21] * x21 + e
        z22 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21],
                        noise_distr=e22, mechanism_fct=f22)

        e23 = norm(0, .1)
        f23 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, e: \
            W[0, 22] * x1 \
            + W[1, 22] * x2 \
            + W[2, 22] * x3 \
            + W[3, 22] * x4 \
            + W[4, 22] * x5 \
            + W[5, 22] * x6 \
            + W[6, 22] * x7 \
            + W[7, 22] * x8 \
            + W[8, 22] * x9 \
            + W[9, 22] * x10 \
            + W[10, 22] * x11 \
            + W[11, 22] * x12 \
            + W[12, 22] * x13 \
            + W[13, 22] * x14 \
            + W[14, 22] * x15 \
            + W[15, 22] * x16 \
            + W[16, 22] * x17 \
            + W[17, 22] * x18 \
            + W[18, 22] * x19 \
            + W[19, 22] * x20 \
            + W[20, 22] * x21 \
            + W[21, 22] * x22 + e
        z23 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22],
                        noise_distr=e23, mechanism_fct=f23)

        e24 = norm(0, .1)
        f24 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, e: \
            W[0, 23] * x1 \
            + W[1, 23] * x2 \
            + W[2, 23] * x3 \
            + W[3, 23] * x4 \
            + W[4, 23] * x5 \
            + W[5, 23] * x6 \
            + W[6, 23] * x7 \
            + W[7, 23] * x8 \
            + W[8, 23] * x9 \
            + W[9, 23] * x10 \
            + W[10, 23] * x11 \
            + W[11, 23] * x12 \
            + W[12, 23] * x13 \
            + W[13, 23] * x14 \
            + W[14, 23] * x15 \
            + W[15, 23] * x16 \
            + W[16, 23] * x17 \
            + W[17, 23] * x18 \
            + W[18, 23] * x19 \
            + W[19, 23] * x20 \
            + W[20, 23] * x21 \
            + W[21, 23] * x22 \
            + W[22, 23] * x23 + e
        z24 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23],
                        noise_distr=e24, mechanism_fct=f24)

        e25 = norm(0, .1)
        f25 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, e: \
            W[0, 24] * x1 \
            + W[1, 24] * x2 \
            + W[2, 24] * x3 \
            + W[3, 24] * x4 \
            + W[4, 24] * x5 \
            + W[5, 24] * x6 \
            + W[6, 24] * x7 \
            + W[7, 24] * x8 \
            + W[8, 24] * x9 \
            + W[9, 24] * x10 \
            + W[10, 24] * x11 \
            + W[11, 24] * x12 \
            + W[12, 24] * x13 \
            + W[13, 24] * x14 \
            + W[14, 24] * x15 \
            + W[15, 24] * x16 \
            + W[16, 24] * x17 \
            + W[17, 24] * x18 \
            + W[18, 24] * x19 \
            + W[19, 24] * x20 \
            + W[20, 24] * x21 \
            + W[21, 24] * x22 \
            + W[22, 24] * x23 \
            + W[23, 24] * x24 + e
        z25 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24],
                        noise_distr=e25, mechanism_fct=f25)

        e26 = norm(0, .1)
        f26 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, x25, e: \
            W[0, 25] * x1 \
            + W[1, 25] * x2 \
            + W[2, 25] * x3 \
            + W[3, 25] * x4 \
            + W[4, 25] * x5 \
            + W[5, 25] * x6 \
            + W[6, 25] * x7 \
            + W[7, 25] * x8 \
            + W[8, 25] * x9 \
            + W[9, 25] * x10 \
            + W[10, 25] * x11 \
            + W[11, 25] * x12 \
            + W[12, 25] * x13 \
            + W[13, 25] * x14 \
            + W[14, 25] * x15 \
            + W[15, 25] * x16 \
            + W[16, 25] * x17 \
            + W[17, 25] * x18 \
            + W[18, 25] * x19 \
            + W[19, 25] * x20 \
            + W[20, 25] * x21 \
            + W[21, 25] * x22 \
            + W[22, 25] * x23 \
            + W[23, 25] * x24 \
            + W[24, 25] * x25  + e
        z26 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24, z25],
                        noise_distr=e26, mechanism_fct=f26)

        e27 = norm(0, .1)
        f27 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, x25, x26, e: \
            W[0, 26] * x1 \
            + W[1, 26] * x2 \
            + W[2, 26] * x3 \
            + W[3, 26] * x4 \
            + W[4, 26] * x5 \
            + W[5, 26] * x6 \
            + W[6, 26] * x7 \
            + W[7, 26] * x8 \
            + W[8, 26] * x9 \
            + W[9, 26] * x10 \
            + W[10, 26] * x11 \
            + W[11, 26] * x12 \
            + W[12, 26] * x13 \
            + W[13, 26] * x14 \
            + W[14, 26] * x15 \
            + W[15, 26] * x16 \
            + W[16, 26] * x17 \
            + W[17, 26] * x18 \
            + W[18, 26] * x19 \
            + W[19, 26] * x20 \
            + W[20, 26] * x21 \
            + W[21, 26] * x22 \
            + W[22, 26] * x23 \
            + W[23, 26] * x24 \
            + W[24, 26] * x25 \
            + W[25, 26] * x26 + e
        z27 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24, z25, z26],
                        noise_distr=e27, mechanism_fct=f27)

        e28 = norm(0, .1)
        f28 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, x25, x26, x27, e: \
            W[0, 27] * x1 \
            + W[1, 27] * x2 \
            + W[2, 27] * x3 \
            + W[3, 27] * x4 \
            + W[4, 27] * x5 \
            + W[5, 27] * x6 \
            + W[6, 27] * x7 \
            + W[7, 27] * x8 \
            + W[8, 27] * x9 \
            + W[9, 27] * x10 \
            + W[10, 27] * x11 \
            + W[11, 27] * x12 \
            + W[12, 27] * x13 \
            + W[13, 27] * x14 \
            + W[14, 27] * x15 \
            + W[15, 27] * x16 \
            + W[16, 27] * x17 \
            + W[17, 27] * x18 \
            + W[18, 27] * x19 \
            + W[19, 27] * x20 \
            + W[20, 27] * x21 \
            + W[21, 27] * x22 \
            + W[22, 27] * x23 \
            + W[23, 27] * x24 \
            + W[24, 27] * x25 \
            + W[25, 27] * x26 \
            + W[26, 27] * x27 + e
        z28 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24, z25, z26, z27],
                        noise_distr=e28, mechanism_fct=f28)

        e29 = norm(0, .1)
        f29 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, x25, x26, x27, x28, e: \
            W[0, 28] * x1 \
            + W[1, 28] * x2 \
            + W[2, 28] * x3 \
            + W[3, 28] * x4 \
            + W[4, 28] * x5 \
            + W[5, 28] * x6 \
            + W[6, 28] * x7 \
            + W[7, 28] * x8 \
            + W[8, 28] * x9 \
            + W[9, 28] * x10 \
            + W[10, 28] * x11 \
            + W[11, 28] * x12 \
            + W[12, 28] * x13 \
            + W[13, 28] * x14 \
            + W[14, 28] * x15 \
            + W[15, 28] * x16 \
            + W[16, 28] * x17 \
            + W[17, 28] * x18 \
            + W[18, 28] * x19 \
            + W[19, 28] * x20 \
            + W[20, 28] * x21 \
            + W[21, 28] * x22 \
            + W[22, 28] * x23 \
            + W[23, 28] * x24 \
            + W[24, 28] * x25 \
            + W[25, 28] * x26 \
            + W[26, 28] * x27 \
            + W[27, 28] * x28 + e
        z29 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24, z25, z26, z27, z28],
                        noise_distr=e29, mechanism_fct=f29)

        e30 = norm(0, .1)
        f30 = lambda x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, \
                     x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, \
                     x21, x22, x23, x24, x25, x26, x27, x28, x29, e: \
                W[0, 29] * x1 \
            + W[1, 29] * x2 \
            + W[2, 29] * x3 \
            + W[3, 29] * x4 \
            + W[4, 29] * x5 \
            + W[5, 29] * x6 \
            + W[6, 29] * x7 \
            + W[7, 29] * x8 \
            + W[8, 29] * x9 \
            + W[9, 29] * x10 \
            + W[10, 29] * x11 \
            + W[11, 29] * x12 \
            + W[12, 29] * x13 \
            + W[13, 29] * x14 \
            + W[14, 29] * x15 \
            + W[15, 29] * x16 \
            + W[16, 29] * x17 \
            + W[17, 29] * x18 \
            + W[18, 29] * x19 \
            + W[19, 29] * x20 \
            + W[20, 29] * x21 \
            + W[21, 29] * x22 \
            + W[22, 29] * x23 \
            + W[23, 29] * x24 \
            + W[24, 29] * x25 \
            + W[25, 29] * x26 \
            + W[26, 29] * x27 \
            + W[27, 29] * x28 \
            + W[28, 29] * x29 + e

        z30 = CausalVar(parents=[z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                                 z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                                 z21, z22, z23, z24, z25, z26, z27, z28, z29],
                        noise_distr=e30, mechanism_fct=f30)

        variables = (z1, z2, z3, z4, z5, z6, z7, z8, z9, z10,
                     z11, z12, z13, z14, z15, z16, z17, z18, z19, z20,
                     z21, z22, z23, z24, z25, z26, z27, z28, z29, z30)
        # Interventions
        # Test to generate interventions automatically
        iv_list = []
        for i in range(30):
            # This stays the same across interventions
            new_mechanism=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            targets = np.delete(np.arange(30, dtype=int), i)
            iv_list.append(Intervention(targets=targets,
                                        new_mechanism=new_mechanism))

        interventions = iv_list

    # Get index of obs_target if not None
    if obs_target is not None:
        obs_target = variables.index(obs_target)

    return SCM(variables=variables, seed=seed), interventions, obs_target