"""
Simon Bing, TU Berlin
2024
"""
import numpy as np


def rand_weight_matrix(seed, nodes=3, connect_prob=0.5, wmin=0.1, wmax=1.0):
    """
    :param nodes: number of nodes
    :param connect_prob: probability of an edge
    :return: Upper diagonal weight matrix
    """
    rng = np.random.RandomState(seed=seed)

    adjacency_matrix = np.zeros([nodes, nodes],
                                dtype=np.int32)  # [parents, nodes]
    weight_matrix = np.zeros([nodes, nodes],
                             dtype=np.float32)  # [parents, nodes]

    causal_order = np.flip(np.arange(nodes))

    for i in range(nodes - 1):
        node = causal_order[i]
        potential_parents = causal_order[(i + 1):]
        num_parents = rng.binomial(n=nodes - i - 1, p=connect_prob)
        parents = rng.choice(potential_parents, size=num_parents,
                                   replace=False)
        adjacency_matrix[parents, node] = 1

    for i in range(nodes):
        for j in range(nodes):
            if adjacency_matrix[i, j] == 1:
                weight_matrix[i, j] = rng.uniform(wmin, wmax)

    return weight_matrix


def get_rand_A(n, seed=42):
    rng = np.random.RandomState(seed=seed)
    A = rng.rand(n, n)
    Ax = np.sum(np.abs(A), axis=1)
    np.fill_diagonal(A, Ax)

    return A
