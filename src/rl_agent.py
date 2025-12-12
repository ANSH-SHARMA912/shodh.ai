import numpy as np

try:
    import d3rlpy
    from d3rlpy.dataset import MDPDataset
    from d3rlpy.algos import CQL
except:
    d3rlpy = None

def build_mdp_dataset(X, actions, rewards):
    """Creates a single-step MDPDataset for offline RL."""
    terminals = np.ones(len(X), dtype=bool)
    next_obs = np.zeros_like(X)

    return MDPDataset(
        observations=X,
        actions=actions.reshape(-1, 1),
        rewards=rewards.reshape(-1, 1),
        terminals=terminals,
        next_observations=next_obs
    )

def train_cql(dataset, epochs=20, batch_size=256):
    cql = CQL(actor_learning_rate=3e-4, critic_learning_rate=3e-4)
    cql.fit(dataset.episodes, n_epochs=epochs, batch_size=batch_size, n_steps_per_epoch=100)
    return cql
