import numpy as np

from src.retrieval.metric_learning import combine_weighted_similarity, fit_block_weights


def test_fit_block_weights_favors_informative_block_over_noise():
    rng = np.random.RandomState(0)
    n = 40
    # relevance is HIGH exactly when two items share the same one of 4 latent groups
    groups = rng.randint(0, 4, size=n)
    relevance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and groups[i] == groups[j]:
                relevance[i, j] = 2.0

    # "informative" block directly encodes the same latent groups -> should get high weight
    onehot = np.eye(4)[groups]
    informative_sim = onehot @ onehot.T

    # "noise" block is unrelated random similarity -> should get near-zero weight
    noise = rng.randn(n, 8)
    noise_sim = noise @ noise.T

    block_sims = {"informative": informative_sim, "noise": noise_sim}
    weights = fit_block_weights(list(range(n)), relevance, block_sims, ["informative", "noise"], seed=1)

    assert weights["informative"] > weights["noise"]


def test_combine_weighted_similarity_sums_weighted_blocks():
    sims = {"a": np.array([[1.0, 0.5], [0.5, 1.0]]), "b": np.array([[1.0, 0.0], [0.0, 1.0]])}
    combined = combine_weighted_similarity(sims, {"a": 2.0, "b": 1.0})
    expected = 2.0 * sims["a"] + 1.0 * sims["b"]
    assert np.allclose(combined, expected)
