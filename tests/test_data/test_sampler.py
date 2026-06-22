from collections import Counter

import pytest

from hpm.data.sampler import PKSampler


def _labels(num_ids=10, per_id=5):
    return [i for i in range(num_ids) for _ in range(per_id)]


def test_batch_has_p_identities_k_each():
    labels = _labels(num_ids=10, per_id=5)
    P, K = 4, 3
    sampler = PKSampler(labels, P=P, K=K, seed=0)
    for batch in sampler:
        assert len(batch) == P * K
        counts = Counter(labels[i] for i in batch)
        assert len(counts) == P
        assert all(c == K for c in counts.values())


def test_len_matches_num_batches():
    labels = _labels(20, 4)
    sampler = PKSampler(labels, P=4, K=2, seed=1)
    assert len(sampler) == len(list(sampler))


def test_set_epoch_changes_order():
    labels = _labels(20, 4)
    sampler = PKSampler(labels, P=4, K=2, seed=1)
    sampler.set_epoch(0)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second


def test_p_larger_than_identities_raises():
    labels = _labels(3, 5)
    with pytest.raises(ValueError):
        PKSampler(labels, P=4, K=2)
