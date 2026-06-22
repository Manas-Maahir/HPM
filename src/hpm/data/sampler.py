from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler


class PKSampler(Sampler[list[int]]):
    """Batch sampler yielding P identities × K images per batch.

    Guarantees in-batch positives so the supervised-contrastive signal never
    collapses (phaseA_celeba_contrastive.md §3). Each yielded batch is a list of
    dataset indices of length ``P * K`` with exactly K samples per chosen identity.
    Identities with fewer than K images are sampled with replacement.

    Pass to ``DataLoader(dataset, batch_sampler=PKSampler(...))``. Call
    ``set_epoch`` each epoch for a reproducible-yet-varying shuffle on resume.
    """

    def __init__(
        self,
        labels: list[int],
        P: int,
        K: int,
        num_batches: int | None = None,
        seed: int = 0,
    ) -> None:
        self.labels = list(labels)
        self.P = P
        self.K = K
        self.seed = seed
        self.epoch = 0

        self.label_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, lab in enumerate(self.labels):
            self.label_to_indices[lab].append(idx)
        self.unique_labels = list(self.label_to_indices.keys())

        if P > len(self.unique_labels):
            raise ValueError(
                f"PKSampler: P={P} exceeds available identities " f"({len(self.unique_labels)})"
            )

        self.batch_size = P * K
        if num_batches is None:
            num_batches = max(1, len(self.labels) // self.batch_size)
        self.num_batches = num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.num_batches):
            chosen = rng.sample(self.unique_labels, self.P)
            batch: list[int] = []
            for lab in chosen:
                idxs = self.label_to_indices[lab]
                if len(idxs) >= self.K:
                    batch.extend(rng.sample(idxs, self.K))
                else:
                    batch.extend(rng.choices(idxs, k=self.K))
            yield batch
