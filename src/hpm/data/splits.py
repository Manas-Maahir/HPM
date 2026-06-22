from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from omegaconf import DictConfig


def make_identity_splits(
    root: Path,
    cfg: DictConfig,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Return (train, val, test) sample lists stratified by identity.

    Each sample is (image_path, identity_label_int).
    Stratification guarantees no identity appears in more than one split.

    VGGFace2 layout: root/{n000001}/{0001_01.jpg, ...}
    Identity label is the index into the sorted identity list — stable across runs.
    """
    identity_dirs = sorted(d for d in root.iterdir() if d.is_dir())

    # Shuffle identities (not images) with a seeded RNG for reproducibility.
    rng = random.Random(seed)
    shuffled = list(identity_dirs)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_val = max(1, int(n * cfg.data.val_fraction))
    n_test = max(1, int(n * cfg.data.test_fraction))
    n_train = n - n_val - n_test

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    # identity_int = position in the original sorted list → stable across seeds
    id_to_int = {d.name: i for i, d in enumerate(identity_dirs)}

    def _collect(dirs: list[Path]) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for d in dirs:
            label = id_to_int[d.name]
            for img_path in sorted(d.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    samples.append((img_path, label))
        return samples

    return _collect(train_ids), _collect(val_ids), _collect(test_ids)


def _resolve_celeba_paths(root: Path, cfg: DictConfig) -> tuple[Path, Path]:
    """Locate the CelebA image directory and identity file under ``root``.

    Kaggle's ``jessicali9530/celeba-dataset`` unzips with a doubly-nested image
    folder (``img_align_celeba/img_align_celeba/*.jpg``); descend into it if present.
    """
    identity_name = cfg.data.get("identity_file", "identity_CelebA.txt")
    image_dir_name = cfg.data.get("image_dir", "img_align_celeba")

    identity_file = root / identity_name
    if not identity_file.exists():
        # Fall back to a recursive search so layout quirks don't break the run.
        matches = list(root.rglob(identity_name))
        if not matches:
            raise FileNotFoundError(f"{identity_name} not found under {root}")
        identity_file = matches[0]

    image_dir = root / image_dir_name
    if not image_dir.is_dir():
        matches = [p for p in root.rglob(image_dir_name) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"{image_dir_name}/ not found under {root}")
        image_dir = matches[0]
    # Descend one level if the images live in a same-named child folder.
    nested = image_dir / image_dir_name
    if nested.is_dir():
        image_dir = nested
    return image_dir, identity_file


def make_celeba_identity_splits(
    root: Path,
    cfg: DictConfig,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Identity-disjoint train/val/test splits for the flat CelebA layout.

    Reads ``identity_CelebA.txt`` (lines: ``000001.jpg 2880``), partitions
    **by identity** (~80/10/10), and persists the identity→split assignment to disk
    so it is never silently regenerated (phaseA_celeba_contrastive.md §2). On a later
    run with the same seed the saved assignment is reloaded.

    Identity labels are the index into the sorted identity list → stable across runs.
    """
    root = Path(root)
    image_dir, identity_file = _resolve_celeba_paths(root, cfg)

    id_to_imgs: dict[int, list[str]] = defaultdict(list)
    with open(identity_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            fname, idstr = parts
            id_to_imgs[int(idstr)].append(fname)

    identities = sorted(id_to_imgs)
    id_to_int = {ident: i for i, ident in enumerate(identities)}

    splits_dir = Path(cfg.data.get("splits_dir", None) or (root / "splits"))
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits_path = splits_dir / f"celeba_seed{seed}.json"

    if splits_path.exists():
        assignment = json.loads(splits_path.read_text())
        train_ids = [int(i) for i in assignment["train"]]
        val_ids = [int(i) for i in assignment["val"]]
        test_ids = [int(i) for i in assignment["test"]]
    else:
        rng = random.Random(seed)
        shuffled = list(identities)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = max(1, int(n * cfg.data.val_fraction))
        n_test = max(1, int(n * cfg.data.test_fraction))
        n_train = n - n_val - n_test
        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]
        splits_path.write_text(json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}))

    def _collect(ids: list[int]) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for ident in ids:
            label = id_to_int[ident]
            for fname in sorted(id_to_imgs[ident]):
                samples.append((image_dir / fname, label))
        return samples

    return _collect(train_ids), _collect(val_ids), _collect(test_ids)


def make_splits(
    root: Path,
    cfg: DictConfig,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Dispatch to the right split builder based on ``cfg.data.name``.

    CelebA uses the flat-folder + identity-file layout; everything else uses the
    per-identity-directory layout handled by ``make_identity_splits``.
    """
    name = str(cfg.data.get("name", "")).lower()
    if name == "celeba":
        return make_celeba_identity_splits(root, cfg, seed)
    return make_identity_splits(root, cfg, seed)
