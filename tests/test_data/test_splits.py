from pathlib import Path

from omegaconf import OmegaConf

from hpm.data.splits import make_celeba_identity_splits, make_splits


def _make_celeba(tmp_path: Path, num_ids: int = 20, per_id: int = 5) -> Path:
    root = tmp_path / "celeba"
    img_dir = root / "img_align_celeba"
    img_dir.mkdir(parents=True)
    lines = []
    n = 1
    for ident in range(1, num_ids + 1):
        for _ in range(per_id):
            fname = f"{n:06d}.jpg"
            (img_dir / fname).write_bytes(b"")  # path only; splits don't open images
            lines.append(f"{fname} {ident}")
            n += 1
    (root / "identity_CelebA.txt").write_text("\n".join(lines) + "\n")
    return root


def _cfg(root: Path) -> OmegaConf:
    return OmegaConf.create(
        {
            "data": {
                "name": "celeba",
                "image_dir": "img_align_celeba",
                "identity_file": "identity_CelebA.txt",
                "splits_dir": str(root / "splits"),
                "val_fraction": 0.1,
                "test_fraction": 0.1,
            }
        }
    )


def test_celeba_splits_are_identity_disjoint(tmp_path):
    root = _make_celeba(tmp_path)
    cfg = _cfg(root)
    train, val, test = make_celeba_identity_splits(root, cfg, seed=0)
    a = {lab for _, lab in train}
    b = {lab for _, lab in val}
    c = {lab for _, lab in test}
    assert a and b and c
    assert a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)


def test_celeba_splits_persist_and_reload(tmp_path):
    root = _make_celeba(tmp_path)
    cfg = _cfg(root)
    first = make_celeba_identity_splits(root, cfg, seed=3)
    assert (root / "splits" / "celeba_seed3.json").exists()
    second = make_celeba_identity_splits(root, cfg, seed=3)
    assert {lab for _, lab in first[0]} == {lab for _, lab in second[0]}


def test_make_splits_dispatches_on_name(tmp_path):
    root = _make_celeba(tmp_path)
    cfg = _cfg(root)
    train, _, _ = make_splits(root, cfg, seed=0)
    assert len(train) > 0
