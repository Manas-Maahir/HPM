"""Behavioral-signature evaluation entry-point.

Loads a checkpoint, re-applies lesion_gain from config, then runs all
behavioral metrics across n_seeds and reports per-stream effect sizes.

Usage:
    python -m hpm.eval.signatures checkpoint=<path> experiment=<variant>
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from hpm.eval.composite import CompositeFaceEffect
from hpm.eval.inversion import InversionEffect
from hpm.eval.part_whole import PartWholeEffect
from hpm.eval.readout_inspect import ReadoutInspector
from hpm.models.hpm_model import HPMModel
from hpm.utils.logging import log_run_metadata
from hpm.utils.seed import seed_everything


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.train.seed)
    log_run_metadata(cfg)

    model: HPMModel = ...  # load from cfg.checkpoint, then:
    # model.callosum.set_lesion_gain(cfg.model.callosum.lesion_gain)

    probe_loader = ...

    results: dict[str, object] = {}
    results["inversion"] = InversionEffect()(model, probe_loader)
    results["composite"] = CompositeFaceEffect()(model, probe_loader)
    results["part_whole"] = PartWholeEffect()(model, probe_loader)
    results["readout"] = ReadoutInspector()(model, probe_loader)

    # Log and report results ...


if __name__ == "__main__":
    main()
