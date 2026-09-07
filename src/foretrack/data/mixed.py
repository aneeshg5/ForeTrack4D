# Dataset-string mixing follows forehand4d's mechanism. See NOTICE.md.

from torch.utils.data import ConcatDataset

from .arctic import ArcticTracks
from .dexycb import DexYCBTracks
from .h2o import H2OTracks
from .label_weights import horizon_weights, load_calibration
from .pseudo import PseudoTracks

PSEUDO_DATASETS = {"pseudo"}

DATASET_CLASSES = {
    "dexycb": DexYCBTracks, "arctic_ego": ArcticTracks, "h2o": H2OTracks, "pseudo": PseudoTracks,
    "egoexo4d_zeroshot": PseudoTracks,
}


def pseudo_label_weight(cfg: dict, t: int):
    w_cfg = cfg["data"].get("pseudo_label_weighting")
    if not w_cfg or not w_cfg.get("enabled", False):
        return None
    calib = load_calibration(w_cfg["calibration"])
    return horizon_weights(
        t, float(w_cfg.get("fps", calib["fps"])), calib,
        tol_cm=float(w_cfg.get("tol_cm", 5.0)), floor=float(w_cfg.get("floor", 0.0)),
    )


def build_dataset(cfg: dict, split: str, n: int, t: int, augment: bool = False, noise_factor: float = 0.0, scale_factor: float = 0.0):
    names = cfg["dataset"].split("+")
    weight = pseudo_label_weight(cfg, t)
    datasets = []
    for name in names:
        if name not in DATASET_CLASSES:
            raise ValueError(f"unknown dataset '{name}', expected one of {list(DATASET_CLASSES)}")
        ds_cfg = cfg["data"]["datasets"][name]
        cls = DATASET_CLASSES[name]
        datasets.append(
            cls(
                ds_cfg["gt_root"], split, n=n, t=t, transl_stats_path=ds_cfg["transl_stats"],
                augment=augment, noise_factor=noise_factor, scale_factor=scale_factor,
                label_weight=weight if name in PSEUDO_DATASETS else None,
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
