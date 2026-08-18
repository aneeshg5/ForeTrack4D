# Dataset-string mixing follows forehand4d's mechanism. See NOTICE.md.

from torch.utils.data import ConcatDataset

from .arctic import ArcticTracks
from .dexycb import DexYCBTracks
from .h2o import H2OTracks
from .pseudo import PseudoTracks

DATASET_CLASSES = {
    "dexycb": DexYCBTracks, "arctic_ego": ArcticTracks, "h2o": H2OTracks, "pseudo": PseudoTracks,
    "egoexo4d_zeroshot": PseudoTracks,
}


def build_dataset(cfg: dict, split: str, n: int, t: int, augment: bool = False, noise_factor: float = 0.0, scale_factor: float = 0.0):
    names = cfg["dataset"].split("+")
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
            )
        )
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
