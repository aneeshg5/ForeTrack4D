from torch.utils.data import ConcatDataset

from .arctic import ArcticTracks
from .dexycb import DexYCBTracks
from .h2o import H2OTracks
from .pseudo import PseudoTracks

# forehand4d's dataset-string mechanism ('dexycb+arctic_ego+h2o') --
# each name maps to its own Dataset class and its own gt_root/transl_stats entry under
# cfg["data"]["datasets"][name] (per-dataset translation normalization, since DexYCB/ARCTIC/H2O
# have very different depth distributions -- see their computed stats in downloads/stats/).
# 'pseudo' reuses this same mechanism for Stage 2's GT + TAPIP3D-imputed mix, e.g.
# 'dexycb+arctic_ego+h2o+pseudo'. 'egoexo4d_zeroshot' reuses the identical PseudoTracks class
# (same npz schema -- tracks/visibility/intrinsics/image_paths/query_xyz_t0) for the EgoExo4D
# zero-shot eval set: never trained on, only ever passed as a single-name eval_dataset to
# scripts/eval.py, never included in cfg["dataset"]'s '+'-joined training mix.
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
