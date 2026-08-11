import glob

from .dexycb import TrackFramesDataset


class PseudoTracks(TrackFramesDataset):
    """loads TAPIP3D-imputed npz labels (source='pseudo'), same schema as build_track_npz's
    GT output, so TrackFramesDataset's crop/augmentation/padding
    logic applies unchanged -- only the file source differs."""

    def __init__(self, root: str, split: str, **kwargs):
        files = sorted(glob.glob(f"{root}/{split}/**/*.npz", recursive=True))
        if len(files) == 0:
            raise ValueError(f"no npz files found under {root}/{split}")
        super().__init__(files, **kwargs)
