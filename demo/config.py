# Deployment-specific paths for this host; update explicitly when running elsewhere.

REPO_ROOT = "/home/aneeshg5/foretrack4d"

HAND_MODEL_PATH = "/data01/aneeshg5/hand_landmarker.task"
SAM2_CHECKPOINT = f"{REPO_ROOT}/third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

TAPIP3D_PYTHON = f"{REPO_ROOT}/venv_tapip3d/bin/python"
TAPIP3D_REPO = f"{REPO_ROOT}/third_party/tapip3d"
TAPIP3D_CHECKPOINT = f"{REPO_ROOT}/third_party/tapip3d/checkpoints/tapip3d_final.pth"

FORECAST_PYTHON = f"{REPO_ROOT}/venv_glacier/bin/python"
FORECAST_SCRIPT = f"{REPO_ROOT}/scripts/forecast_infer.py"
# expanded-data Stage 2 diffusion checkpoint -- diffusion is the mandated primary architecture, not
# the regressor, even though it doesn't yet have an OOD-gated regressor counterpart.
FORECAST_CONFIG = f"{REPO_ROOT}/configs/mixed_stage2_diffusion_lowlr_expanded.yaml"
FORECAST_CHECKPOINT = "/data01/aneeshg5/model_ckpts/foretrack4d/mixed_stage2_diffusion_lowlr_expanded/diffusion/epoch20.pt"

JOBS_DIR = f"{REPO_ROOT}/demo/jobs"
K_SAMPLES = 5
DEFAULT_CONDITIONING_FRAC = 0.25
