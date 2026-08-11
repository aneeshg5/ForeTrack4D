#!/usr/bin/env bash
# clones TAPIP3D and sam2 into third_party/, then prints the two-env instructions.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p third_party

if [ ! -d third_party/tapip3d ]; then
    git clone https://github.com/zbw001/TAPIP3D third_party/tapip3d
fi

if [ ! -d third_party/sam2 ]; then
    git clone https://github.com/facebookresearch/sam2 third_party/sam2
fi

cat <<'EOF'

third_party/tapip3d and third_party/sam2 are cloned.

Two separate environments are needed:

1. foretrack env (this repo's training stack, derived from forehand4d):
   conda create -n foretrack python=3.10
   conda activate foretrack
   pip install -e ".[dev]"

2. tapip3d env (kept separate, do not merge with the above):
   cd third_party/tapip3d
   follow that repo's README, including compiling pointops2 and megasam
   the labeling wrapper (src/foretrack/labeling/run_tapip3d.py) calls this
   env's python via subprocess and exchanges data through npz files

Next: download the TAPIP3D checkpoint and smoke-test its inference.py on
demo_inputs/dexycb.npz from within the tapip3d env.
EOF
