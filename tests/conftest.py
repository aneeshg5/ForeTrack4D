# SAM2 has no PyPI release (it is installed from the third_party/ clone) and mediapipe is an
# optional extra, so a base install cannot import the labeling stack. Skip only the modules that
# need it rather than failing collection, so CI and a plain `pip install -e ".[dev]"` still run
# the model, data, metric and rendering tests.

import importlib.util

_LABELING_MODULES = ["test_segment.py", "test_demo_pipeline.py", "test_impute.py"]

collect_ignore = []
if any(importlib.util.find_spec(m) is None for m in ("mediapipe", "sam2")):
    collect_ignore.extend(_LABELING_MODULES)
