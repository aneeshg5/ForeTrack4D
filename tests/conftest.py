import importlib.util

_LABELING_MODULES = ["test_segment.py", "test_demo_pipeline.py", "test_impute.py"]

collect_ignore = []
if any(importlib.util.find_spec(m) is None for m in ("mediapipe", "sam2")):
    collect_ignore.extend(_LABELING_MODULES)
