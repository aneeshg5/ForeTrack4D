"Forecast vs. reality" demo. Imports src/foretrack as a library and reuses
labeling/ + viz/ as-is; no model or data logic belongs here.

## Running

Needs three environments already set up per the main README:
`venv_labeling` (SAM2 + mediapipe + Flask), `venv_tapip3d` (TAPIP3D +
MegaSaM), `venv_glacier` (the forecaster). Paths for this deployment are in
`config.py` -- update them if running somewhere other than glacier.

```
# install flask into venv_labeling once:
venv_labeling/bin/python -m pip install flask

# terminal 1: worker (processes queued jobs)
cd demo && ../venv_labeling/bin/python worker.py

# terminal 2: web app
cd demo && ../venv_labeling/bin/python app.py
```

Then visit http://localhost:5000, upload a short (<15s) single-shot video
of a hand manipulating one object, and pick a conditioning frame fraction
(default 0.25). Jobs run async -- the status page auto-refreshes until
done. Jobs are async by design: MegaSaM latency makes anything interactive
misleading.

## Known gaps

- "No zooms" input validation is not implemented (needs known intrinsics
  or optical flow analysis).
- Plain HTML range input for the conditioning frame, not a JS video
  scrubber with live preview.
- Single worker process, jobs processed sequentially.
- ADE-vs-horizon curves are returned in the job JSON but not yet rendered
  as a chart on the results page.
- Poorly-lit uploads can fail hand detection outright and are rejected
  with an explicit message.
