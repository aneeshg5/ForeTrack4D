import argparse
import json
from pathlib import Path

import numpy as np

from foretrack.data.dexycb import load_transl_stats


def load_sequences(root, split, limit=None):
    files = sorted(Path(root).glob(f"{split}/**/*.npz"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        d = np.load(f)
        q = d["tracks"][0].astype(np.float64)  # (N, 3) metric, t=0
        if np.isfinite(q).all():
            out.append(q)
    return out


def scheme_a(q, mean, std):
    return (q - mean) / std


def scheme_c(q, k):
    return (q - q.mean(0)) / (np.median(q[:, 2]) / k)


def scheme_d(q):
    return (q - q.mean(0)) / (q.std(0) + 1e-6)


def scheme_d_iso(q):
    # one scalar instead of three: keeps the object's aspect ratio, which per-axis scaling
    # destroys by squashing every shape toward a sphere
    c = q - q.mean(0)
    return c / (np.sqrt((c ** 2).sum(-1).mean()) + 1e-6)


def summarize(name, coords, ref=None):
    x = np.concatenate(coords)
    std = x.std(0)
    row = {
        "scheme": name,
        "std": np.round(std, 3).tolist(),
        "max_abs": float(np.abs(x).max()),
        "frac_gt3": float((np.abs(x) > 3).mean()),
    }
    if ref is not None:
        # what matters is not absolute spread but spread relative to the training domain the
        # model was actually fit on
        row["std_ratio_to_ref"] = np.round(std / ref, 2).tolist()
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True, help="name=root:split:stats, comma separated")
    p.add_argument("--ref", required=True, help="dataset name to treat as the in-domain reference")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    specs = {}
    for item in args.datasets.split(","):
        name, rest = item.split("=", 1)
        root, split, stats = rest.split(":")
        specs[name] = (root, split, stats)

    seqs = {n: load_sequences(r, s) if args.limit is None else load_sequences(r, s, args.limit)
            for n, (r, s, _) in specs.items()}
    for n, v in seqs.items():
        print(f"{n}: {len(v)} sequences")

    # scheme C's per-axis constant is calibrated on the reference set so that it reproduces unit
    # spread there, exactly as the frozen-weight probe did
    ref_c = np.concatenate([scheme_c(q, np.ones(3)) for q in seqs[args.ref]])
    k = 1.0 / (ref_c.std(0) + 1e-9)
    print(f"scheme C per-axis constant calibrated on {args.ref}: {np.round(k, 3).tolist()}")

    results = {}
    for scheme in ["A", "C", "D", "D_iso"]:
        rows = []
        ref_std = None
        for n in specs:
            _, _, stats = specs[n]
            if scheme == "A":
                mean, std = load_transl_stats(specs[args.ref][2])  # reference stats: a new scene has no stats of its own
                coords = [scheme_a(q, mean, std) for q in seqs[n]]
            elif scheme == "C":
                coords = [scheme_c(q, k) for q in seqs[n]]
            elif scheme == "D":
                coords = [scheme_d(q) for q in seqs[n]]
            else:
                coords = [scheme_d_iso(q) for q in seqs[n]]
            if n == args.ref:
                ref_std = np.concatenate(coords).std(0)
            rows.append((n, coords))
        results[scheme] = [summarize(n, c, ref_std) | {"dataset": n} for n, c in rows]

    for scheme, rows in results.items():
        label = {"A": "dataset mean/std", "C": "query-centered / query depth", "D": "query-centered / query spread, per-axis",
                 "D_iso": "query-centered / query spread, isotropic"}[scheme]
        print(f"\n=== scheme {scheme}: {label} ===")
        print(f"{'dataset':<20} {'std':<24} {'max|x|':>8} {'frac>3':>8}  std ratio to ref")
        for r in rows:
            print(f"{r['dataset']:<20} {str(r['std']):<24} {r['max_abs']:>8.2f} {r['frac_gt3']:>8.3f}  {r.get('std_ratio_to_ref')}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
