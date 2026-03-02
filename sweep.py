"""
Hyperparameter sweep for phase_gru_mel.

Usage:
    python sweep.py --chunks-path data/chunks/lakh_clean [--epochs 40] [--batch-size 8]

Each combination gets its own tag (used for checkpoints) and MLflow run name,
all logged under the experiment --experiment.
"""

import argparse
import subprocess
import sys
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--chunks-path", type=str, required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--experiment", type=str, default="gru_mel_hparam_sweep")
parser.add_argument("--dry-run", action="store_true", help="print commands without running them")
args = parser.parse_args()

# ── Hyperparameter grid ────────────────────────────────────────────────────────
grid = {
    "n_layers": [2, 3],
    "hidden":   [256, 512],
    "n_mels":   [64, 96],
}

keys   = list(grid.keys())
combos = list(itertools.product(*[grid[k] for k in keys]))

print(f"Sweep: {len(combos)} combinations × {args.epochs} epochs each\n")

for combo in combos:
    params = dict(zip(keys, combo))
    tag = "sweep_L{n_layers}_H{hidden}_M{n_mels}".format(**params)

    cmd = [
        sys.executable, "train.py", "phase_gru_mel",
        "--chunks-path",  args.chunks_path,
        "--experiment",   args.experiment,
        "-t",             tag,
        "-e",             str(args.epochs),
        "-b",             str(args.batch_size),
        "--n-layers",     str(params["n_layers"]),
        "--hidden",       str(params["hidden"]),
        "--n-mels",       str(params["n_mels"]),
    ]

    print("▶", " ".join(cmd))
    if not args.dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ✗ failed (exit {result.returncode}), continuing sweep…\n")
        else:
            print(f"  ✓ done\n")
    else:
        print()

print("Sweep complete." if not args.dry_run else "Dry run done.")
