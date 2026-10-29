"""One-line, self-updating progress display for scripts/finetune_all.sh.

    python scripts/train_progress.py          # Ctrl+C to quit (training keeps running)

Reads the files training writes (runs/finetune/<model>_s<seed>/results.csv and the exported
weights/<model>_s<seed>.json), so it works no matter how training was started.
"""

from __future__ import annotations

import csv
import math
import re
import sys
import time
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "finetune"
WEIGHTS = REPO / "weights"
BAR = 24


def planned_runs() -> list[str]:
    """Run names in the order finetune_all.sh trains them (read from the script itself)."""
    script = (REPO / "scripts" / "finetune_all.sh").read_text()
    return [f"{m}_s{s}" for m, s in re.findall(r'"([\w.-]+) (\d+)"', script)]


def epochs_planned(run_dir: Path) -> int:
    args = run_dir / "args.yaml"
    if args.exists():
        match = re.search(r"^epochs:\s*(\d+)", args.read_text(), re.MULTILINE)
        if match:
            return int(match.group(1))
    return 100


def status_line() -> str:
    runs = planned_runs()
    done = [r for r in runs if (WEIGHTS / f"{r}.json").exists()]
    pending = [r for r in runs if r not in done]
    if not pending:
        return f"all {len(runs)} fine-tunes done"

    current = pending[0]
    results = RUNS / current / "results.csv"
    head = f"[{len(done) + 1}/{len(runs)}] {current:<12}"
    if not results.exists():
        return f"{head} starting..."

    with results.open() as f:
        rows = list(csv.DictReader(f, skipinitialspace=True))
    if not rows:
        return f"{head} epoch 1 running..."
    total = epochs_planned(RUNS / current)
    epoch = int(float(rows[-1]["epoch"]))
    # Ultralytics' "time" is cumulative but restarts on resume, so time each epoch from the
    # difference to the one before it and skip the jump back at a resume.
    times = [float(r["time"]) for r in rows]
    steps = [b - a for a, b in pairwise(times) if b > a][-5:] or times[-1:]
    per_epoch = sorted(steps)[len(steps) // 2]
    eta_min = per_epoch * (total - epoch) / 60
    map50 = float(rows[-1]["metrics/mAP50(B)"])
    best = max(float(r["metrics/mAP50(B)"]) for r in rows)
    filled = round(BAR * epoch / total)
    bar = "█" * filled + "░" * (BAR - filled)
    stale = time.time() - results.stat().st_mtime > max(3 * per_epoch, 600)
    losses = [float(v) for k, v in rows[-1].items() if k.startswith("train/")]
    if any(math.isnan(x) for x in losses):
        note = "  (NaN loss: diverging!)"
    elif not map50 >= 0.5 * best:  # same rule as finetune.check_not_collapsed
        note = "  (val mAP collapsed: diverged!)"
    elif stale:
        note = "  (no update lately: stopped?)"
    else:
        note = ""
    return (
        f"{head} {bar} {epoch:>3}/{total}  mAP50 {map50:.3f} (best {best:.3f})  "
        f"{per_epoch:.0f}s/ep  ETA {eta_min // 60:.0f}h{eta_min % 60:02.0f}m{note}"
    )


def main() -> None:
    try:
        while True:
            line = status_line()
            sys.stdout.write("\r\033[K" + line)  # return to line start, clear it, redraw
            sys.stdout.flush()
            time.sleep(10)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
