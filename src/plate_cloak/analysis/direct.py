"""Milestone 2b: reports over several direct-attack runs (spec 2026-11-12, sections 4 and 5).

    python -m plate_cloak direct strength --runs DIR... --out DIR
    python -m plate_cloak direct verdict --runs DIR... --out DIR

Each run attacks one model with its own weights (its single surrogate, also one of its targets);
its white-box drop is that model's mAP50 drop on its own attacked images.

- `strength`: every run's white-box drop against the strength bar (0.70, the owner's decision of
  2026-11-12 after the first results; the spec's pre-registered 0.80 is recorded in
  docs/experiment-decisions.md). A point estimate: it checks that the attack works.
- `verdict`: H1-direct. The CNN control's white-box drop, averaged over seeds, minus the
  Transformer's, with one bootstrap over images shared by every run (the runs attack the same
  test images, so the difference is paired). Supported when the difference is at least 0.35 and
  its CI excludes 0. Also writes the transfer matrix: each attacked set's drop on every target
  (a by-product, not part of the verdict).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from plate_cloak.analysis.report import Run, _attack_success, load_run, surrogates_of
from plate_cloak.evaluation.stats import bootstrap_ci

STRENGTH_BAR = 0.70
H1_THRESHOLD = 0.35


def white_box(run: Run) -> tuple[str, int]:
    """The model a direct run attacked: its single surrogate, which must also be a target."""
    surrogates = surrogates_of(run.cfg)
    if len(surrogates) != 1:
        raise ValueError(f"{run.name}: a direct run has one surrogate, not {surrogates}")
    (me,) = surrogates
    if me not in [(t.name, t.seed) for t in run.cfg.models.targets]:
        raise ValueError(f"{run.name}: its surrogate {me} is not one of its targets")
    return me


def strength_check(runs: list[Path], bar: float = STRENGTH_BAR) -> list[dict]:
    """Per run: the white-box mAP50 drop (with its CI) and whether its point reaches `bar`."""
    rows = []
    for run_dir in runs:
        run = load_run(run_dir)
        me = white_box(run)
        drop = run.ci(lambda w, r=run, t=me: r.drop([t], "map50", w))
        rows.append(
            {
                "run": run.name,
                "model": me[0],
                "seed": me[1],
                "drop": drop,
                "bar": bar,
                "passed": bool(drop["point"] >= bar),
            }
        )
    return rows


def h1_direct(
    runs: list[Path],
    threshold: float = H1_THRESHOLD,
    cnn: str = "yolov10l",
    transformer: str = "rtdetr-l",
) -> dict:
    """The pre-registered H1-direct test over the white-box runs of both models."""
    by_model = _by_white_box([load_run(r) for r in runs])
    groups = {
        m: {s: r for (name, s), r in by_model.items() if name == m} for m in (cnn, transformer)
    }
    seeds = sorted(groups[cnn])
    if not seeds or seeds != sorted(groups[transformer]):
        raise ValueError(
            f"{cnn} and {transformer} need the same seeds: "
            f"{sorted(groups[cnn])} vs. {sorted(groups[transformer])}"
        )
    runs_used = [groups[m][s] for m in (cnn, transformer) for s in seeds]
    _same_images(runs_used)

    def mean_drop(model, w):
        return float(np.mean([groups[model][s].drop([(model, s)], "map50", w) for s in seeds]))

    first = runs_used[0]
    b = first.cfg.eval.bootstrap

    def ci(statistic):
        return bootstrap_ci(first.n_images, statistic, b.n, b.ci, first.cfg.seed).as_dict()

    difference = ci(lambda w: mean_drop(cnn, w) - mean_drop(transformer, w))
    return {
        "metric": "map50",
        "seeds": seeds,
        "cnn": cnn,
        "transformer": transformer,
        "cnn_drop": ci(lambda w: mean_drop(cnn, w)),
        "transformer_drop": ci(lambda w: mean_drop(transformer, w)),
        "difference": difference,
        "threshold": threshold,
        "supported": bool(difference["point"] >= threshold and difference["low"] > 0),
        "attack_success": {
            f"{m}_s{s}": _attack_success(groups[m][s]) for m in (cnn, transformer) for s in seeds
        },
    }


def transfer_matrix(runs: list[Path]) -> pd.DataFrame:
    """Rows: the attacked model and seed; columns: each target's mAP50 drop on those images."""
    records = {}
    for run_dir in runs:
        run = load_run(run_dir)
        name, seed = white_box(run)
        ones = np.ones(run.n_images, dtype=np.int64)
        records[f"{name}_s{seed}"] = {
            f"{t.name}_s{t.seed}": run.drop([(t.name, t.seed)], "map50", ones)
            for t in run.cfg.models.targets
        }
    return pd.DataFrame.from_dict(records, orient="index")


def write_strength(runs: list[Path], out: Path, bar: float = STRENGTH_BAR) -> None:
    rows = strength_check(runs, bar)
    out.mkdir(parents=True, exist_ok=True)
    (out / "strength.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Strength check: white-box mAP50 drop per run",
        "",
        f"Passes when the point estimate reaches {bar} (95% bootstrap CI over images shown).",
        "",
        "| Attacked model | Run | mAP50 drop | Passed |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']}_s{r['seed']} | {r['run']} | {_ci(r['drop'])} "
            f"| {'yes' if r['passed'] else 'no'} |"
        )
    (out / "strength.md").write_text("\n".join(lines) + "\n")
    print(f"direct: wrote {out / 'strength.md'}")


def write_verdict(runs: list[Path], out: Path) -> None:
    verdict = h1_direct(runs)
    matrix = transfer_matrix(runs)
    out.mkdir(parents=True, exist_ok=True)
    data = {"verdict": verdict, "transfer": matrix.reset_index(names="attacked").to_dict("records")}
    (out / "h1_direct.json").write_text(json.dumps(data, indent=2) + "\n")
    v = verdict
    outcome = "supported" if v["supported"] else "not supported"
    seeds = ", ".join(str(s) for s in v["seeds"])
    lines = [
        "# H1-direct: each model attacked with its own weights",
        "",
        f"mAP50 drops are means over seeds {seeds}, each seed's images attacked with that seed's "
        "weights; 95% bootstrap CIs over images, one resampling shared by every run.",
        "",
        f"- {v['cnn']} (CNN control): {_ci(v['cnn_drop'])}",
        f"- {v['transformer']} (Transformer): {_ci(v['transformer_drop'])}",
        "",
        f"**H1-direct {outcome}.** Drop({v['cnn']}) - drop({v['transformer']}) = "
        f"{_ci(v['difference'])} mAP50 (needs >= {v['threshold']} with the CI above 0).",
        "",
        "Attack success (share of plates missed when attacked, among plates every target found "
        "when clean, on the attacked model itself):",
        "",
    ]
    for key, success in v["attack_success"].items():
        rate = success["rate"].get(f"{key.rsplit('_s', 1)[0]} (white-box)", float("nan"))
        lines.append(f"- {key}: {rate:.3f} of {success['n_plates']} plates")
    lines += [
        "",
        "## Transfer (by-product, not part of the verdict)",
        "",
        "Rows: the attacked model and seed. Columns: each target's mAP50 drop on those images.",
        "",
        "| Attacked | " + " | ".join(matrix.columns) + " |",
        "|---|" + "---|" * len(matrix.columns),
    ]
    for attacked, row in matrix.iterrows():
        lines.append(f"| {attacked} | " + " | ".join(f"{x:.3f}" for x in row) + " |")
    (out / "h1_direct.md").write_text("\n".join(lines) + "\n")
    print(f"direct: wrote {out / 'h1_direct.md'}")


def _by_white_box(runs: list[Run]) -> dict[tuple[str, int], Run]:
    found = {}
    for run in runs:
        me = white_box(run)
        if me in found:
            raise ValueError(f"{me} is attacked twice: {found[me].name} and {run.name}")
        found[me] = run
    return found


def _same_images(runs: list[Run]) -> None:
    ids = [list(pd.unique(r.plates["image_id"])) for r in runs]
    for run, other in zip(runs[1:], ids[1:], strict=True):
        if other != ids[0]:
            raise ValueError(
                f"{run.name} and {runs[0].name} must cover the same images in the same order"
            )


def _ci(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
