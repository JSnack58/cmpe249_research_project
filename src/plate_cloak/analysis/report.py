"""H1 report: does the attack transfer less to RT-DETR than to the CNN control?

    python -m plate_cloak report --runs results/runs/A [results/runs/B ...] --out results/reports/X

Reads run folders written by the evaluate stage and writes into --out:
    h1_table.md   per run: clean and attacked mAP50, the drop, recall, one row per architecture
    h1.json       the same numbers plus the H1 verdict per run
    h1_bars.png   clean vs. attacked mAP50 per architecture, with 95% CI error bars

Rows group the targets by role: the surrogate itself (white-box), the surrogate's architecture
retrained with other seeds, then every other architecture. A row's value is the mean over its
seeds. Every number is a bootstrap CI over images, and every statistic in a run uses the same
resamples, so differences between rows or conditions are paired.

H1 holds when the CNN control's drop minus RT-DETR's drop is at least `threshold` (0.35 mAP50,
fixed in advance) and its CI excludes 0.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from plate_cloak.evaluation.detection import Pooled
from plate_cloak.evaluation.matching import ImageMatch
from plate_cloak.evaluation.stats import bootstrap_ci

CONDITIONS = ("clean", "attacked")
METRICS = {"map50": "ap", "recall50": "recall"}  # column name -> Pooled method


@dataclass
class Run:
    """One run folder, loaded: settings, image order, and each target's pooled matches."""

    name: str
    cfg: DictConfig
    n_images: int
    pooled: dict[tuple[str, int, str], Pooled]  # (model, seed, condition) -> matches
    plates: pd.DataFrame  # per_plate.csv
    _cache: dict = field(default_factory=dict, repr=False)

    def score(self, target: tuple[str, int], condition: str, metric: str, w: np.ndarray) -> float:
        """One target's metric on one resample, computed once per resample and reused."""
        key = (*target, condition, metric, hashlib.blake2b(w.tobytes(), digest_size=16).digest())
        if key not in self._cache:
            pooled = self.pooled[(*target, condition)]
            self._cache[key] = getattr(pooled, METRICS[metric])(w)
        return self._cache[key]

    def ci(self, statistic) -> dict[str, float]:
        b = self.cfg.eval.bootstrap
        return bootstrap_ci(self.n_images, statistic, b.n, b.ci, self.cfg.seed).as_dict()

    def seed_mean(self, targets, condition, metric, w) -> float:
        return float(np.mean([self.score(t, condition, metric, w) for t in targets]))

    def drop(self, targets, metric, w) -> float:
        clean = self.seed_mean(targets, "clean", metric, w)
        return clean - self.seed_mean(targets, "attacked", metric, w)


def load_run(run_dir: Path) -> Run:
    """Rebuild every target's per-image matches from a run folder's CSVs."""
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    plates = pd.read_csv(run_dir / "per_plate.csv")
    dets = pd.read_csv(run_dir / "per_detection.csv.gz")
    image_ids = list(pd.unique(plates["image_id"]))  # every eligible image has a counted plate
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    n = len(image_ids)

    pooled = {}
    groups = dict(list(dets.groupby(["model", "seed", "condition"], sort=False)))
    for (model, seed, condition), p in plates.groupby(["model", "seed", "condition"], sort=False):
        p_img = p["image_id"].map(index).to_numpy()
        n_gt = np.bincount(p_img, minlength=n)
        found = _split(p["found"].to_numpy().astype(bool), p_img, n)
        d = groups.get((model, seed, condition))
        if d is None:
            d = dets.iloc[:0]
        d_img = d["image_id"].map(index).to_numpy()
        scores = _split(d["score"].to_numpy(dtype=float), d_img, n)
        is_tp = _split(d["is_tp"].to_numpy().astype(bool), d_img, n)
        matches = [ImageMatch(scores[i], is_tp[i], int(n_gt[i]), found[i]) for i in range(n)]
        pooled[(model, int(seed), condition)] = Pooled(matches)
    return Run(run_dir.name, cfg, n, pooled, plates)


def architecture_rows(cfg: DictConfig) -> dict[str, list[tuple[str, int]]]:
    """Row label -> its targets: white-box surrogate, retrained surrogate, then the others."""
    surrogate = (cfg.models.surrogate.name, cfg.models.surrogate.seed)
    targets = [(t.name, t.seed) for t in cfg.models.targets]
    rows = {}
    if surrogate in targets:
        rows[f"{surrogate[0]} (white-box)"] = [surrogate]
    retrained = [t for t in targets if t[0] == surrogate[0] and t != surrogate]
    if retrained:
        rows[f"{surrogate[0]} (retrained)"] = retrained
    for name in dict.fromkeys(t[0] for t in targets if t[0] != surrogate[0]):
        rows[name] = [t for t in targets if t[0] == name]
    return rows


def h1_table(run_dir: Path | Run) -> pd.DataFrame:
    """One row per architecture: clean, attacked and drop for mAP50 and recall50, as seed means
    with paired CIs, plus the smallest and largest single-seed mAP50 drop."""
    run = run_dir if isinstance(run_dir, Run) else load_run(run_dir)
    ones = np.ones(run.n_images, dtype=np.int64)
    records = {}
    for label, targets in architecture_rows(run.cfg).items():
        row = {"seeds": ",".join(str(seed) for _, seed in targets)}
        for metric in METRICS:
            stats = {
                "clean": lambda w, m=metric, t=targets: run.seed_mean(t, "clean", m, w),
                "attacked": lambda w, m=metric, t=targets: run.seed_mean(t, "attacked", m, w),
                "drop": lambda w, m=metric, t=targets: run.drop(t, m, w),
            }
            for part, statistic in stats.items():
                ci = run.ci(statistic)
                col = f"{part}_{metric}"
                row[col], row[f"{col}_low"], row[f"{col}_high"] = ci["point"], ci["low"], ci["high"]
        seed_drops = [run.drop([t], "map50", ones) for t in targets]
        row["drop_seed_min"], row["drop_seed_max"] = min(seed_drops), max(seed_drops)
        records[label] = row
    table = pd.DataFrame.from_dict(records, orient="index")
    table.index.name = "architecture"
    return table


def h1_verdict(
    run_dir: Path | Run,
    threshold: float = 0.35,
    cnn_control: str = "yolov10l",
    transformer: str = "rtdetr-l",
) -> dict | None:
    """The H1 test on mAP50 drops, or None if the run lacks either architecture."""
    run = run_dir if isinstance(run_dir, Run) else load_run(run_dir)
    targets = [(t.name, t.seed) for t in run.cfg.models.targets]
    cnn = [t for t in targets if t[0] == cnn_control]
    tf = [t for t in targets if t[0] == transformer]
    if not cnn or not tf:
        return None
    difference = run.ci(lambda w: run.drop(cnn, "map50", w) - run.drop(tf, "map50", w))
    return {
        "metric": "map50",
        "cnn_control": cnn_control,
        "transformer": transformer,
        "threshold": threshold,
        "difference": difference,
        "supported": bool(difference["point"] >= threshold and difference["low"] > 0),
        "attack_success": _attack_success(run, targets),
    }


def report(runs: list[Path], out: Path) -> None:
    """Write h1_table.md, h1.json and h1_bars.png for the given run folders."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for run_dir in runs:
        run = load_run(run_dir)
        results[run.name] = (h1_table(run), h1_verdict(run))
        print(f"report: {run.name} done")
    (out / "h1_table.md").write_text(_markdown(results))
    data = {
        name: {"table": table.reset_index().to_dict(orient="records"), "verdict": verdict}
        for name, (table, verdict) in results.items()
    }
    (out / "h1.json").write_text(json.dumps(data, indent=2) + "\n")
    _bar_chart(results, out / "h1_bars.png")
    print(f"report: wrote {out}")


def _attack_success(run: Run, targets: list[tuple[str, int]]) -> dict:
    """Share of plates missed when attacked, among plates every target found when clean.

    This isolates the attack's effect from plates some model never finds anyway.
    Values are seed means per architecture.
    """
    p = run.plates
    clean = p[p["condition"] == "clean"]
    found_by_all = clean.groupby(["image_id", "plate_idx"])["found"].min()
    kept = found_by_all[found_by_all == 1].index
    attacked = p[p["condition"] == "attacked"].set_index(["image_id", "plate_idx"])
    attacked = attacked[attacked.index.isin(kept)]
    rate = {}
    for name in dict.fromkeys(t[0] for t in targets):
        rows = attacked[attacked["model"] == name]
        missed = 1 - rows.groupby("seed")["found"].mean()
        rate[name] = float(missed.mean()) if len(missed) else float("nan")
    return {"n_plates": int(len(kept)), "rate": rate}


def _split(values: np.ndarray, image: np.ndarray, n: int) -> list[np.ndarray]:
    """values grouped by image index, as a list of n arrays (in order within each image)."""
    order = np.argsort(image, kind="stable")
    bounds = np.cumsum(np.bincount(image, minlength=n))[:-1]
    return np.split(values[order], bounds)


def _fmt(row: pd.Series, col: str) -> str:
    return f"{row[col]:.3f} [{row[f'{col}_low']:.3f}, {row[f'{col}_high']:.3f}]"


def _markdown(results: dict) -> str:
    lines = [
        "# H1: clean vs. attacked plate detection",
        "",
        "Values are seed means with 95% paired bootstrap CIs over images.",
        "Drop range is the smallest and largest single-seed mAP50 drop.",
        "",
    ]
    for name, (table, verdict) in results.items():
        lines += [
            f"## {name}",
            "",
            "| Architecture | Seeds | Clean mAP50 | Attacked mAP50 | mAP50 drop | Drop range "
            "| Clean recall | Attacked recall |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label, row in table.iterrows():
            lines.append(
                f"| {label} | {row['seeds']} | {_fmt(row, 'clean_map50')} "
                f"| {_fmt(row, 'attacked_map50')} | {_fmt(row, 'drop_map50')} "
                f"| {row['drop_seed_min']:.3f} to {row['drop_seed_max']:.3f} "
                f"| {_fmt(row, 'clean_recall50')} | {_fmt(row, 'attacked_recall50')} |"
            )
        lines.append("")
        if verdict is None:
            lines += ["No H1 verdict: this run lacks the CNN control or the Transformer.", ""]
            continue
        d = verdict["difference"]
        outcome = "supported" if verdict["supported"] else "not supported"
        lines += [
            f"**H1 {outcome}.** "
            f"Drop({verdict['cnn_control']}) - drop({verdict['transformer']}) = "
            f"{d['point']:.3f} [{d['low']:.3f}, {d['high']:.3f}] mAP50 "
            f"(needs >= {verdict['threshold']} with the CI above 0).",
            "",
            f"Attack success on the {verdict['attack_success']['n_plates']} plates every target "
            "found when clean: "
            + ", ".join(f"{k} {v:.3f}" for k, v in verdict["attack_success"]["rate"].items())
            + ".",
            "",
        ]
    return "\n".join(lines)


def _bar_chart(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1, len(results), figsize=(5 * len(results), 4), squeeze=False, sharey=True
    )
    for ax, (name, (table, _)) in zip(axes[0], results.items(), strict=True):
        x = np.arange(len(table))
        for offset, part, color in ((-0.2, "clean", "#4c78a8"), (0.2, "attacked", "#e45756")):
            col = f"{part}_map50"
            err = [table[col] - table[f"{col}_low"], table[f"{col}_high"] - table[col]]
            ax.bar(x + offset, table[col], 0.4, yerr=err, capsize=3, color=color, label=part)
        ax.set_xticks(x, table.index, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(name)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("plate mAP@0.5")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))  # leave room above the plots for the legend
    fig.savefig(path, dpi=150)
    plt.close(fig)
