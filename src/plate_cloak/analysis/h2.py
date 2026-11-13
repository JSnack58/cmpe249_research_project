"""H2 by its written rule (main design 2026-10-20; 2b spec 2026-11-12, section 7).

    python -m plate_cloak h2 --ycbcr-run DIR --rgb-run DIR --ycbcr-lpips CSV --rgb-lpips CSV
        --out DIR [--chart-rungs VIS_DIR... --eyetest-scores JSON... --runs-root DIR]

H2: putting most of the budget in chroma hides plates from YOLOv8s about as well as an RGB
attack, while being less visible. The rule: the YCbCr attack's attacked YOLOv8s (seed 0) recall
is within `tolerance` (0.05) of the RGB attack's, with the whole 95% CI of the difference inside
plus or minus 0.05, and its mean LPIPS is lower.

- Recall: one image bootstrap shared by both runs (they must cover the same images in order).
- LPIPS: paired over the plates scored in both CSVs (skipped crops dropped), bootstrap over
  those plates.

The chart plots each visibility rung's attacked YOLOv8s recall against its mean LPIPS, one line
per ladder, with the eye-test outcome marked: the evidence the rule did not name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from plate_cloak.analysis.direct import check_same_images
from plate_cloak.analysis.report import load_run
from plate_cloak.evaluation.stats import bootstrap_ci

SURROGATE = ("yolov8s", 0)
TOLERANCE = 0.05
EYE_MARKERS = {"pass": "o", "fail": "X", "untested": "s"}


def h2_verdict(
    ycbcr_run: Path,
    rgb_run: Path,
    ycbcr_lpips: Path,
    rgb_lpips: Path,
    tolerance: float = TOLERANCE,
) -> dict:
    """The written H2 rule on one YCbCr run and one RGB run of the same images."""
    y, r = load_run(ycbcr_run), load_run(rgb_run)
    check_same_images([y, r])
    b = y.cfg.eval.bootstrap

    def ci(n, statistic):
        return bootstrap_ci(n, statistic, b.n, b.ci, y.cfg.seed).as_dict()

    def recall(run, w):
        return run.score(SURROGATE, "attacked", "recall50", w)

    n = y.n_images
    recall_diff = ci(n, lambda w: recall(y, w) - recall(r, w))

    both = pd.merge(
        pd.read_csv(ycbcr_lpips), pd.read_csv(rgb_lpips), on=["image_id", "plate_idx"]
    ).dropna(subset=["lpips_x", "lpips_y"])
    ly, lr = both["lpips_x"].to_numpy(), both["lpips_y"].to_numpy()

    def weighted(values):
        return lambda w: float(np.sum(w * values) / np.sum(w))

    m = len(both)
    lpips_diff = ci(m, weighted(ly - lr))
    recall_equivalent = bool(-tolerance < recall_diff["low"] and recall_diff["high"] < tolerance)
    less_visible = bool(lpips_diff["point"] < 0)
    return {
        "surrogate": f"{SURROGATE[0]}_s{SURROGATE[1]}",
        "ycbcr_run": y.name,
        "rgb_run": r.name,
        "tolerance": tolerance,
        "ycbcr_recall": ci(n, lambda w: recall(y, w)),
        "rgb_recall": ci(n, lambda w: recall(r, w)),
        "recall_difference": recall_diff,
        "n_plates_lpips": int(m),
        "ycbcr_lpips": ci(m, weighted(ly)),
        "rgb_lpips": ci(m, weighted(lr)),
        "lpips_difference": lpips_diff,
        "recall_equivalent": recall_equivalent,
        "less_visible": less_visible,
        "supported": recall_equivalent and less_visible,
    }


def h2_chart(
    visibility_dirs: list[Path],
    eyetest_scores: list[Path],
    out: Path,
    runs_root: Path = Path("results/runs"),
) -> list[dict]:
    """Attacked recall against mean LPIPS per visibility rung, eye-test outcome by marker.

    A visibility folder `<ladder>_<rung>` pairs with the run of the same name under runs_root
    and with eye-test budget `<ladder>/<rung>`. Returns the plotted points.
    """
    eye = {}
    for path in eyetest_scores:
        for key, budget in json.loads(Path(path).read_text())["budgets"].items():
            if budget.get("complete", True):
                outcome = "pass" if budget["passed"] else "fail"
                eye[key] = (outcome, f"{budget.get('correct', '?')}/{budget.get('n', '?')}")
    points = []
    for vis in map(Path, visibility_dirs):
        ladder, rung = vis.name.rsplit("_", 1)
        run = load_run(Path(runs_root) / vis.name)
        ones = np.ones(run.n_images, dtype=np.int64)
        points.append(
            {
                "ladder": ladder,
                "rung": rung,
                "lpips": json.loads((vis / "summary.json").read_text())["mean"],
                "recall": run.score(SURROGATE, "attacked", "recall50", ones),
                "eye": eye.get(f"{ladder}/{rung}", ("untested", ""))[0],
                "eye_score": eye.get(f"{ladder}/{rung}", ("untested", ""))[1],
            }
        )
    points.sort(key=lambda p: (p["ladder"], p["lpips"]))  # each ladder from least visible up
    _plot(points, Path(out))
    return points


def write_h2(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    v = h2_verdict(args.ycbcr_run, args.rgb_run, args.ycbcr_lpips, args.rgb_lpips)
    points = []
    if args.chart_rungs:
        points = h2_chart(
            args.chart_rungs, args.eyetest_scores, out / "h2_chart.png", args.runs_root
        )
    (out / "h2.json").write_text(json.dumps({"verdict": v, "chart": points}, indent=2) + "\n")
    outcome = "supported" if v["supported"] else "not supported"
    t = v["tolerance"]
    lines = [
        "# H2: chroma noise against RGB noise",
        "",
        f"Runs: `{v['ycbcr_run']}` (YCbCr) and `{v['rgb_run']}` (RGB), same test images.",
        "95% bootstrap CIs: over images for recall (shared by both runs), over plates for LPIPS.",
        "",
        f"- Attacked {v['surrogate']} recall: YCbCr {_ci(v['ycbcr_recall'])}, "
        f"RGB {_ci(v['rgb_recall'])}",
        f"- Recall difference (YCbCr - RGB): {_ci(v['recall_difference'])}; "
        f"equivalent (CI inside +/-{t}): {'yes' if v['recall_equivalent'] else 'no'}",
        f"- Mean LPIPS over {v['n_plates_lpips']} plates scored in both: "
        f"YCbCr {_ci(v['ycbcr_lpips'])}, RGB {_ci(v['rgb_lpips'])}",
        f"- LPIPS difference (YCbCr - RGB): {_ci(v['lpips_difference'])}; "
        f"less visible: {'yes' if v['less_visible'] else 'no'}",
        "",
        f"**H2 {outcome}** by its written rule (recall within {t} and lower LPIPS).",
        "",
    ]
    if points:
        lines += [
            "## Visibility rungs (evidence the rule did not name)",
            "",
            "| Ladder | Rung | Mean LPIPS | Attacked recall | Eye test |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {p['ladder']} | {p['rung']} | {p['lpips']:.4f} | {p['recall']:.3f} "
            f"| {p['eye']} {p['eye_score']} |"
            for p in points
        ]
        lines += ["", "![H2 chart](h2_chart.png)", ""]
    (out / "h2.md").write_text("\n".join(lines))
    print(f"h2: wrote {out / 'h2.md'}")


def _plot(points: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ladders = list(dict.fromkeys(p["ladder"] for p in points))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # Labels above-right for the first ladder, below-left for the next, and so on, so the
    # labels of two close curves do not land on each other.
    offsets = [((5, 4), "left"), ((-5, -11), "right")]
    for i, (ladder, color) in enumerate(zip(ladders, colors, strict=False)):
        (dx, dy), align = offsets[i % 2]
        rung = sorted((p for p in points if p["ladder"] == ladder), key=lambda p: p["lpips"])
        ax.plot([p["lpips"] for p in rung], [p["recall"] for p in rung], color=color, lw=1)
        for p in rung:
            ax.scatter(p["lpips"], p["recall"], color=color, marker=EYE_MARKERS[p["eye"]], s=50)
            ax.annotate(
                p["rung"],
                (p["lpips"], p["recall"]),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=align,
                fontsize=8,
            )
        ax.plot([], [], color=color, label=ladder)
    present = {p["eye"] for p in points}
    for outcome, marker in EYE_MARKERS.items():
        if outcome not in present:
            continue
        ax.scatter([], [], color="gray", marker=marker, label=f"eye test: {outcome}")
    ax.set_xlabel("Mean LPIPS (higher = more visible)")
    ax.set_ylabel("Attacked YOLOv8s recall (lower = better hiding)")
    ax.set_title("Hiding power against visibility, per budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ci(ci: dict) -> str:
    return f"{ci['point']:.3f} [{ci['low']:.3f}, {ci['high']:.3f}]"
