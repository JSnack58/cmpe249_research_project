"""Command-line entry point: one subcommand per pipeline stage.

Usage: python -m plate_cloak <stage> [options]

Each stage reads its settings from a config file and hands its results to the next stage through
files on disk (see docs/superpowers/specs/2026-10-20-plate-cloak-experiment-design.md, section 6.5).
Stage-specific arguments are added in the task that implements each stage.
"""

import argparse
from pathlib import Path

from plate_cloak.config import BASE_CONFIG, load_config, resolve_path


def prepare_data(args: argparse.Namespace) -> int:
    """Download the dataset (skipping files already present) and freeze the eligible test list."""
    from plate_cloak.data.eval_images import write_eligible_list
    from plate_cloak.data.openimages import prepare_openimages

    cfg = load_config(args.config, args.overrides)
    root = resolve_path(cfg.data.root)
    prepare_openimages(root, workers=args.workers)
    n = write_eligible_list(
        root,
        resolve_path(cfg.data.eligible_list),
        cfg.data.imgsz,
        cfg.data.min_plate_w,
        cfg.data.min_plate_h,
    )
    print(f"eligible test images: {n} -> {cfg.data.eligible_list}")
    return 0


def finetune(args: argparse.Namespace) -> int:
    """Fine-tune one model with one seed; epochs/batch/workers come from the config."""
    from plate_cloak.models.finetune import finetune as run_finetune

    cfg = load_config(args.config, args.overrides)
    run_finetune(
        name=args.model,
        seed=args.seed,
        data_yaml=resolve_path(cfg.data.root) / "data.yaml",
        epochs=args.epochs if args.epochs is not None else cfg.models.epochs,
        batch=cfg.models.batch,
        workers=cfg.models.workers,
        device=args.device,
        timing=args.timing,
    )
    return 0


def attack(args: argparse.Namespace) -> int:
    """Make (or reuse) the cached adversarial images for one config or sweep variant."""
    from plate_cloak.attack.stage import run_attack

    run_attack(load_config(args.config, args.overrides, variant=args.variant))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    """Score every target on the cached attack for this config; the cache must already exist."""
    from plate_cloak.attack.stage import fingerprint
    from plate_cloak.data.eval_images import eval_image_ids
    from plate_cloak.evaluation.stage import evaluate as run_evaluate

    cfg = load_config(args.config, args.overrides, variant=args.variant)
    adv_dir = resolve_path(cfg.output.cache_root) / fingerprint(cfg, eval_image_ids(cfg))
    if not (adv_dir / "done.json").exists():
        variant = f" --variant {args.variant}" if args.variant else ""
        raise FileNotFoundError(
            f"no finished attack cache at {adv_dir}; first run: "
            f"python -m plate_cloak attack --config {args.config}{variant} (same overrides)"
        )
    run_evaluate(cfg, adv_dir, _run_dir(cfg, args.name))
    return 0


def run(args: argparse.Namespace) -> int:
    """Attack (or reuse the cache), then evaluate, for one config or sweep variant."""
    from plate_cloak.attack.stage import run_attack
    from plate_cloak.evaluation.stage import evaluate as run_evaluate

    cfg = load_config(args.config, args.overrides, variant=args.variant)
    run_dir = _run_dir(cfg, args.name)
    if run_dir.exists():  # fail before the attack, which can take hours
        raise FileExistsError(f"{run_dir} already exists; pick another --name")
    run_evaluate(cfg, run_attack(cfg), run_dir)
    return 0


def report(args: argparse.Namespace) -> int:
    """Build the H1 table, verdict and chart from one or more run folders."""
    from plate_cloak.analysis.report import report as run_report

    run_report(args.runs, args.out)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to the chosen stage.

    argv is None when run from a terminal (argparse then reads the real command line);
    tests pass a list such as ["--help"] to drive the CLI directly.
    """
    parser = argparse.ArgumentParser(prog="plate-cloak")
    # A slot for subcommands; required=True makes a bare `plate-cloak` an error instead of a no-op.
    sub = parser.add_subparsers(dest="command", required=True)

    # One parser per stage. help= is the text shown by --help.
    # set_defaults(func=...) attaches the function that runs the stage, so main() can dispatch
    # with args.func(args) instead of an if/elif chain on the stage name.
    p_prepare = sub.add_parser(
        "prepare-data", help="download Open Images plates and freeze the eligible test list"
    )
    _add_config_args(p_prepare)
    p_prepare.add_argument("--workers", type=int, default=32, help="parallel downloads")
    p_prepare.set_defaults(func=prepare_data)

    from plate_cloak.models.zoo import ZOO

    p_finetune = sub.add_parser("finetune", help="fine-tune one detector on the plate dataset")
    _add_config_args(p_finetune)
    p_finetune.add_argument("--model", required=True, choices=sorted(ZOO))
    p_finetune.add_argument("--seed", required=True, type=int)
    p_finetune.add_argument("--epochs", type=int, help="override models.epochs (e.g. 2 for timing)")
    p_finetune.add_argument("--device", default="0", help="GPU index for Ultralytics, or 'cpu'")
    p_finetune.add_argument(
        "--timing", action="store_true", help="throwaway speed test in runs/timing/, never exported"
    )
    p_finetune.set_defaults(func=finetune)

    p_attack = sub.add_parser(
        "attack", help="generate adversarial images on the surrogate (cached)"
    )
    _add_config_args(p_attack)
    _add_variant_arg(p_attack)
    p_attack.set_defaults(func=attack)

    p_evaluate = sub.add_parser(
        "evaluate", help="measure every target model on clean and attacked images"
    )
    _add_run_args(p_evaluate)
    p_evaluate.set_defaults(func=evaluate)

    p_run = sub.add_parser("run", help="attack then evaluate")
    _add_run_args(p_run)
    p_run.set_defaults(func=run)

    p_report = sub.add_parser("report", help="build the H1 table and charts from run folders")
    p_report.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders")
    p_report.add_argument("--out", type=Path, required=True, help="folder for the report files")
    p_report.set_defaults(func=report)

    args = parser.parse_args(argv)
    return args.func(args)  # run the chosen stage; its return value becomes the exit code


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """--config plus trailing key=value overrides, shared by every config-driven stage."""
    parser.add_argument("--config", type=Path, default=BASE_CONFIG, help="config or sweep file")
    parser.add_argument("overrides", nargs="*", help="config overrides such as attack.eps.cb=6")


def _add_variant_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", help="variant name, when --config is a sweep file")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Options for stages that write a run folder: config, variant, and the run's name."""
    _add_config_args(parser)
    _add_variant_arg(parser)
    parser.add_argument(
        "--name", help="run folder name (default: timestamp plus settings that differ from base)"
    )


def _run_dir(cfg, name: str | None) -> Path:
    """results/runs/<name>, with a generated name when none is given."""
    from datetime import datetime

    from plate_cloak.runs import run_name

    if name is None:
        name = run_name(cfg, load_config(BASE_CONFIG), datetime.now())
    return resolve_path(cfg.output.runs_root) / name
