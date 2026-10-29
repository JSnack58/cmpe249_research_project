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


def not_implemented(args: argparse.Namespace) -> int:
    """Placeholder for stages that are not built yet."""
    # args.command holds the subcommand name because of dest="command" below.
    print(f"{args.command}: not implemented yet")
    return 1  # non-zero exit code = the stage did not succeed


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

    attack = sub.add_parser("attack", help="generate adversarial images on the surrogate (cached)")
    attack.set_defaults(func=not_implemented)

    evaluate = sub.add_parser(
        "evaluate", help="measure every target model on clean and attacked images"
    )
    evaluate.set_defaults(func=not_implemented)

    run = sub.add_parser("run", help="attack then evaluate")
    run.set_defaults(func=not_implemented)

    report = sub.add_parser("report", help="build the H1 table and charts from run folders")
    report.set_defaults(func=not_implemented)

    args = parser.parse_args(argv)
    return args.func(args)  # run the chosen stage; its return value becomes the exit code


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    """--config plus trailing key=value overrides, shared by every config-driven stage."""
    parser.add_argument("--config", type=Path, default=BASE_CONFIG, help="config or sweep file")
    parser.add_argument("overrides", nargs="*", help="config overrides such as attack.eps.cb=6")
