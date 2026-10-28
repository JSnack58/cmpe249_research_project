"""Command-line entry point: one subcommand per pipeline stage.

Usage: python -m plate_cloak <stage> [options]

Each stage reads its settings from a config file and hands its results to the next stage through
files on disk (see docs/superpowers/specs/2026-10-20-plate-cloak-experiment-design.md, section 6.5).
Stage-specific arguments are added in the task that implements each stage.
"""

import argparse


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
    prepare_data = sub.add_parser(
        "prepare-data", help="download Open Images plates and freeze the eligible test list"
    )
    prepare_data.set_defaults(func=not_implemented)

    finetune = sub.add_parser("finetune", help="fine-tune one detector on the plate dataset")
    finetune.set_defaults(func=not_implemented)

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
