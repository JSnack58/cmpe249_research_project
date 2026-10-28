"""Makes `python -m plate_cloak` run the CLI."""

from plate_cloak.cli import main

# SystemExit carries main()'s return value back to the shell as the exit code.
raise SystemExit(main())
