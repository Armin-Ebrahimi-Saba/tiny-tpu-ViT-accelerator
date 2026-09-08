# ABOUTME: Enables `python -m sw <command>`.
# ABOUTME: All argument handling lives in sw/cli.py.

from .cli import main

raise SystemExit(main())
