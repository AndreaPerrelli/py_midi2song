"""Compatibility wrapper for running the CLI directly from a checkout."""
from src.midi2song import cli

if __name__ == "__main__":
    raise SystemExit(cli.main())
