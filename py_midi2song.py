"""Compatibility wrapper for running the CLI directly from a checkout."""
from py_midi2song.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
