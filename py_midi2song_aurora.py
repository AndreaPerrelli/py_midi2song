"""Compatibility wrapper for running the CLI directly from a checkout."""
from src.midi2song import aurora_runway

if __name__ == "__main__":
    raise SystemExit(aurora_runway.main())
