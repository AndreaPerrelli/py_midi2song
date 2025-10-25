"""Minimal tests for the :mod:`py_midi2song.cli` module.

These tests focus on small, side-effect free helpers so that the project is
testable inside the GitHub Actions workflow without requiring real MIDI
hardware or the external ``mido`` dependency to be installed.  A lightweight
stub for :mod:`mido` is provided to satisfy the imports used by the module
under test.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


# Ensure the source package is importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _install_mido_stub() -> None:
    """Install a very small stub of the :mod:`mido` package for the tests."""

    if "mido" in sys.modules:
        return

    mido = types.ModuleType("mido")

    class _BaseOutput:
        def __init__(self, name: str):
            self.name = name

        def send(self, message):  # pragma: no cover - the tests never call it
            self.last_message = message

    class _Message:
        def __init__(self, type: str, time: int = 0, **kwargs):
            self.type = type
            self.time = time
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _MetaMessage(_Message):
        pass

    def _open_output(name: str) -> _BaseOutput:
        return _BaseOutput(name)

    def _get_output_names():  # pragma: no cover - behaviour exercised indirectly
        return ["Dummy MIDI"]

    mido.Message = _Message
    mido.MetaMessage = _MetaMessage
    mido.open_output = _open_output
    mido.get_output_names = _get_output_names

    ports_module = types.SimpleNamespace(BaseOutput=_BaseOutput)
    mido.ports = ports_module

    sys.modules["mido"] = mido


_install_mido_stub()

from py_midi2song import cli


def test_parse_list_spec_variants():
    mode, values = cli._parse_list_spec("all")
    assert mode == "all" and values == set()

    mode, values = cli._parse_list_spec("include:0, 2,4")
    assert mode == "include" and values == {0, 2, 4}

    mode, values = cli._parse_list_spec("exclude:1,3")
    assert mode == "exclude" and values == {1, 3}


def test_apply_filters_tracks_and_channels():
    note_on = cli.Event(
        abs_ticks=0,
        time_s=0.0,
        message=sys.modules["mido"].Message("note_on", velocity=64),
        track_index=0,
        channel=1,
        _order=0,
    )
    note_off = cli.Event(
        abs_ticks=10,
        time_s=0.5,
        message=sys.modules["mido"].Message("note_off"),
        track_index=1,
        channel=2,
        _order=1,
    )
    control_change = cli.Event(
        abs_ticks=20,
        time_s=1.0,
        message=sys.modules["mido"].Message("control_change", control=7),
        track_index=0,
        channel=1,
        _order=2,
    )

    events = [note_on, note_off, control_change]

    # Channel filter: exclude channel 1, so only the note_off event survives.
    filtered = cli.apply_filters(events, tracks_spec="all", channels_spec="exclude:1", ignore_meta=False)
    assert filtered == [note_off]

    # Track filter: include only track 0, so events from track 1 are removed.
    filtered_tracks = cli.apply_filters(events, tracks_spec="include:0", channels_spec="all", ignore_meta=False)
    assert filtered_tracks == [note_on, control_change]


def test_seek_index_binary_search():
    events = [
        cli.Event(abs_ticks=i * 10, time_s=float(i), message=sys.modules["mido"].Message("noop"), track_index=0, channel=None, _order=i)
        for i in range(5)
    ]

    assert cli.seek_index(events, 0.0) == 0
    assert cli.seek_index(events, 2.5) == 3
    assert cli.seek_index(events, 10.0) == len(events)


def test_build_beat_grid_basic_downbeats():
    tempo_seg = cli.TempoSeg(start_tick=0, start_time_s=0.0, tempo_us_per_beat=500_000, end_tick=1920, end_time_s=2.0)
    timesig_seg = cli.TimeSigSeg(start_tick=0, start_time_s=0.0, numer=4, denom_pow2=2, end_tick=1920, end_time_s=2.0)

    beat_grid = cli.build_beat_grid([tempo_seg], [timesig_seg], ticks_per_beat=480)

    # Quarti (ticks_per_beat) -> 0.5 secondi ciascuno a 120 BPM
    assert beat_grid.beat_times[:4] == [0.0, 0.5, 1.0, 1.5]

    # Downbeat corrispondono agli inizi misura (0s, 2s)
    assert beat_grid.downbeat_times == {0.0, 2.0}

    # nearest_beat deve trovare il beat più vicino e la distanza
    t_beat, distance = beat_grid.nearest_beat(0.6)
    assert t_beat == 0.5 and abs(distance - 0.1) < 1e-9
