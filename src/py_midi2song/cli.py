#!/usr/bin/env python3
"""Command line interface for :mod:`py_midi2song`.

The tool plays ``.mid`` / ``.midi`` files through any MIDI OUT port provided by
the operating system, with optional real-time textual visualisations and
utilities such as filtering tracks/channels, tempo scaling, gain adjustment and
metronome overlays.

Example usage::

    py-midi2song --list-ports
    py-midi2song --midi song.mid
    py-midi2song --midi song.mid --port "Microsoft GS"
    py-midi2song --midi song.mid --start-at-seconds 12.5 --tempo-scale 0.9
    py-midi2song --midi song.mid --tracks include:0,2 --channels exclude:9

Visualisations::

    py-midi2song --midi song.mid --viz --viz-mode grid --viz-beats
    py-midi2song --midi song.mid --viz --viz-mode lanes --viz-lanes 7 \
        --viz-window-seconds 4 --viz-height 22 --viz-beats
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import threading
import os
import ctypes
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set

import mido


# ==========================
# Console helpers (Windows)
# ==========================

def _enable_win_vt_mode() -> None:
    """Abilita VT sequences su Windows 10+."""
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(h, new_mode)
    except Exception:
        pass


class Terminal:
    """Renderer a buffer singolo: meno flicker, hide/show cursore, reposition."""
    def __init__(self):
        self._cursor_hidden = False
        _enable_win_vt_mode()

    def hide_cursor(self):
        if not self._cursor_hidden:
            sys.stdout.write("\x1b[?25l")
            self._cursor_hidden = True

    def show_cursor(self):
        if self._cursor_hidden:
            sys.stdout.write("\x1b[?25h")
            self._cursor_hidden = False

    def draw(self, buffer_text: str):
        # Reposition to home and output a single buffer write
        sys.stdout.write("\x1b[H")
        sys.stdout.write(buffer_text)
        sys.stdout.flush()

    def clear_full(self):
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()


# ========= Dataclass evento / segmenti / tempo ========

@dataclass
class Event:
    abs_ticks: int
    time_s: float
    message: mido.Message | mido.MetaMessage
    track_index: int
    channel: Optional[int]
    _order: int

@dataclass
class NoteSegment:
    t0: float
    t1: float
    note: int
    channel: int

@dataclass
class TempoSeg:
    start_tick: int
    start_time_s: float
    tempo_us_per_beat: int
    end_tick: Optional[int] = None
    end_time_s: Optional[float] = None

@dataclass
class TimeSigSeg:
    start_tick: int
    start_time_s: float
    numer: int
    denom_pow2: int  # denominator = 2**denom_pow2
    end_tick: Optional[int] = None
    end_time_s: Optional[float] = None


# ========= Porte MIDI =========

def list_output_ports() -> List[str]:
    try:
        ports = mido.get_output_names()
    except Exception as e:
        logging.error("Errore nel recupero delle porte MIDI: %s", e)
        return []
    return ports


def select_output_port(spec: Optional[str]) -> mido.ports.BaseOutput:
    ports = list_output_ports()
    if not ports:
        raise RuntimeError(
            "Nessuna porta MIDI OUT disponibile. Su Windows, abilita 'Microsoft GS Wavetable Synth' "
            "oppure usa un dispositivo virtuale (es. loopMIDI) collegato a un synth software."
        )

    def open_by_name(name: str) -> mido.ports.BaseOutput:
        return mido.open_output(name)

    if spec is None or str(spec).strip() == "":
        preferred = next((p for p in ports if "microsoft gs wavetable" in p.lower()), None)
        target = preferred or ports[0]
        logging.info("Porta selezionata (auto): %s", target)
        return open_by_name(target)

    try:
        idx = int(spec)
        if idx < 0 or idx >= len(ports):
            raise IndexError
        logging.info("Porta selezionata (indice %d): %s", idx, ports[idx])
        return open_by_name(ports[idx])
    except (ValueError, IndexError):
        pass

    spec_low = spec.lower()
    matches = [p for p in ports if spec_low in p.lower()]
    if not matches:
        raise RuntimeError(f"Nessuna porta corrisponde a '{spec}'. Usa --list-ports per l'elenco.")
    logging.info("Porta selezionata (match '%s'): %s", spec, matches[0])
    return open_by_name(matches[0])


# ========= Parsing & Timeline (con segmenti tempo/timesig) =========

def _collect_events_with_abs_ticks(mid: mido.MidiFile) -> List[Tuple[int, mido.Message | mido.MetaMessage, int, Optional[int], int]]:
    collected: List[Tuple[int, mido.Message | mido.MetaMessage, int, Optional[int], int]] = []
    order = 0
    for ti, track in enumerate(mid.tracks):
        abs_ticks = 0
        for msg in track:
            abs_ticks += msg.time
            ch = getattr(msg, 'channel', None)
            collected.append((abs_ticks, msg, ti, ch, order))
            order += 1
    return collected


def _compute_times_seconds_and_segments(
    events: List[Tuple[int, mido.Message | mido.MetaMessage, int, Optional[int], int]],
    ticks_per_beat: int,
):
    """Ritorna (events_ts, tempo_segments, timesig_segments) con time_s e segmenti per metronomo."""
    def prio(e: Tuple[int, mido.Message | mido.MetaMessage, int, Optional[int], int]) -> Tuple[int, int, int]:
        abs_ticks, msg, ti, ch, order = e
        # set_tempo e time_signature prima degli altri al medesimo tick
        t = getattr(msg, 'type', '')
        is_ctrl = 1
        if t == 'set_tempo':
            is_ctrl = 0
        elif t == 'time_signature':
            is_ctrl = 0
        return (abs_ticks, is_ctrl, order)

    events_sorted = sorted(events, key=prio)

    current_tempo = 500000  # 120 BPM
    last_tick = 0
    current_time_s = 0.0
    out_events: List[Event] = []

    tempo_segments: List[TempoSeg] = [TempoSeg(start_tick=0, start_time_s=0.0, tempo_us_per_beat=current_tempo)]
    timesig_segments: List[TimeSigSeg] = [TimeSigSeg(start_tick=0, start_time_s=0.0, numer=4, denom_pow2=2)]  # 4/4 default

    for abs_ticks, msg, track_index, channel, order in events_sorted:
        delta_ticks = abs_ticks - last_tick
        if delta_ticks < 0:
            delta_ticks = 0
        current_time_s += (delta_ticks * current_tempo) / 1_000_000.0 / ticks_per_beat
        last_tick = abs_ticks

        out_events.append(Event(abs_ticks, current_time_s, msg, track_index, channel, order))

        t = getattr(msg, 'type', '')
        if t == 'set_tempo':
            # chiudi seg tempo corrente
            tempo_segments[-1].end_tick = abs_ticks
            tempo_segments[-1].end_time_s = current_time_s
            # nuovo seg
            current_tempo = msg.tempo
            tempo_segments.append(TempoSeg(start_tick=abs_ticks, start_time_s=current_time_s, tempo_us_per_beat=current_tempo))
        elif t == 'time_signature':
            # chiudi seg time sig corrente
            timesig_segments[-1].end_tick = abs_ticks
            timesig_segments[-1].end_time_s = current_time_s
            numer = getattr(msg, 'numerator', 4)
            denom_pow2 = getattr(msg, 'denominator', 2)  # mido already stores power-of-two exponent
            timesig_segments.append(TimeSigSeg(start_tick=abs_ticks, start_time_s=current_time_s, numer=numer, denom_pow2=denom_pow2))

    # chiudi segmenti finali
    tempo_segments[-1].end_tick = last_tick
    tempo_segments[-1].end_time_s = current_time_s
    timesig_segments[-1].end_tick = last_tick
    timesig_segments[-1].end_time_s = current_time_s

    return out_events, tempo_segments, timesig_segments


def build_timeline(mid: mido.MidiFile, tempo_scale: float):
    if tempo_scale <= 0:
        raise ValueError("--tempo-scale deve essere > 0")
    raw = _collect_events_with_abs_ticks(mid)
    events, tempo_segs, ts_segs = _compute_times_seconds_and_segments(raw, mid.ticks_per_beat)
    # tempo scale globale: dilata i tempi in secondi
    for ev in events:
        ev.time_s = ev.time_s / tempo_scale
    for seg in tempo_segs:
        seg.start_time_s /= tempo_scale
        seg.end_time_s = (seg.end_time_s or 0.0) / tempo_scale
        # Nota: il valore di tempo_us_per_beat resta, ma la scala globale agisce sui time_s
    for seg in ts_segs:
        seg.start_time_s /= tempo_scale
        seg.end_time_s = (seg.end_time_s or 0.0) / tempo_scale
    return events, tempo_segs, ts_segs


# ========= Beat grid (metronomo) =========

@dataclass
class BeatGrid:
    beat_times: List[float]          # tutti i quarti (o base T PB) in secondi
    downbeat_times: Set[float]       # inizio battute (approssimato)
    ticks_per_beat: int

    def nearest_beat(self, t: float) -> Tuple[float, float]:
        """Ritorna (t_beat, distanza_abs). Ricerca lineare rapida con bisect."""
        import bisect
        i = bisect.bisect_left(self.beat_times, t)
        cand = []
        if i < len(self.beat_times):
            cand.append(self.beat_times[i])
        if i > 0:
            cand.append(self.beat_times[i-1])
        if not cand:
            return (0.0, float('inf'))
        best = min(cand, key=lambda x: abs(x - t))
        return (best, abs(best - t))

    def is_downbeat_near(self, t: float, tol: float) -> bool:
        # check con tolleranza su insiemi densi: logico trasformare in ricerca su beat_times e passo misura
        tb, dist = self.nearest_beat(t)
        if dist > tol:
            return False
        # considera downbeat: quelli più vicini tra downbeat_times
        # per efficienza, stimiamo: se tb in downbeat_times (con tol)
        for db in (tb,):
            # controllo diretto
            if any(abs(db - x) <= tol for x in self.downbeat_times):
                return True
        return False


def build_beat_grid(
    tempo_segs: List[TempoSeg],
    ts_segs: List[TimeSigSeg],
    ticks_per_beat: int,
) -> BeatGrid:
    """
    Genera tutti i beat (tick multipli di ticks_per_beat) e downbeat (inizio battuta secondo timesig).
    La mappa misura cambia ai time_signature; ogni cambio definisce nuovo allineamento.
    """
    beat_times: List[float] = []

    # Pre-calcola mapping tick->time per ogni tempo_seg (lineare su tick)
    # Per ogni seg: t = t0 + (tick - tick0) * tempo / 1e6 / TPB
    def time_at(seg: TempoSeg, tick: int) -> float:
        dtick = tick - seg.start_tick
        return seg.start_time_s + (dtick * seg.tempo_us_per_beat) / 1_000_000.0 / ticks_per_beat

    # Genera quarti (ticks_per_beat)
    for seg in tempo_segs:
        start_tick = seg.start_tick
        end_tick = seg.end_tick or start_tick
        k_start = (start_tick + ticks_per_beat - 1) // ticks_per_beat  # ceil
        k_end = (end_tick) // ticks_per_beat
        for k in range(k_start, k_end + 1):
            tick = k * ticks_per_beat
            if tick < start_tick or tick > end_tick:
                continue
            beat_times.append(time_at(seg, tick))

    # Genera downbeat (inizio battuta)
    downbeats: Set[float] = set()
    # Per ogni time_sig segment: definisci beats per misura = numerator
    # e allinea la prima misura al suo start_tick (approx arrotondando al beat più vicino).
    import math
    for tss in ts_segs:
        start_tick = tss.start_tick
        end_tick = tss.end_tick or start_tick
        beats_per_measure = max(1, tss.numer)
        # scegli seg tempo attivo a start_tick
        seg = next((s for s in tempo_segs if s.start_tick <= start_tick <= (s.end_tick or s.start_tick)), tempo_segs[0])
        # allinea a beat boundary più vicino
        k0 = round(start_tick / ticks_per_beat)
        tick0 = int(k0 * ticks_per_beat)
        # Propaga i successivi inizi misura
        tick = tick0
        while tick <= end_tick:
            downbeats.add(round(time_at(seg, tick), 6))  # arrotonda per robustezza confronto float
            tick += beats_per_measure * ticks_per_beat

    # Arrotonda beat_times per confronti robusti
    beat_times = sorted(round(x, 6) for x in beat_times)

    return BeatGrid(beat_times=beat_times, downbeat_times=downbeats, ticks_per_beat=ticks_per_beat)


# ========= Filtri =========

def _parse_list_spec(spec: str) -> Tuple[str, Set[int]]:
    s = spec.strip().lower()
    if s == "all":
        return ("all", set())
    if s.startswith("include:"):
        values = s[len("include:"):].strip()
        items = {int(x.strip()) for x in values.split(",") if x.strip() != ""}
        return ("include", items)
    if s.startswith("exclude:"):
        values = s[len("exclude:"):].strip()
        items = {int(x.strip()) for x in values.split(",") if x.strip() != ""}
        return ("exclude", items)
    raise ValueError(f"Spec non valida: '{spec}' (usa 'all', 'include:CSV' o 'exclude:CSV').")


def _is_channel_message(msg: mido.Message | mido.MetaMessage) -> bool:
    t = getattr(msg, 'type', '')
    return t in {
        'note_on', 'note_off', 'control_change', 'program_change',
        'pitchwheel', 'aftertouch', 'polytouch', 'channel_pressure'
    }


def apply_filters(
    events: List[Event],
    tracks_spec: str,
    channels_spec: str,
    ignore_meta: bool
) -> List[Event]:
    track_mode, track_set = _parse_list_spec(tracks_spec)
    ch_mode, ch_set = _parse_list_spec(channels_spec)

    def track_ok(ti: int) -> bool:
        if track_mode == "all":
            return True
        if track_mode == "include":
            return ti in track_set
        return ti not in track_set

    def channel_ok(ch: Optional[int], msg: mido.Message | mido.MetaMessage) -> bool:
        if not _is_channel_message(msg):
            return True
        if ch is None:
            return False
        if ch_mode == "all":
            return True
        if ch_mode == "include":
            return ch in ch_set
        return ch not in ch_set

    filtered = []
    for ev in events:
        if not track_ok(ev.track_index):
            continue
        if not channel_ok(ev.channel, ev.message):
            continue
        filtered.append(ev)
    return filtered


# ========= Seek & Stato Canali =========

def seek_index(events: List[Event], t_start: float) -> int:
    lo, hi = 0, len(events)
    while lo < hi:
        mid = (lo + hi) // 2
        if events[mid].time_s < t_start:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _gather_channel_state_before(
    events: List[Event],
    t_start: float
) -> Dict[int, Dict[str, Dict]]:
    state: Dict[int, Dict[str, Dict]] = {}
    for ev in events:
        if ev.time_s >= t_start:
            break
        msg = ev.message
        if not _is_channel_message(msg):
            continue
        ch = ev.channel
        if ch is None:
            continue
        if ch not in state:
            state[ch] = {'program': None, 'controls': {}, 'pitch': None, 'pressure': None}
        t = msg.type
        if t == 'program_change':
            state[ch]['program'] = msg.program
        elif t == 'control_change':
            state[ch]['controls'][msg.control] = msg.value
        elif t == 'pitchwheel':
            state[ch]['pitch'] = msg.pitch
        elif t in ('aftertouch', 'channel_pressure'):
            state[ch]['pressure'] = getattr(msg, 'value', None)
    return state


def _send_channel_state(out_port: mido.ports.BaseOutput, state: Dict[int, Dict[str, Dict]]) -> None:
    for ch, st in state.items():
        if st.get('program') is not None:
            out_port.send(mido.Message('program_change', channel=ch, program=st['program']))
        for cc, val in st.get('controls', {}).items():
            out_port.send(mido.Message('control_change', channel=ch, control=cc, value=val))
        if st.get('pitch') is not None:
            out_port.send(mido.Message('pitchwheel', channel=ch, pitch=st['pitch']))
        if st.get('pressure') is not None:
            out_port.send(mido.Message('aftertouch', channel=ch, value=st['pressure']))


# ========= Visualizzazione GRID 4×4 =========

class PadGrid4x4:
    def __init__(self, base_note: int = 60):
        self.base_note = base_note

    def note_to_pad(self, note: int) -> Optional[int]:
        if note < 0 or note > 127:
            return None
        base = self.base_note
        top = base + 15
        n = note
        while n < base:
            n += 12
        while n > top:
            n -= 12
        idx = n - base
        return idx if 0 <= idx <= 15 else None


class GridVisualizer(threading.Thread):
    def __init__(self, term: Terminal, grid: PadGrid4x4, refresh_hz: float = 30.0, beatgrid: Optional[BeatGrid] = None, show_beats: bool = False):
        super().__init__(daemon=True)
        self.term = term
        self.grid = grid
        self.refresh_dt = max(1.0 / max(refresh_hz, 1.0), 0.01)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._active_pads: Set[int] = set()
        self._status_lines: List[str] = []
        self._t_cur: float = 0.0
        self._t_end: Optional[float] = None
        self.beatgrid = beatgrid
        self.show_beats = show_beats

    def update_active_note(self, note: int, on: bool) -> None:
        idx = self.grid.note_to_pad(note)
        if idx is None:
            return
        with self._lock:
            if on:
                self._active_pads.add(idx)
            else:
                self._active_pads.discard(idx)

    def set_status(self, lines: List[str]) -> None:
        with self._lock:
            self._status_lines = lines[:]

    def set_time_window(self, t_cur: float, t_end: Optional[float]) -> None:
        with self._lock:
            self._t_cur = t_cur
            self._t_end = t_end

    def stop(self) -> None:
        self._stop.set()

    def _render_to_buffer(self) -> str:
        with self._lock:
            pads = set(self._active_pads)
            status = list(self._status_lines)
            t_cur = self._t_cur
            t_end = self._t_end

        buf: List[str] = []
        buf.append("MIDI Player — Grid 4×4 (Ctrl+C per uscire)")
        for line in status:
            buf.append(line)

        # Metronomo / tempo bar
        if self.show_beats and self.beatgrid is not None:
            tbeat, dist = self.beatgrid.nearest_beat(t_cur)
            tick = "●" if dist <= 0.07 else "○"
            # Stima conteggio 1..4 usando beat index % 4
            import bisect
            i = bisect.bisect_left(self.beatgrid.beat_times, tbeat)
            count = (i % 4) + 1
            buf.append(f"Metronomo: {tick}  (beat {count})  t={t_cur:6.2f}s")
        else:
            if t_end:
                width = 40
                pos = int(max(0.0, min(1.0, t_cur / t_end)) * width)
                buf.append("Tempo: [" + "#" * pos + "-" * (width - pos) + "]")
            else:
                buf.append(f"Tempo: t = {t_cur:.2f}s")

        buf.append("")

        base = self.grid.base_note
        for r in range(4):
            row_cells = []
            for c in range(4):
                idx = (3 - r) * 4 + c
                note = base + idx
                active = (idx in pads)
                label = f"{note:3d}"
                cell = f" {label} "
                row_cells.append(("\x1b[7m\x1b[1m" + cell + "\x1b[0m") if active else "[" + label + "]")
            buf.append("  " + " ".join(row_cells))
        buf.append("")
        return "\n".join(buf)

    def run(self) -> None:
        self.term.hide_cursor()
        self.term.clear_full()
        last_draw = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_draw >= self.refresh_dt:
                self.term.draw(self._render_to_buffer())
                last_draw = now
            time.sleep(0.005)
        self.term.clear_full()
        self.term.show_cursor()


# ========= Visualizzazione LANES =========

def _pair_note_segments(events: List[Event], t_end: Optional[float]) -> List[NoteSegment]:
    last_time = events[-1].time_s if events else 0.0
    t_final = t_end if t_end is not None else last_time
    active: Dict[Tuple[int, int], float] = {}
    segments: List[NoteSegment] = []
    for ev in events:
        msg = ev.message
        if not isinstance(msg, mido.Message):
            continue
        if msg.type == 'note_on' and msg.velocity > 0 and ev.channel is not None:
            key = (ev.channel, msg.note)
            if key in active:
                t0 = active.pop(key)
                segments.append(NoteSegment(t0=t0, t1=ev.time_s, note=msg.note, channel=ev.channel))
            active[key] = ev.time_s
        elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and ev.channel is not None:
            key = (ev.channel, msg.note)
            t0 = active.pop(key, None)
            if t0 is not None and ev.time_s > t0:
                segments.append(NoteSegment(t0=t0, t1=ev.time_s, note=msg.note, channel=ev.channel))
    for (ch, note), t0 in active.items():
        t1 = max(t0, t_final)
        segments.append(NoteSegment(t0=t0, t1=t1, note=note, channel=ch))
    return segments


class LanesVisualizer(threading.Thread):
    def __init__(
        self,
        term: Terminal,
        segments: List[NoteSegment],
        lanes: int = 7,
        refresh_hz: float = 30.0,
        window_seconds: float = 4.0,
        height: int = 20,
        base_note: int = 60,
        beatgrid: Optional[BeatGrid] = None,
        show_beats: bool = False,
    ):
        super().__init__(daemon=True)
        self.term = term
        self.segments = segments
        self.lanes = lanes
        self.refresh_dt = max(1.0 / max(refresh_hz, 1.0), 0.01)
        self.window = max(window_seconds, 0.5)
        self.height = max(height, 8)
        self.base_note = base_note
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status_lines: List[str] = []
        self._t_cur: float = 0.0
        self._t_end: Optional[float] = None
        self.beatgrid = beatgrid
        self.show_beats = show_beats

    def set_status(self, lines: List[str]) -> None:
        with self._lock:
            self._status_lines = lines[:]

    def set_time_window(self, t_cur: float, t_end: Optional[float]) -> None:
        with self._lock:
            self._t_cur = t_cur
            self._t_end = t_end

    def stop(self) -> None:
        self._stop.set()

    def _lane_for_note(self, note: int) -> int:
        norm = (note - self.base_note) % 12
        return norm % self.lanes

    def _render_to_buffer(self) -> str:
        with self._lock:
            status = list(self._status_lines)
            t_cur = self._t_cur
            t_end = self._t_end

        buf: List[str] = []
        header = f"MIDI Player — Lanes ({self.lanes}) (Ctrl+C per uscire)"
        if self.show_beats and self.beatgrid is not None:
            tbeat, dist = self.beatgrid.nearest_beat(t_cur)
            tick = "●" if dist <= 0.07 else "○"
            import bisect
            i = bisect.bisect_left(self.beatgrid.beat_times, tbeat)
            count = (i % 4) + 1
            header += f"   Metronomo: {tick} (beat {count})"
        buf.append(header)
        for line in status:
            buf.append(line)

        if t_end is not None and t_end > 0:
            width = 50
            pos = int(max(0.0, min(1.0, t_cur / t_end)) * width)
            buf.append("Tempo: [" + "#" * pos + "-" * (width - pos) + f"]  t={t_cur:6.2f}s")
        else:
            buf.append(f"Tempo: t = {t_cur:.2f}s")
        buf.append("")

        H = self.height
        W = self.lanes
        dt_row = self.window / (H - 1) if H > 1 else self.window

        # header colonne lane
        buf.append("    " + " ".join(f"{i}" for i in range(W)))

        for r in range(H):
            t_row = t_cur + self.window * (1 - (r / (H - 1))) if H > 1 else t_cur
            row = [" "] * W

            # Beat grid
            if self.show_beats and self.beatgrid is not None:
                # Se vicino a un beat -> ":" ; se downbeat -> "|"
                t_near, d = self.beatgrid.nearest_beat(t_row)
                if d <= dt_row * 0.5:
                    # downbeat?
                    is_down = self.beatgrid.is_downbeat_near(t_row, tol=dt_row * 0.6)
                    mark = "|" if is_down else ":"
                    # disegna su tutte le lane in background
                    row = [mark] * W

            # Note segments
            t0_bin = t_row - dt_row * 0.5
            t1_bin = t_row + dt_row * 0.5
            for seg in self.segments:
                if seg.t1 < t0_bin or seg.t0 > t1_bin:
                    continue
                lane = self._lane_for_note(seg.note)
                # scegliere char in base alla distanza da inizio/fine segmento
                center = t_row
                char = "█"
                if abs(center - seg.t0) <= dt_row * 0.6:
                    char = "O"
                if abs(center - seg.t1) <= dt_row * 0.6:
                    char = "o"
                row[lane] = char  # nota vince sul marker beat

            prefix = "=> " if r == H - 1 else "   "
            buf.append(prefix + " " + " ".join(row))

        buf.append("")
        return "\n".join(buf)

    def run(self) -> None:
        self.term.hide_cursor()
        self.term.clear_full()
        last_draw = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_draw >= self.refresh_dt:
                self.term.draw(self._render_to_buffer())
                last_draw = now
            time.sleep(0.005)
        self.term.clear_full()
        self.term.show_cursor()


# ========= Riproduzione =========

def _clamp_velocity(vel: int, gain: float) -> int:
    if vel <= 0:
        return 0
    v = int(round(vel * gain))
    if v < 1:
        v = 1
    if v > 127:
        v = 127
    return v


def _sleep_until(target_monotonic: float) -> None:
    while True:
        now = time.monotonic()
        dt = target_monotonic - now
        if dt <= 0:
            return
        if dt > 0.01:
            time.sleep(dt - 0.005)
        else:
            time.sleep(0)


def _is_meta(msg) -> bool:
    return isinstance(msg, mido.MetaMessage)


def _all_notes_off(out_port: mido.ports.BaseOutput, used_channels: Set[int]) -> None:
    for ch in sorted(used_channels):
        out_port.send(mido.Message('control_change', channel=ch, control=64, value=0))
        out_port.send(mido.Message('control_change', channel=ch, control=123, value=0))
        out_port.send(mido.Message('control_change', channel=ch, control=120, value=0))


def play_events(
    events: List[Event],
    start_idx: int,
    t_start: float,
    t_end: Optional[float],
    out_port: mido.ports.BaseOutput,
    gain: float,
    ignore_meta: bool,
    logger: logging.Logger,
    grid_visual: Optional[GridVisualizer] = None,
    lanes_visual: Optional[LanesVisualizer] = None,
) -> None:
    active_notes: Dict[Tuple[int, int], bool] = {}
    used_channels: Set[int] = set()

    state = _gather_channel_state_before(events, t_start)
    _send_channel_state(out_port, state)

    start_mono = time.monotonic()
    end_idx_limit = None
    if t_end is not None:
        end_idx_limit = seek_index(events, t_end)

    logger.info("Riproduzione: start=%.3fs%s", t_start, f", end={t_end:.3f}s" if t_end is not None else "")

    try:
        i = start_idx
        while i < len(events) and (end_idx_limit is None or i < end_idx_limit):
            ev = events[i]
            target_rel = ev.time_s - t_start
            if target_rel < 0:
                i += 1
                continue
            target_abs = start_mono + target_rel

            if grid_visual is not None:
                grid_visual.set_time_window(target_rel + t_start, t_end if t_end is not None else None)
            if lanes_visual is not None:
                lanes_visual.set_time_window(target_rel + t_start, t_end if t_end is not None else None)

            _sleep_until(target_abs)

            msg = ev.message
            if ignore_meta and _is_meta(msg):
                i += 1
                continue
            if _is_meta(msg):
                i += 1
                continue

            mtype = msg.type
            if mtype == 'note_on':
                used_channels.add(msg.channel)
                if msg.velocity <= 0:
                    key = (msg.channel, msg.note)
                    if active_notes.pop(key, None):
                        out_port.send(mido.Message('note_off', channel=msg.channel, note=msg.note, velocity=0))
                        if grid_visual is not None:
                            grid_visual.update_active_note(msg.note, False)
                else:
                    vel = _clamp_velocity(msg.velocity, gain)
                    out_port.send(mido.Message('note_on', channel=msg.channel, note=msg.note, velocity=vel))
                    active_notes[(msg.channel, msg.note)] = True
                    if grid_visual is not None:
                        grid_visual.update_active_note(msg.note, True)
            elif mtype == 'note_off':
                used_channels.add(msg.channel)
                key = (msg.channel, msg.note)
                if active_notes.pop(key, None):
                    out_port.send(msg.copy(velocity=0))
                    if grid_visual is not None:
                        grid_visual.update_active_note(msg.note, False)
            elif mtype == 'control_change':
                used_channels.add(msg.channel)
                out_port.send(msg)
            elif mtype == 'program_change':
                used_channels.add(msg.channel)
                out_port.send(msg)
            elif mtype == 'pitchwheel':
                used_channels.add(msg.channel)
                out_port.send(msg)
            elif mtype in ('aftertouch', 'channel_pressure', 'polytouch'):
                used_channels.add(msg.channel)
                out_port.send(msg)
            else:
                if hasattr(msg, 'channel'):
                    used_channels.add(msg.channel)
                out_port.send(msg)

            i += 1

    except KeyboardInterrupt:
        logger.warning("Interruzione richiesta (Ctrl+C). Sto spegnendo le note...")
    finally:
        for (ch, note) in list(active_notes.keys()):
            out_port.send(mido.Message('note_off', channel=ch, note=note, velocity=0))
            if grid_visual is not None:
                grid_visual.update_active_note(note, False)
        _all_notes_off(out_port, used_channels)


# ========= CLI / main =========

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Riproduci file MIDI (.mid/.midi) su una porta MIDI OUT di sistema, con visualizzazioni opzionali e beat grid."
    )
    parser.add_argument("--midi", required=False, help="Percorso al file MIDI da riprodurre.")
    parser.add_argument("--list-ports", action="store_true", help="Elenca le porte MIDI OUT disponibili e termina.")
    parser.add_argument("--port", help="Nome parziale (case-insensitive) o indice della porta MIDI OUT.")
    parser.add_argument("--start-at-seconds", type=float, default=0.0, help="Avvio riproduzione a t (s).")
    parser.add_argument("--end-at-seconds", type=float, default=None, help="Termina riproduzione a t (s).")
    parser.add_argument("--tempo-scale", type=float, default=1.0, help="Scala globale del tempo. 0.8 = più lento.")
    parser.add_argument("--gain", type=float, default=1.0, help="Moltiplicatore velocity (clamp 1..127).")
    parser.add_argument("--tracks", default="all", help="Filtra tracce: 'all' | 'include:CSV' | 'exclude:CSV'.")
    parser.add_argument("--channels", default="all", help="Filtra canali: 'all' | 'include:CSV' | 'exclude:CSV'.")
    parser.add_argument("--ignore-meta", action="store_true", help="Ignora meta-eventi non necessari ai tempi.")
    parser.add_argument("--log-level", choices=["INFO", "DEBUG", "WARNING"], default="INFO", help="Livello logging.")

    # Visualizzazione comune
    parser.add_argument("--viz", action="store_true", help="Abilita la visualizzazione in console.")
    parser.add_argument("--viz-mode", choices=["grid", "lanes"], default="grid", help="Tipo di visualizzazione.")
    parser.add_argument("--viz-refresh-hz", type=float, default=30.0, help="Frequenza refresh visualizzazione (Hz).")
    parser.add_argument("--viz-beats", action="store_true", help="Mostra metronomo / beat grid.")

    # Opzioni GRID
    parser.add_argument("--viz-base-note", type=int, default=60, help="Nota base per grid e mappa lanes (C4=60).")

    # Opzioni LANES
    parser.add_argument("--viz-lanes", type=int, choices=[5, 7], default=7, help="Numero lane per 'lanes'.")
    parser.add_argument("--viz-window-seconds", type=float, default=4.0, help="Finestra futura (s) in 'lanes'.")
    parser.add_argument("--viz-height", type=int, default=20, help="Altezza righe in 'lanes'.")

    args = parser.parse_args(argv)

    # Logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s"
    )
    logger = logging.getLogger("py_midi2song")

    # Solo elenco porte?
    if args.list_ports:
        ports = list_output_ports()
        if not ports:
            print("Nessuna porta MIDI OUT trovata.")
            return 1
        for i, name in enumerate(ports):
            print(f"[{i}] {name}")
        return 0

    if not args.midi:
        logger.error("Specificare --midi PATH per riprodurre un file.")
        return 2

    # Carica MIDI
    try:
        mid = mido.MidiFile(args.midi)
    except FileNotFoundError:
        logging.error("File non trovato: %s", args.midi)
        return 3
    except Exception as e:
        logging.error("Errore nell'aprire/parsing del MIDI: %s", e)
        return 3

    # Timeline + segmenti
    try:
        events, tempo_segs, ts_segs = build_timeline(mid, args.tempo_scale)
    except Exception as e:
        logging.error("Errore nella costruzione della timeline: %s", e)
        return 4

    # Beat grid (se richiesta)
    beatgrid: Optional[BeatGrid] = None
    if args.viz and args.viz_beats:
        beatgrid = build_beat_grid(tempo_segs, ts_segs, mid.ticks_per_beat)

    duration = events[-1].time_s if events else 0.0
    n_meta = sum(1 for ev in events if isinstance(ev.message, mido.MetaMessage))
    n_ev = len(events) - n_meta
    logger.info(
        "File: '%s' | ticks_per_beat=%d | tracce=%d | eventi_canale=%d | meta=%d | durata=%.3fs",
        args.midi, mid.ticks_per_beat, len(mid.tracks), n_ev, n_meta, duration
    )

    # Filtri
    try:
        events_f = apply_filters(events, args.tracks, args.channels, args.ignore_meta)
    except Exception as e:
        logging.error("Spec filtri non valida: %s", e)
        return 5

    t_start = max(0.0, float(args.start_at_seconds or 0.0))
    t_end = None
    if args.end_at_seconds is not None:
        if args.end_at_seconds <= t_start:
            logging.error("--end-at-seconds deve essere > --start-at-seconds")
            return 6
        t_end = float(args.end_at_seconds)

    start_idx = seek_index(events_f, t_start)

    if logger.isEnabledFor(logging.DEBUG):
        preview = events_f[start_idx:start_idx + 10]
        for k, ev in enumerate(preview):
            msg = ev.message
            ch = f"Ch={getattr(msg, 'channel', '-')}"
            t = getattr(msg, 'type', '')
            logger.debug("DBG[%d] t=%.3fs Track=%d %s %s", k, ev.time_s, ev.track_index, ch, t)

    # Porta
    try:
        out = select_output_port(args.port)
    except Exception as e:
        logging.error("Errore selezionando/aprendo la porta MIDI: %s", e)
        return 7

    # Terminal + Visualizzatori
    term = Terminal()
    grid_visual: Optional[GridVisualizer] = None
    lanes_visual: Optional[LanesVisualizer] = None

    if args.viz:
        if args.viz_mode == "grid":
            grid = PadGrid4x4(base_note=int(args.viz_base_note))
            grid_visual = GridVisualizer(term=term, grid=grid, refresh_hz=float(args.viz_refresh_hz), beatgrid=beatgrid, show_beats=args.viz_beats)
            status = [
                f"Porta: {out.name}",
                f"Grid base note: {grid.base_note} (range {grid.base_note}..{grid.base_note+15})",
                f"Durata: {duration:.2f}s",
            ]
            grid_visual.set_status(status)
            grid_visual.set_time_window(t_start, t_end if t_end is not None else None)
            grid_visual.start()
        else:
            segments = _pair_note_segments(events_f, t_end)
            lanes_visual = LanesVisualizer(
                term=term,
                segments=segments,
                lanes=int(args.viz_lanes),
                refresh_hz=float(args.viz_refresh_hz),
                window_seconds=float(args.viz_window_seconds),
                height=int(args.viz_height),
                base_note=int(args.viz_base_note),
                beatgrid=beatgrid,
                show_beats=args.viz_beats,
            )
            status = [
                f"Porta: {out.name}",
                f"Lanes: {args.viz_lanes} | window={args.viz_window_seconds}s | height={args.viz_height}",
                f"Map base note: {args.viz_base_note}",
                f"Durata: {duration:.2f}s",
            ]
            lanes_visual.set_status(status)
            lanes_visual.set_time_window(t_start, t_end if t_end is not None else None)
            lanes_visual.start()

    # Riproduci
    try:
        play_events(
            events=events_f,
            start_idx=start_idx,
            t_start=t_start,
            t_end=t_end,
            out_port=out,
            gain=float(args.gain),
            ignore_meta=args.ignore_meta,
            logger=logger,
            grid_visual=grid_visual,
            lanes_visual=lanes_visual,
        )
    except Exception as e:
        logging.error("Errore durante la riproduzione: %s", e)
        return 8
    finally:
        try:
            out.close()
        except Exception:
            pass
        if grid_visual is not None:
            grid_visual.stop()
            grid_visual.join(timeout=1.0)
        if lanes_visual is not None:
            lanes_visual.stop()
            lanes_visual.join(timeout=1.0)
        term.show_cursor()

    return 0


if __name__ == "__main__":
    sys.exit(main())
