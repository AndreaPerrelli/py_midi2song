"""
Interactive MIDI visualizer prototype for midi2song.

Additions:
- Hitsounds (stereo-panned) on note_on, with velocity-based timbre
- Visual hit effects (shockwave + vertical flare) at the performance line
"""

from __future__ import annotations

import argparse
import bisect
import math
import sys
import time
from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Tuple
from array import array

import pygame
import mido

from .cli import (
    BeatGrid,
    Event,
    build_beat_grid,
    build_timeline,
    list_output_ports,
    select_output_port,
    TempoSeg,
    TimeSigSeg,
)


@dataclass
class VisualNote:
    start: float
    end: float
    note: int
    channel: int
    velocity: int


@dataclass
class RibbonParticle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    size: float


LANE_NOTE_MIN = 24   # C2
LANE_NOTE_MAX = 96   # C7
WINDOW_PAST = 1.5
WINDOW_FUTURE = 6.0
FPS = 60


def _pair_visual_notes(events: Iterable[Event], t_end: float) -> List[VisualNote]:
    active: Dict[Tuple[int, int], Tuple[float, int]] = {}
    notes: List[VisualNote] = []
    for ev in events:
        msg = ev.message
        if not isinstance(msg, mido.Message):
            continue
        if msg.type == "note_on" and msg.velocity > 0 and ev.channel is not None:
            active[(ev.channel, msg.note)] = (ev.time_s, msg.velocity)
        elif msg.type in ("note_off", "note_on") and ev.channel is not None:
            key = (ev.channel, msg.note)
            state = active.pop(key, None)
            if state is None:
                continue
            start, vel = state
            if ev.time_s > start:
                notes.append(
                    VisualNote(
                        start=start,
                        end=ev.time_s,
                        note=msg.note,
                        channel=ev.channel,
                        velocity=vel,
                    )
                )
    # sustain dangling notes to song end
    for (ch, note), (start, vel) in active.items():
        notes.append(
            VisualNote(
                start=start,
                end=max(start, t_end),
                note=note,
                channel=ch,
                velocity=vel,
            )
        )
    notes.sort(key=lambda n: n.start)
    return notes


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    if s == 0:
        r = g = b = l
    else:
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1 / 3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1 / 3)
    return (int(r * 255), int(g * 255), int(b * 255))


def _calc_palette() -> List[Tuple[int, int, int]]:
    base_hues = [210, 50, 140, 0, 280, 100, 330, 180]
    palette: List[Tuple[int, int, int]] = []
    for h in base_hues:
        palette.append(_hsl_to_rgb(h / 360.0, 0.55, 0.55))
    return palette


PALETTE = _calc_palette()


def _color_for_note(note: VisualNote, glow_boost: float = 1.0) -> Tuple[int, int, int]:
    base = PALETTE[note.channel % len(PALETTE)]
    vel_ratio = max(0.15, min(note.velocity / 127.0, 1.0))
    factor = vel_ratio * glow_boost
    return tuple(min(255, int(c * factor)) for c in base)


def _note_to_lane_x(note_number: int, width: int) -> float:
    span = LANE_NOTE_MAX - LANE_NOTE_MIN
    clamped = max(LANE_NOTE_MIN, min(LANE_NOTE_MAX, note_number))
    rel = (clamped - LANE_NOTE_MIN) / span if span > 0 else 0.5
    margin = width * 0.08
    return margin + rel * (width - 2 * margin)


def _ease_pulse(t: float) -> float:
    return math.exp(-4.0 * max(0.0, t)) * (1.0 - math.exp(-6.0 * max(0.0, t)))


def _time_to_screen_y(now: float, t: float, height: int) -> int:
    span = WINDOW_FUTURE + WINDOW_PAST
    rel = (t - (now - WINDOW_PAST)) / span
    clamped = max(0.0, min(1.0, rel))
    return int((1.0 - clamped) * height)


def _draw_background(surface: pygame.Surface, now: float) -> None:
    w, h = surface.get_size()
    for y in range(h):
        t = y / h
        glow = 40 + int(40 * math.sin(now * 0.2 + t * math.pi))
        base = 10 + int(20 * t)
        color = (base + glow, base, base + glow // 2)
        pygame.draw.line(surface, color, (0, y), (w, y))


def _draw_runway(surface: pygame.Surface, now: float, notes: List[VisualNote], active_keys: Dict[Tuple[int, int], float], beatgrid: Optional[BeatGrid]) -> None:
    w, h = surface.get_size()
    lane_width = 6
    tail_height = 8
    horizon = (now - WINDOW_PAST, now + WINDOW_FUTURE)
    # Beats
    if beatgrid:
        idx = bisect.bisect_left(beatgrid.beat_times, horizon[0])
        while idx < len(beatgrid.beat_times) and beatgrid.beat_times[idx] <= horizon[1]:
            t = beatgrid.beat_times[idx]
            y = _time_to_screen_y(now, t, h)
            pulse = _ease_pulse(abs(t - now))
            color = (int(80 + 80 * pulse), int(120 + 135 * pulse), 255)
            pygame.draw.line(surface, color, (0, y), (w, y), width=1 + int(4 * pulse))
            idx += 1

    # Notes
    for note in notes:
        if note.end < horizon[0] - 0.1 or note.start > horizon[1]:
            continue
        y_start = _time_to_screen_y(now, note.start, h)
        y_end = _time_to_screen_y(now, note.end, h)
        lane_x = _note_to_lane_x(note.note, w)
        glow = 1.0
        if (note.channel, note.note) in active_keys and now >= note.start:
            glow = 1.4
        color = _color_for_note(note, glow)
        rect = pygame.Rect(0, 0, lane_width + int(glow * 3), max(4, y_end - y_start))
        rect.centerx = int(lane_x)
        rect.top = min(y_start, y_end)
        pygame.draw.rect(surface, color, rect, border_radius=3)
        if glow > 1.1:
            halo = pygame.Rect(rect)
            halo.inflate_ip(glow * 8, glow * 4)
            alpha = max(60, int(80 * glow))
            halo_surf = pygame.Surface(halo.size, pygame.SRCALPHA)
            pygame.draw.ellipse(
                halo_surf,
                (color[0], color[1], color[2], alpha),
                halo_surf.get_rect(),
                width=2,
            )
            surface.blit(halo_surf, halo)

    # Performance line
    play_y = _time_to_screen_y(now, now, h)
    pygame.draw.line(surface, (250, 250, 255), (0, play_y), (w, play_y), width=3)
    pygame.draw.line(surface, (40, 130, 240), (0, play_y + tail_height), (w, play_y + tail_height), width=1)


# -------------------- Hit Effects (visual) --------------------

@dataclass
class HitRing:
    x: float
    y: float
    life: float
    max_life: float
    color: Tuple[int, int, int]
    base_radius: float
    thickness: int


class HitEffects:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.rings: List[HitRing] = []

    def spawn(self, x: float, y: float, color: Tuple[int, int, int], intensity: float) -> None:
        # intensity ~ velocity 0..1
        base = 10 + 10 * intensity
        thick = 2 + int(3 * intensity)
        self.rings.append(HitRing(x, y, life=0.35, max_life=0.35, color=color, base_radius=base, thickness=thick))

    def update_and_draw(self, surface: pygame.Surface, dt: float) -> None:
        if dt <= 0:
            # draw existing with no time passing
            for r in self.rings:
                self._draw_ring(surface, r, 0.0)
            return

        updated: List[HitRing] = []
        for r in self.rings:
            r.life -= dt
            if r.life <= 0:
                continue
            self._draw_ring(surface, r, dt)
            updated.append(r)
        self.rings = updated

    def _draw_ring(self, surface: pygame.Surface, r: HitRing, dt: float) -> None:
        # expand radius over life
        t = 1.0 - (r.life / r.max_life)
        radius = r.base_radius + 140 * (t ** 0.7)
        alpha = max(0, min(255, int(220 * (1.0 - t))))
        col = (*r.color, alpha)
        ring_surf = pygame.Surface((int(radius * 2 + 8), int(radius * 2 + 8)), pygame.SRCALPHA)
        pygame.draw.circle(ring_surf, col, (ring_surf.get_width() // 4, ring_surf.get_height() // 24), int(radius), width=r.thickness)
        surface.blit(ring_surf, (int(r.x - ring_surf.get_width() / 4), int(r.y - ring_surf.get_height() / 4)))
        # vertical flare
        flare_height = 120 + 160 * t
        flare = pygame.Surface((4, int(flare_height)), pygame.SRCALPHA)
        pygame.draw.rect(flare, (*r.color, int(180 * (1.0 - t))), flare.get_rect(), border_radius=1)
        surface.blit(flare, (int(r.x - 2), int(r.y - flare_height / 2)))


# -------------------- Metronome & Ribbon (existing) --------------------

class MetronomeOverlay:
    def __init__(
        self,
        width: int,
        height: int,
        beatgrid: Optional[BeatGrid],
        tempo_segments: List[TempoSeg],
        timesig_segments: List[TimeSigSeg],
        font: pygame.font.Font,
        polyrhythms: Optional[List[int]] = None,
    ):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.font = font
        self.center = (width // 2, int(height * 0.62))
        self.base_radius = min(width, height) * 0.32
        self.beatgrid = beatgrid
        self.beats = beatgrid.beat_times if beatgrid else []
        self.downbeats = sorted(beatgrid.downbeat_times) if beatgrid else []
        self.polyrhythms = polyrhythms or [3]
        self.tempo_map = self._prep_tempo_map(tempo_segments)
        self.time_sig_segments = self._prep_time_sigs(timesig_segments)
        self.default_spb = self.tempo_map[0][2] if self.tempo_map else 0.5

    def _prep_tempo_map(self, segments: List[TempoSeg]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        for seg in segments:
            start = seg.start_time_s
            end = seg.end_time_s if seg.end_time_s is not None else float("inf")
            spb = seg.tempo_us_per_beat / 1_000_000.0
            out.append((start, end, spb))
        if not out:
            out.append((0.0, float("inf"), 0.5))
        return out

    def _prep_time_sigs(self, segments: List[TimeSigSeg]) -> List[Tuple[float, float, int, int]]:
        out: List[Tuple[float, float, int, int]] = []
        for seg in segments:
            start = seg.start_time_s
            end = seg.end_time_s if seg.end_time_s is not None else float("inf")
            numer = max(1, seg.numer)
            denom = 2 ** seg.denom_pow2
            out.append((start, end, numer, denom))
        if not out:
            out.append((0.0, float("inf"), 4, 4))
        return out

    def _beat_length(self, now: float) -> float:
        for start, end, spb in self.tempo_map:
            if start <= now < end:
                return spb
        return self.default_spb

    def _current_signature(self, now: float) -> Tuple[int, int]:
        for start, end, numer, denom in self.time_sig_segments:
            if start <= now < end:
                return numer, denom
        return (4, 4)

    def _current_measure_length(self, now: float) -> float:
        numer, _ = self._current_signature(now)
        return max(1, numer) * self._beat_length(now)

    def _beat_info(self, now: float) -> Tuple[float, float, float]:
        if not self.beats:
            spb = self._beat_length(now)
            prev = math.floor(now / spb) * spb
            nxt = prev + spb
            phase = (now - prev) / max(1e-6, nxt - prev)
            return (min(max(phase, 0.0), 1.0), prev, nxt)
        idx = max(0, bisect.bisect_right(self.beats, now) - 1)
        prev = self.beats[idx]
        nxt = self.beats[idx + 1] if idx + 1 < len(self.beats) else prev + self._beat_length(now)
        phase = (now - prev) / max(1e-6, nxt - prev)
        return (min(max(phase, 0.0), 1.0), prev, nxt)

    def _measure_info(self, now: float) -> Tuple[float, int, float, float]:
        if not self.downbeats:
            length = self._current_measure_length(now)
            idx = int(now // max(1e-6, length))
            start = idx * length
            end = start + length
            phase = (now - start) / max(1e-6, end - start)
            return (min(max(phase, 0.0), 1.0), idx, start, end)
        idx = max(0, bisect.bisect_right(self.downbeats, now) - 1)
        start = self.downbeats[idx] if idx < len(self.downbeats) else self.downbeats[-1]
        if idx + 1 < len(self.downbeats):
            end = self.downbeats[idx + 1]
        else:
            end = start + self._current_measure_length(now)
        phase = (now - start) / max(1e-6, end - start)
        return (min(max(phase, 0.0), 1.0), idx, start, end)

    def _measure_color(self, measure_idx: int, measure_phase: float) -> Tuple[int, int, int]:
        hue = ((measure_idx * 47) % 360) / 360.0
        pulse = 0.5 + 0.2 * math.sin(measure_phase * math.tau)
        return _hsl_to_rgb(hue, 0.65, pulse)

    def _brighten(self, color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        return tuple(min(255, int(c * factor)) for c in color)

    def _regular_polygon(self, center: Tuple[int, int], radius: float, count: int, rotation: float) -> List[Tuple[int, int]]:
        count = max(3, count)
        cx, cy = center
        pts: List[Tuple[int, int]] = []
        for i in range(count):
            angle = rotation + (i / count) * math.tau
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            pts.append((int(x), int(y)))
        return pts

    def update(self, now: float, dt: float) -> pygame.Surface:
        self.surface.fill((0, 0, 0, 0))
        beat_phase, _, _ = self._beat_info(now)
        measure_phase, measure_idx, _, _ = self._measure_info(now)
        numer, denom = self._current_signature(now)
        base_count = max(3, numer)
        color = self._measure_color(measure_idx, measure_phase)

        pulse = 1.0 - beat_phase
        radius = self.base_radius * (0.68 + 0.28 * (pulse ** 0.7))
        glow_radius = radius * 1.1

        # Soft glow background
        glow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*self._brighten(color, 0.6), 90), self.center, int(glow_radius))
        pygame.draw.circle(glow_surf, (*self._brighten(color, 1.2), 120), self.center, int(radius), width=4 + int(4 * pulse))
        self.surface.blit(glow_surf, (0, 0))

        # Base polygon
        poly_rot = measure_phase * math.tau
        polygon = self._regular_polygon(self.center, radius * 0.82, base_count, poly_rot)
        pygame.draw.polygon(self.surface, self._brighten(color, 1.4), polygon, width=3)

        # Radial spokes
        spoke_color = self._brighten(color, 1.6)
        for i in range(base_count):
            frac = (i / base_count + measure_phase) % 1.0
            angle = frac * math.tau
            inner = (
                int(self.center[0] + math.cos(angle) * radius * 0.25),
                int(self.center[1] + math.sin(angle) * radius * 0.25),
            )
            outer = (
                int(self.center[0] + math.cos(angle) * radius),
                int(self.center[1] + math.sin(angle) * radius),
            )
            pygame.draw.line(self.surface, spoke_color, inner, outer, width=2)

        # Polyrhythm overlays
        for idx, count in enumerate(self.polyrhythms):
            sub_phase = (measure_phase * count) % 1.0
            sub_rot = sub_phase * math.tau
            sub_radius = radius * (0.55 - idx * 0.08)
            if sub_radius <= 10:
                continue
            shade = self._brighten(color, 1.0 + 0.25 * (idx + 1))
            polygon_pts = self._regular_polygon(self.center, sub_radius, count, sub_rot)
            pygame.draw.polygon(self.surface, (*shade, 80), polygon_pts, width=2)
            for px, py in polygon_pts:
                pygame.draw.circle(self.surface, self._brighten(shade, 1.4), (px, py), 4)

        # Spiral orbit
        orbit_radius = radius * 0.35
        orbit_angle = measure_phase * math.tau * (base_count / max(1, denom / 4))
        orbit_pos = (
            int(self.center[0] + math.cos(orbit_angle) * orbit_radius),
            int(self.center[1] + math.sin(orbit_angle) * orbit_radius),
        )
        pygame.draw.circle(self.surface, (*self._brighten(color, 2.0), 220), orbit_pos, 6)

        # Labels
        label = f"Metronome {numer}/{denom}"
        sub_label = f"Polyrhythm: {' + '.join(str(p) for p in [base_count] + self.polyrhythms)}"
        text_img = self.font.render(label, True, self._brighten(color, 1.4))
        text_img2 = self.font.render(sub_label, True, self._brighten(color, 1.1))
        self.surface.blit(text_img, (20, 16))
        self.surface.blit(text_img2, (20, 16 + text_img.get_height() + 6))

        return self.surface


class VelocityRibbon:
    def __init__(self, width: int, height: int, font: pygame.font.Font):
        self.width = width
        self.height = height
        self.font = font
        self.surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.segment_width = width / 16.0
        self.base_background = self._create_background()
        self.energy = [0.0] * 16
        self.display = [0.0] * 16
        self.particles: List[RibbonParticle] = []
        self.palette = [
            _hsl_to_rgb(((20 + ch * 25) % 360) / 360.0, 0.7, 0.5) for ch in range(16)
        ]

    def _create_background(self) -> pygame.Surface:
        bg = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for y in range(self.height):
            t = y / max(1, self.height)
            color = (
                int(12 + 35 * t),
                int(18 + 50 * t),
                int(28 + 90 * t),
                220,
            )
            pygame.draw.line(bg, color, (0, y), (self.width, y))
        for ch in range(17):
            x = int(ch * self.segment_width)
            pygame.draw.line(bg, (25, 30, 55), (x, 0), (x, self.height), width=1)
        pygame.draw.line(bg, (80, 100, 140), (0, self.height - 18), (self.width, self.height - 18), width=2)
        return bg

    def _channel_color(self, channel: int, level: float) -> Tuple[int, int, int]:
        base = self.palette[channel % len(self.palette)]
        factor = 0.6 + 0.6 * level
        return tuple(min(255, int(c * factor)) for c in base)

    def register_hit(self, channel: int, velocity: int, timestamp: float) -> None:
        value = min(1.0, max(velocity / 127.0, 0.0))
        self.energy[channel] = min(1.0, self.energy[channel] + value * 0.6)
        spawn = max(1, int(3 * value))
        for _ in range(spawn):
            self._spawn_particle(channel, value)

    def _spawn_particle(self, channel: int, intensity: float) -> None:
        seg_left = channel * self.segment_width
        base_x = seg_left + self.segment_width / 2 + random.uniform(-self.segment_width * 0.25, self.segment_width * 0.25)
        base_y = self.height - 18
        angle = random.uniform(-math.pi / 3, math.pi / 3)
        speed = 120 + 180 * intensity * random.random()
        vx = math.sin(angle) * speed
        vy = -abs(math.cos(angle) * speed)
        life = 0.5 + 0.45 * random.random()
        color = self._channel_color(channel, 0.7 + 0.3 * random.random())
        size = 3 + 5 * intensity + random.random() * 2
        self.particles.append(RibbonParticle(base_x, base_y, vx, vy, life, life, color, size))

    def update(self, dt: float) -> pygame.Surface:
        decay = math.exp(-dt * 2.1) if dt > 0 else 1.0
        for ch in range(16):
            self.energy[ch] *= decay
            smooth = min(1.0, dt * 6.5) if dt > 0 else 0.0
            self.display[ch] = (1 - smooth) * self.display[ch] + smooth * self.energy[ch]
        # update particles
        updated: List[RibbonParticle] = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 80 * dt
            updated.append(p)
        self.particles = updated

        self.surface.blit(self.base_background, (0, 0))
        margin = 18
        for ch in range(16):
            level = max(0.0, min(1.0, self.display[ch]))
            seg_left = ch * self.segment_width
            width = self.segment_width - 6
            max_height = self.height - margin * 1.4
            bar_height = level * max_height
            top = (self.height - margin) - bar_height
            color = self._channel_color(ch, level)
            rect = pygame.Rect(int(seg_left + 3), int(top), int(width), int(bar_height))
            if rect.height > 0:
                pygame.draw.rect(self.surface, color, rect, border_radius=6)
                glow_rect = rect.copy()
                glow_rect.inflate_ip(6, 8)
                glow = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
                pygame.draw.rect(glow, (*color, int(120 * level + 40)), glow.get_rect(), border_radius=8)
                self.surface.blit(glow, glow_rect)
                peak_y = max(top - 6, 6)
                pygame.draw.line(
                    self.surface,
                    self._channel_color(ch, 1.0),
                    (int(seg_left + 6), int(peak_y)),
                    (int(seg_left + self.segment_width - 6), int(peak_y)),
                    width=2,
                )
            label = self.font.render(f"{ch}", True, (210, 220, 240))
            label_pos = (
                int(seg_left + self.segment_width / 2 - label.get_width() / 2),
                self.height - label.get_height() - 4,
            )
            self.surface.blit(label, label_pos)

        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p.life / p.max_life))))
            size = max(1, int(p.size * (p.life / p.max_life)))
            pygame.draw.circle(
                self.surface,
                (*p.color, alpha),
                (int(p.x), int(p.y)),
                size,
            )

        caption = self.font.render("Velocity Ribbon", True, (200, 210, 230))
        self.surface.blit(caption, (12, 6))

        return self.surface


def _draw_hud(surface: pygame.Surface, font: pygame.font.Font, now: float, duration: float, port_name: str, fps: float) -> None:
    text_color = (230, 240, 255)
    entries = [
        f"Port: {port_name}",
        f"Time: {now:6.2f}s / {duration:6.2f}s",
        f"FPS: {fps:4.1f}",
        "Controls: ESC/Q quit  |  SPACE toggle pause",
    ]
    rendered = [font.render(line, True, text_color) for line in entries]
    max_w = max(img.get_width() for img in rendered) if rendered else 0
    x = surface.get_width() - max_w - 28
    y = 24
    for img in rendered:
        surface.blit(img, (x, y))
        y += img.get_height() + 4


# -------------------- Hitsound Engine (audio) --------------------

# -------------------- Hitsound Engine (audio) --------------------
from array import array

class HitsoundEngine:
    """
    Lightweight synthetic hitsounds with stereo panning.
    No external files; uses short decaying clicks with 3 timbres mapped to velocity.
    """
    def __init__(self, sample_rate: int = 44100, master_volume: float = 0.6):
        self.enabled = True
        self.sr = sample_rate
        self.master_volume = max(0.0, min(1.0, master_volume))
        # init mixer
        try:
            pygame.mixer.init(frequency=self.sr, size=-16, channels=2, buffer=512)
        except pygame.error as exc:
            print(f"[WARN] Impossibile inizializzare mixer: {exc}")
            self.enabled = False
            return
        pygame.mixer.set_num_channels(64)
        # prebuild three variants
        self.sounds = {
            "soft": self._make_click(freq=800.0, dur_ms=36, decay=0.004, noise=0.15),
            "mid":  self._make_click(freq=1200.0, dur_ms=48, decay=0.0035, noise=0.20),
            "hard": self._make_click(freq=1700.0, dur_ms=60, decay=0.0030, noise=0.25),
        }

    def _make_click(self, freq: float, dur_ms: int, decay: float, noise: float) -> pygame.mixer.Sound:
        n = max(1, int(self.sr * dur_ms / 1000.0))
        # stereo interleaved 16-bit
        buf = array("h", [0] * (n * 2))
        phase = 0.0
        dp = 2.0 * math.pi * freq / self.sr
        amp = 28000
        rnd = random.random
        for i in range(n):
            t = i / self.sr
            envelope = math.exp(-t / max(1e-6, decay))
            s = math.sin(phase) * (1.0 - noise) + (rnd() * 2 - 1) * noise
            sample = int(max(-32767, min(32767, amp * s * envelope)))
            # same sample to both channels; panning via channel volumes
            buf[2 * i] = sample
            buf[2 * i + 1] = sample
            phase += dp
        return pygame.mixer.Sound(buffer=buf.tobytes())

    def play(self, velocity: int, pan: float, accent: float = 1.0) -> None:
        """
        Riproduce un hitsound con:
        - curva di volume esponenziale (più naturale),
        - panning equal-power (evita buchi al centro),
        - micro-jitter su pan e volume per umanizzare,
        - blending timbrico fra soft/mid/hard in base alla velocity,
        - leggera apertura stereo alle alte velocity (doppio layer attenuato).

        Args:
            velocity: 0..127
            pan: 0.0 (sinistra) .. 1.0 (destra)
            accent: moltiplicatore opzionale (1.0 = normale)
        """
        if not self.enabled:
            return

        v = max(0, min(127, velocity))
        if v == 0:
            return

        # ---- Curva di volume più musicale + accento ----
        vnorm = v / 127.0
        vol = self.master_volume * (0.25 + 0.9 * (vnorm ** 0.8)) * max(0.0, accent)
        vol = max(0.0, min(1.0, vol))

        # ---- Equal-power panning + micro-jitter per naturalità ----
        pan = max(0.0, min(1.0, pan))
        pan += random.uniform(-0.03, 0.03)         # ±3% di jitter stereo
        pan = max(0.0, min(1.0, pan))
        # equal-power: radice per mantenere energia costante
        left_gain = math.sqrt(1.0 - pan)
        right_gain = math.sqrt(pan)

        # ---- Blending timbrico in base alla velocity ----
        if v < 60:
            snd_a, wa = self.sounds["soft"], 1.0
            snd_b, wb = self.sounds["mid"],  (v / 60.0) * 0.35
        elif v < 105:
            t = (v - 60) / 45.0
            snd_a, wa = self.sounds["mid"],  1.0 - 0.25 * t
            snd_b, wb = self.sounds["hard"], 0.25 * t + 0.15
        else:
            snd_a, wa = self.sounds["hard"], 1.0
            snd_b, wb = self.sounds["mid"],  0.10

        # micro-variazione di gain per “human feel”
        var = 1.0 + random.uniform(-0.06, 0.06)
        base_left = vol * left_gain * var
        base_right = vol * right_gain * var

        # ---- Riproduzione layer A ----
        ch_a = pygame.mixer.find_channel()
        if ch_a is not None:
            ch_a.set_volume(base_left * wa, base_right * wa)
            ch_a.play(snd_a, fade_ms=18)

        # ---- Riproduzione layer B (blend timbrico) ----
        if wb > 0.01:
            ch_b = pygame.mixer.find_channel()
            if ch_b is not None:
                ch_b.set_volume(base_left * wb, base_right * wb)
                ch_b.play(snd_b, fade_ms=12)

        # ---- Apertura stereo per colpi forti ----
        if v > 100 and vol > 0.35:
            widen = 0.22 * (vnorm ** 0.8)
            wl = vol * math.sqrt(pan) * 0.6 * widen    # inverti i canali per “larghezza”
            wr = vol * math.sqrt(1.0 - pan) * 0.6 * widen
            ch_w = pygame.mixer.find_channel()
            if ch_w is not None:
                ch_w.set_volume(wl, wr)
                ch_w.play(self.sounds["hard"], fade_ms=8)



# -------------------- Event sending & helpers --------------------

def _send_pending_events(
    events: List[Event],
    idx: int,
    now: float,
    out_port: mido.ports.BaseOutput,
    active_keys: Dict[Tuple[int, int], float],
    velocity_ribbon: Optional[VelocityRibbon] = None,
    hits: Optional[HitsoundEngine] = None,
    hit_fx: Optional[HitEffects] = None,
    screen_size: Optional[Tuple[int, int]] = None,
) -> int:
    width, height = screen_size if screen_size else (1280, 720)
    play_y = _time_to_screen_y(now, now, height)
    while idx < len(events) and events[idx].time_s <= now:
        ev = events[idx]
        msg = ev.message
        if isinstance(msg, mido.MetaMessage):
            idx += 1
            continue
        if msg.type == "note_on":
            if msg.velocity <= 0:
                key = (msg.channel, msg.note)
                active_keys.pop(key, None)
                out_port.send(mido.Message("note_off", channel=msg.channel, note=msg.note, velocity=0))
            else:
                out_port.send(msg)
                active_keys[(msg.channel, msg.note)] = now
                if velocity_ribbon is not None and msg.channel is not None:
                    velocity_ribbon.register_hit(msg.channel, msg.velocity, ev.time_s)
                # hitsound + visual spark
                lane_x = _note_to_lane_x(msg.note, width)
                if hits is not None:
                    pan = lane_x / max(1, width)  # 0..1
                    hits.play(msg.velocity, pan)
                if hit_fx is not None:
                    color = _hsl_to_rgb(((20 + (msg.channel % 16) * 25) % 360) / 360.0, 0.75, 0.55)
                    hit_fx.spawn(lane_x, play_y, color, msg.velocity / 127.0)
        elif msg.type == "note_off":
            out_port.send(msg)
            active_keys.pop((msg.channel, msg.note), None)
        else:
            out_port.send(msg)
        idx += 1
    return idx


def _all_notes_off(out_port: mido.ports.BaseOutput, active_keys: Dict[Tuple[int, int], float]) -> None:
    for (channel, note) in list(active_keys.keys()):
        out_port.send(mido.Message("note_off", channel=channel, note=note, velocity=0))
    active_keys.clear()
    for ch in range(16):
        out_port.send(mido.Message("control_change", channel=ch, control=123, value=0))
        out_port.send(mido.Message("control_change", channel=ch, control=120, value=0))


def run_visualizer(args: argparse.Namespace) -> int:
    try:
        mid = mido.MidiFile(args.midi)
    except FileNotFoundError:
        print(f"File non trovato: {args.midi}")
        return 1
    except Exception as exc:
        print(f"Errore aprendo il MIDI: {exc}")
        return 1

    events, tempo_segs, ts_segs = build_timeline(mid, 1.0)
    duration = events[-1].time_s if events else 0.0
    notes = _pair_visual_notes(events, duration)
    beatgrid = build_beat_grid(tempo_segs, ts_segs, mid.ticks_per_beat) if args.beats else None

    if args.list_ports:
        for i, name in enumerate(list_output_ports()):
            print(f"[{i}] {name}")
        return 0

    try:
        out = select_output_port(args.port)
    except Exception as exc:
        print(f"Errore aprendo la porta: {exc}")
        return 2

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((args.width, args.height))
    pygame.display.set_caption("midi2song Aurora Runway Prototype")
    hud_font = pygame.font.SysFont("Segoe UI", 18)
    metro_font = pygame.font.SysFont("Segoe UI Semibold", 20)
    ribbon_font = pygame.font.SysFont("Segoe UI", 14)
    clock = pygame.time.Clock()

    overlay_height = max(180, int(args.height * 0.28))
    ribbon_height = max(60, int(args.height * 0.15))
    metronome = MetronomeOverlay(
        width=args.width,
        height=overlay_height,
        beatgrid=beatgrid,
        tempo_segments=tempo_segs,
        timesig_segments=ts_segs,
        font=metro_font,
        polyrhythms=[3, 5],
    )
    velocity_ribbon = VelocityRibbon(args.width, ribbon_height, ribbon_font)

    # New: hit effects & hitsounds
    hit_fx = HitEffects(args.width, args.height)
    hits_engine = None
    if args.hitsounds:
        engine = HitsoundEngine(master_volume=args.hitsound_volume)
        hits_engine = engine if engine.enabled else None
        if hits_engine is None:
            print("[WARN] Hitsounds disabilitati (mixer non disponibile).")

    start_time = time.monotonic()
    paused = False
    pause_offset = 0.0
    event_idx = 0
    active_keys: Dict[Tuple[int, int], float] = {}
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0
        raw_now = time.monotonic()
        for p_event in pygame.event.get():
            if p_event.type == pygame.QUIT:
                running = False
            elif p_event.type == pygame.KEYDOWN:
                if p_event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif p_event.key == pygame.K_SPACE:
                    if paused:
                        start_time = raw_now - pause_offset
                        paused = False
                    else:
                        pause_offset = raw_now - start_time
                        paused = True

        now = pause_offset if paused else (raw_now - start_time)

        if not paused:
            event_idx = _send_pending_events(
                events, event_idx, now, out, active_keys,
                velocity_ribbon=velocity_ribbon,
                hits=hits_engine,
                hit_fx=hit_fx,
                screen_size=(args.width, args.height),
            )

        _draw_background(screen, now)
        _draw_runway(screen, now, notes, active_keys, beatgrid)
        overlay_dt = 0.0 if paused else dt
        met_surface = metronome.update(now, overlay_dt) if metronome else None
        ribbon_surface = velocity_ribbon.update(overlay_dt)
        if met_surface is not None:
            screen.blit(met_surface, (0, 0))
        screen.blit(ribbon_surface, (0, args.height - ribbon_surface.get_height()))

        # draw hit effects last so they sit on top
        hit_fx.update_and_draw(screen, overlay_dt)

        fps = clock.get_fps()
        _draw_hud(screen, hud_font, now, duration, out.name, fps)
        pygame.display.flip()

        if event_idx >= len(events) and now > duration + 1.0 and not paused:
            running = False

    _all_notes_off(out, active_keys)
    out.close()
    if pygame.mixer.get_init():
        pygame.mixer.quit()
    pygame.quit()
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aurora Runway visualizer prototype for midi2song (pygame)."
    )
    parser.add_argument("--midi", required=True, help="Percorso al file MIDI.")
    parser.add_argument("--port", help="Nome parziale o indice della porta MIDI da usare.")
    parser.add_argument("--width", type=int, default=1280, help="Larghezza finestra.")
    parser.add_argument("--height", type=int, default=720, help="Altezza finestra.")
    parser.add_argument("--beats", action="store_true", help="Mostra il beat grid pulsante.")
    parser.add_argument("--list-ports", action="store_true", help="Elenca le porte MIDI disponibili e termina.")
    # New hitsound options
    parser.add_argument("--no-hitsounds", dest="hitsounds", action="store_false", help="Disattiva i suoni di hit.")
    parser.add_argument("--hitsounds", dest="hitsounds", action="store_true", help="Attiva i suoni di hit (default).")
    parser.set_defaults(hitsounds=True)
    parser.add_argument("--hitsound-volume", type=float, default=0.6, help="Volume master hitsounds (0..1).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    return run_visualizer(args)


if __name__ == "__main__":
    sys.exit(main())
