"""
Nebula Conductor MVP.

A minimal-yet-real prototype that couples midi2song's MIDI timeline with a
moderngl + pyglet GPU scene and a DMX-ready lighting stub.

Usage:
    python app.py --midi path/to/song.mid [--port MIDI_PORT] [--osc-port 9000]

Requirements:
    Python 3.10+
    pip install moderngl pyglet mido python-rtmidi numpy
    (python-osc optional)
"""

from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mido
import moderngl
import numpy as np
import pyglet

try:
    from pythonosc import dispatcher, osc_server
except ImportError:  # pragma: no cover - optional dependency
    dispatcher = None
    osc_server = None

from src.midi2song.cli import (
    BeatGrid,
    TempoSeg,
    TimeSigSeg,
    build_beat_grid,
    build_timeline,
    list_output_ports,
    select_output_port,
)


# -------------------------------
# Lighting bus (DMX stub)
# -------------------------------


class LightingBus:
    """Stub that would fan-out to DMX / OSC. Currently logs at ~25Hz."""

    def __init__(self, rate_hz: float = 25.0) -> None:
        self.rate_hz = max(1.0, rate_hz)
        self._last_sent = 0.0
        self._latest_payload: Optional[np.ndarray] = None
        self._latest_beat: float = 0.0
        # TODO: swap logging with real ArtNet/sACN transport (python-sacn / ola).

    def publish_frame(self, energies: np.ndarray, beat_phase: float) -> None:
        """Queue the latest channel energies to be flushed on the next tick."""
        self._latest_payload = np.clip(energies, 0.0, 1.0).astype(np.float32)
        self._latest_beat = float(beat_phase)

    def tick(self, now: float) -> None:
        if self._latest_payload is None:
            return
        if now - self._last_sent < 1.0 / self.rate_hz:
            return
        payload = ", ".join(f"{v:.2f}" for v in self._latest_payload)
        logging.debug("LightingBus frame beat=%.3f channels=[%s]", self._latest_beat, payload)
        self._last_sent = now


# -------------------------------
# Velocity ribbon (energy model)
# -------------------------------


class VelocityRibbon:
    """Keeps a per-channel envelope to drive shaders / lighting."""

    def __init__(self, attack: float = 0.35, release: float = 1.6) -> None:
        self.attack = max(1e-3, attack)
        self.release = max(1e-3, release)
        self.energy = np.zeros(16, dtype=np.float32)

    def register_hit(self, channel: int, velocity: int) -> None:
        lvl = np.clip(velocity / 127.0, 0.0, 1.0)
        alpha = self.attack
        self.energy[channel] = np.clip((1.0 - alpha) * self.energy[channel] + alpha * lvl, 0.0, 1.0)

    def decay(self, dt: float) -> None:
        factor = math.exp(-dt * self.release)
        self.energy *= factor


# -------------------------------
# Tempo / polyrhythm helpers
# -------------------------------


class TempoMap:
    def __init__(self, tempo_segs: List[TempoSeg], ts_segs: List[TimeSigSeg], beatgrid: Optional[BeatGrid]) -> None:
        self.tempo_segs = tempo_segs
        self.timesig_segs = ts_segs
        self.beatgrid = beatgrid

    def _current_tempo(self, t: float) -> float:
        for seg in self.tempo_segs:
            if seg.start_time_s <= t < (seg.end_time_s or float("inf")):
                return seg.tempo_us_per_beat / 1_000_000.0
        return self.tempo_segs[-1].tempo_us_per_beat / 1_000_000.0

    def _current_timesig(self, t: float) -> Tuple[int, int]:
        for seg in self.timesig_segs:
            if seg.start_time_s <= t < (seg.end_time_s or float("inf")):
                return seg.numer, 2 ** seg.denom_pow2
        last = self.timesig_segs[-1]
        return last.numer, 2 ** last.denom_pow2

    def _nearest(self, arr: List[float], t: float) -> Tuple[float, float]:
        import bisect

        if not arr:
            return (0.0, 0.0)
        idx = bisect.bisect_right(arr, t) - 1
        idx = max(0, min(idx, len(arr) - 1))
        base = arr[idx]
        next_beat = arr[idx + 1] if idx + 1 < len(arr) else base + self._current_tempo(t)
        phase = (t - base) / max(next_beat - base, 1e-6)
        return (base, phase)

    def phases(self, play_time: float) -> Tuple[float, float, float, Tuple[float, float, float], float]:
        spb = self._current_tempo(play_time)
        bpm = 60.0 / spb if spb > 0 else 120.0
        if self.beatgrid:
            beat_base, beat_phase = self._nearest(self.beatgrid.beat_times, play_time)
            measure_base, measure_phase = self._nearest(sorted(self.beatgrid.downbeat_times), play_time)
            measure_length = max(spb, play_time - measure_base + spb)
        else:
            beat_base = math.floor(play_time / spb) * spb
            beat_phase = (play_time - beat_base) / max(spb, 1e-6)
            numer, _ = self._current_timesig(play_time)
            measure_length = spb * max(1, numer)
            measure_base = math.floor(play_time / measure_length) * measure_length
            measure_phase = (play_time - measure_base) / max(measure_length, 1e-6)
        poly = tuple(((measure_phase * count) % 1.0) for count in (3.0, 4.0, 5.0))
        return beat_phase % 1.0, measure_phase % 1.0, bpm, poly, measure_length


# -------------------------------
# MIDI engine
# -------------------------------


@dataclass
class RenderState:
    time_elapsed: float
    beat_phase: float
    measure_phase: float
    poly_phase: Tuple[float, float, float]
    bpm: float
    channel_energy: np.ndarray
    pitch_warp: np.ndarray
    sustain_boost: np.ndarray
    color_shift: np.ndarray


class MidiEngine:
    def __init__(
        self,
        midi_path: Path,
        port: Optional[str],
        lighting: LightingBus,
        tempo_scale: float = 1.0,
    ):
        self.midi_path = midi_path
        self.mid = mido.MidiFile(str(midi_path))
        self.events, tempo_segs, ts_segs = build_timeline(self.mid, tempo_scale)
        self.beatgrid = build_beat_grid(tempo_segs, ts_segs, self.mid.ticks_per_beat)
        self.tempo_map = TempoMap(tempo_segs, ts_segs, self.beatgrid)
        self.tempo_scale = tempo_scale
        self.start_mono: Optional[float] = None
        self.last_mono: Optional[float] = None
        self._event_idx = 0
        self.out_port = select_output_port(port) if port else None
        self.velocity_ribbon = VelocityRibbon()
        self.lighting = lighting

        self.channel_energy = np.zeros(16, dtype=np.float32)
        self.pitch_warp = np.zeros(16, dtype=np.float32)
        self.sustain = np.zeros(16, dtype=np.float32)
        base = np.linspace(0.0, 1.0, 16, dtype=np.float32)
        self.base_color = base

        self.held_notes: Dict[Tuple[int, int], int] = {}

    def close(self) -> None:
        if self.out_port:
            try:
                self.out_port.close()
            except Exception:
                pass

    def set_tempo_scale(self, scale: float) -> None:
        scale = max(0.25, min(scale, 4.0))
        now = time.monotonic()
        if self.start_mono is not None:
            playback = (now - self.start_mono) * self.tempo_scale
            self.start_mono = now - playback / scale
        self.tempo_scale = scale
        logging.info("Tempo scale set to %.2f", scale)

    def start(self) -> None:
        if self.start_mono is None:
            self.start_mono = time.monotonic()
            self.last_mono = self.start_mono

    def _play_time(self, now: float) -> float:
        if self.start_mono is None:
            return 0.0
        return max(0.0, (now - self.start_mono) * self.tempo_scale)

    def _handle_message(self, msg: mido.Message, event_time: float) -> None:
        if self.out_port:
            try:
                self.out_port.send(msg)
            except Exception as exc:
                logging.warning("Failed sending to port: %s", exc)

        if msg.type == "note_on" and msg.velocity > 0:
            ch = msg.channel
            self.velocity_ribbon.register_hit(ch, msg.velocity)
            lvl = msg.velocity / 127.0
            self.channel_energy[ch] = min(1.5, self.channel_energy[ch] + lvl * 0.8)
            target = (msg.note - 60.0) / 24.0
            self.pitch_warp[ch] = (self.pitch_warp[ch] * 0.6) + target * 0.4
            self.held_notes[(ch, msg.note)] = msg.velocity
        elif msg.type in ("note_off", "note_on") and msg.velocity == 0:
            self.held_notes.pop((msg.channel, msg.note), None)
        elif msg.type == "pitchwheel":
            self.pitch_warp[msg.channel] = msg.pitch / 8192.0
        elif msg.type == "control_change" and msg.control == 64:
            active = 1.0 if msg.value >= 64 else 0.0
            self.sustain[msg.channel] = active

    def advance(self) -> RenderState:
        now = time.monotonic()
        self.start()
        play_time = self._play_time(now)
        dt = 0.0 if self.last_mono is None else now - self.last_mono
        self.last_mono = now

        while self._event_idx < len(self.events) and self.events[self._event_idx].time_s <= play_time:
            ev = self.events[self._event_idx]
            msg = ev.message
            if isinstance(msg, mido.MetaMessage):
                self._event_idx += 1
                continue
            self._handle_message(msg, ev.time_s)
            self._event_idx += 1

        decay = math.exp(-dt * 1.3) if dt > 0 else 1.0
        self.channel_energy *= decay
        self.velocity_ribbon.decay(dt)
        self.sustain *= decay * 0.9 + 0.1  # slow fall after pedal release

        beat_phase, measure_phase, bpm, poly, _ = self.tempo_map.phases(play_time)
        color_shift = (self.base_color + 0.35 * self.channel_energy + 0.1 * measure_phase) % 1.0

        self.lighting.publish_frame(self.channel_energy, beat_phase)

        return RenderState(
            time_elapsed=play_time,
            beat_phase=beat_phase,
            measure_phase=measure_phase,
            poly_phase=poly,
            bpm=bpm,
            channel_energy=self.channel_energy.copy(),
            pitch_warp=self.pitch_warp.copy(),
            sustain_boost=self.sustain.copy(),
            color_shift=color_shift.astype(np.float32),
        )


# -------------------------------
# Shader scene (moderngl)
# -------------------------------


class ShaderScene:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self.ctx = ctx
        self.ctx.enable(moderngl.BLEND)
        self.width = width
        self.height = height
        self.fog = 0.35
        self.bloom_strength = 1.3
        self._load_resources()
        self._create_framebuffers(width, height)
        self.state: Optional[RenderState] = None
        # TODO: inject VisPy / fractal overlays composited on additional quads.

    def _load_resources(self) -> None:
        quad = np.array(
            [
                -1.0, -1.0,
                1.0, -1.0,
                -1.0, 1.0,
                -1.0, 1.0,
                1.0, -1.0,
                1.0, 1.0,
            ],
            dtype="f4",
        )
        self.quad_vbo = self.ctx.buffer(quad.tobytes())
        self.quad_vao = self.ctx.simple_vertex_array(
            self.ctx.program(
                vertex_shader=_QUAD_VERT,
                fragment_shader=_PASSTHROUGH_FRAG,
            ),
            self.quad_vbo,
            "in_pos",
        )
        self.scene_prog = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_NEBULA_FRAG)
        self.blur_prog = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_BLUR_FRAG)
        self.composite_prog = self.ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_COMPOSITE_FRAG)

    def _create_framebuffers(self, width: int, height: int) -> None:
        size = (max(2, width), max(2, height))
        self.scene_tex = self.ctx.texture(size, components=4, dtype="f4")
        self.scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.scene_fbo = self.ctx.framebuffer(self.scene_tex)

        self.blur_tex_a = self.ctx.texture(size, components=4, dtype="f4")
        self.blur_tex_b = self.ctx.texture(size, components=4, dtype="f4")
        self.blur_tex_a.filter = self.blur_tex_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.blur_fbo_a = self.ctx.framebuffer(self.blur_tex_a)
        self.blur_fbo_b = self.ctx.framebuffer(self.blur_tex_b)

    def resize(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._create_framebuffers(width, height)

    def update_state(self, state: RenderState, fog: Optional[float] = None, bloom: Optional[float] = None) -> None:
        self.state = state
        if fog is not None:
            self.fog = fog
        if bloom is not None:
            self.bloom_strength = bloom

    def render(self) -> None:
        if self.state is None:
            return
        state = self.state
        res = (float(self.width), float(self.height))

        # Scene pass with additive blending between emitters
        self.ctx.blend_func = (moderngl.ONE, moderngl.ONE)
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        prog = self.scene_prog
        prog["u_resolution"].value = res
        prog["u_time"].value = state.time_elapsed
        prog["u_beat_phase"].value = state.beat_phase
        prog["u_measure_phase"].value = state.measure_phase
        prog["u_bpm"].value = state.bpm
        prog["u_fog"].value = self.fog
        prog["u_polyPhase"].value = tuple(float(x) for x in state.poly_phase)
        prog["u_channelEnergy"].value = tuple(float(x) for x in state.channel_energy)
        prog["u_pitchWarp"].value = tuple(float(x) for x in state.pitch_warp)
        prog["u_colorShift"].value = tuple(float(x) for x in state.color_shift)
        prog["u_sustainBoost"].value = tuple(float(x) for x in state.sustain_boost)
        self._draw_fullscreen(prog)

        # Two-pass separable blur for bloom
        texel = (1.0 / max(1, self.width), 1.0 / max(1, self.height))
        self._apply_blur(self.scene_tex, texel)

        # Composite
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        comp = self.composite_prog
        comp["u_bloomStrength"].value = self.bloom_strength
        comp["u_scene"].value = 0
        comp["u_bloom"].value = 1
        self.scene_tex.use(location=0)
        self.blur_tex_b.use(location=1)
        self._draw_fullscreen(comp)
        # TODO: add fisheye/dual-eye passes for dome or VR output.

    def _draw_fullscreen(self, program: moderngl.Program) -> None:
        vao = self.ctx.simple_vertex_array(program, self.quad_vbo, "in_pos")
        vao.render(moderngl.TRIANGLES)

    def _apply_blur(self, source_tex: moderngl.Texture, texel: Tuple[float, float]) -> None:
        source_tex.use(location=0)
        self.blur_fbo_a.use()
        self.ctx.clear()
        self.blur_prog["u_direction"].value = (texel[0], 0.0)
        self.blur_prog["u_source"].value = 0
        self._draw_fullscreen(self.blur_prog)

        self.blur_tex_a.use(location=0)
        self.blur_fbo_b.use()
        self.ctx.clear()
        self.blur_prog["u_direction"].value = (0.0, texel[1])
        self._draw_fullscreen(self.blur_prog)

        # Restore standard blending for final composite to avoid accumulating brightness frame-to-frame
        self.ctx.blend_func = (moderngl.ONE, moderngl.ZERO)


# -------------------------------
# OSC control (optional)
# -------------------------------


class OscController:
    """Optional OSC slider interface to tweak shader / tempo parameters."""

    def __init__(self, scene: ShaderScene, midi_engine: MidiEngine, port: Optional[int]):
        self.scene = scene
        self.midi_engine = midi_engine
        self.port = port
        self.server: Optional[osc_server.ThreadingOSCUDPServer] = None
        if port is None:
            logging.info("OSC control disabled.")
            return
        if dispatcher is None or osc_server is None:
            logging.warning("python-osc not installed; OSC disabled.")
            return
        disp = dispatcher.Dispatcher()
        disp.map("/shader/fog", self._set_fog)
        disp.map("/shader/bloom", self._set_bloom)
        disp.map("/tempo/scale", self._set_tempo_scale)
        disp.map("/shader/debug", self._debug)
        self.server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", port), disp)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        logging.info("OSC control listening on udp://0.0.0.0:%d", port)

    def shutdown(self) -> None:
        if self.server:
            self.server.shutdown()

    def _set_fog(self, address: str, value: float) -> None:
        self.scene.fog = float(np.clip(value, 0.0, 1.0))
        logging.info("OSC fog -> %.2f", self.scene.fog)

    def _set_bloom(self, address: str, value: float) -> None:
        self.scene.bloom_strength = float(np.clip(value, 0.0, 4.0))
        logging.info("OSC bloom -> %.2f", self.scene.bloom_strength)

    def _set_tempo_scale(self, address: str, value: float) -> None:
        self.midi_engine.set_tempo_scale(float(value))

    def _debug(self, address: str, *args) -> None:
        logging.info("OSC debug %s %s", address, args)


# -------------------------------
# Main application shell
# -------------------------------


class NebulaConductorApp:
    def __init__(self, args: argparse.Namespace) -> None:
        logging.info("Loading MIDI '%s'", args.midi)
        lighting = LightingBus()
        self.midi_engine = MidiEngine(Path(args.midi), args.port, lighting, tempo_scale=args.tempo_scale)
        self.window = pyglet.window.Window(
            width=args.width,
            height=args.height,
            caption="Nebula Conductor MVP",
            fullscreen=args.fullscreen,
            resizable=True,
        )
        self.ctx = moderngl.create_context()
        self.scene = ShaderScene(self.ctx, self.window.width, self.window.height)
        self.osc = OscController(self.scene, self.midi_engine, args.osc_port)
        self.lighting = lighting
        self.args = args
        self._state = self.midi_engine.advance()  # prime state
        self.scene.update_state(self._state, fog=args.fog, bloom=args.bloom)

        pyglet.clock.schedule_interval(self._update, 1.0 / 120.0)

        @self.window.event
        def on_draw() -> None:
            self.scene.render()

        @self.window.event
        def on_resize(width: int, height: int) -> None:
            self.scene.resize(width, height)

        @self.window.event
        def on_key_press(symbol: int, modifiers: int) -> None:  # pragma: no cover - interactive
            if symbol == pyglet.window.key.ESCAPE:
                pyglet.app.exit()

    def _update(self, dt: float) -> None:
        state = self.midi_engine.advance()
        self.scene.update_state(state)
        self.lighting.tick(time.monotonic())
        self._state = state

    def run(self) -> None:
        try:
            pyglet.app.run()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.osc.shutdown()
        self.midi_engine.close()


# -------------------------------
# CLI glue
# -------------------------------


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nebula Conductor MVP (Moderngl + MIDI + DMX-ready stub).")
    parser.add_argument("--midi", required=True, help="Percorso al file MIDI.")
    parser.add_argument("--port", help="Porta MIDI OUT da usare (nome o indice).")
    parser.add_argument("--width", type=int, default=1280, help="Larghezza finestra.")
    parser.add_argument("--height", type=int, default=720, help="Altezza finestra.")
    parser.add_argument("--fullscreen", action="store_true", help="Avvia a schermo intero.")
    parser.add_argument("--tempo-scale", type=float, default=1.0, help="Scala globale del tempo.")
    parser.add_argument("--bloom", type=float, default=1.3, help="Intensità bloom iniziale (0..4).")
    parser.add_argument("--fog", type=float, default=0.35, help="Fattore fog / fade (0..1).")
    parser.add_argument("--osc-port", type=int, default=None, help="Porta OSC per controlli in tempo reale.")
    parser.add_argument("--list-ports", action="store_true", help="Elenca le porte MIDI disponibili e termina.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log dettagliato.")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    if args.list_ports:
        ports = list_output_ports()
        if not ports:
            print("Nessuna porta MIDI OUT trovata.")
        else:
            for i, name in enumerate(ports):
                print(f"[{i}] {name}")
        return

    pyglet.options["vsync"] = True
    app = NebulaConductorApp(args)
    app.run()


# ------------------------------------
# Shader snippets (can be externalised)
# ------------------------------------

_QUAD_VERT = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_PASSTHROUGH_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_uv, 0.0, 1.0);
}
"""

_NEBULA_FRAG = """
#version 330
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_beat_phase;
uniform float u_measure_phase;
uniform float u_bpm;
uniform float u_fog;
uniform float u_channelEnergy[16];
uniform float u_pitchWarp[16];
uniform float u_colorShift[16];
uniform float u_sustainBoost[16];
uniform float u_polyPhase[3];

in vec2 v_uv;
out vec4 fragColor;

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

vec2 curl(vec2 p) {
    float e = 0.04;
    float n1 = noise(p + vec2(0.0, e));
    float n2 = noise(p - vec2(0.0, e));
    float n3 = noise(p + vec2(e, 0.0));
    float n4 = noise(p - vec2(e, 0.0));
    return vec2(n1 - n2, n3 - n4);
}

vec3 hsl2rgb(vec3 hsl) {
    vec3 rgb = clamp(abs(mod(hsl.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return hsl.z + hsl.y * (rgb - 0.5) * (1.0 - abs(2.0 * hsl.z - 1.0));
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    uv.x *= aspect;

    vec3 accum = vec3(0.0);
    float total = 0.0;
    float measurePulse = 0.6 + 0.4 * sin(u_measure_phase * 6.2831);
    float bpmMod = clamp(u_bpm / 180.0, 0.35, 1.6);

    for (int ch = 0; ch < 16; ++ch) {
        float energy = u_channelEnergy[ch];
        if (energy < 0.002) continue;
        float sustain = clamp(u_sustainBoost[ch], 0.0, 1.0);
        float warp = u_pitchWarp[ch];
        float hue = fract(u_colorShift[ch] + 0.15 * sin(u_time * 0.15 + ch));
        float lfo = sin(u_time * (0.4 + 0.05 * ch) + float(ch)) * 0.5 + 0.5;
        vec2 seed = vec2(float(ch) * 1.37, float(ch) * 4.21);
        vec2 flow = curl(uv * (1.4 + sustain * 1.5) + seed + u_time * (0.1 + energy * 0.3));
        float swirl = dot(flow, vec2(uv.y, -uv.x)) + warp * 0.8;
        float dist = length(uv + flow * (0.8 + sustain));
        float falloff = exp(-dist * (2.4 - energy * 0.5)) * (0.6 + 0.6 * lfo);
        float beatPulse = 0.6 + 0.4 * cos((u_beat_phase) * 6.2831 + float(ch));
        vec3 color = hsl2rgb(vec3(hue, clamp(0.55 + energy * 0.5, 0.0, 1.0), 0.45 + sustain * 0.25 * measurePulse));
        accum += color * falloff * energy * beatPulse * (1.0 + abs(swirl) * bpmMod);
        total += falloff;
    }

    float fog = exp(-length(uv) * (0.8 + 1.5 * u_fog));
    float polyGlow = 0.0;
    for (int i = 0; i < 3; ++i) {
        float phase = u_polyPhase[i];
        polyGlow += exp(-20.0 * pow(abs(fract(phase) - 0.5), 2.0));
    }
    vec3 base = accum * fog + vec3(0.02 * polyGlow);
    fragColor = vec4(base, clamp(total, 0.0, 1.0));
}
"""

_BLUR_FRAG = """
#version 330
uniform sampler2D u_source;
uniform vec2 u_direction;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
    vec3 result = texture(u_source, v_uv).rgb * weights[0];
    for (int i = 1; i < 5; ++i) {
        vec2 offset = u_direction * float(i);
        result += texture(u_source, v_uv + offset).rgb * weights[i];
        result += texture(u_source, v_uv - offset).rgb * weights[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

_COMPOSITE_FRAG = """
#version 330
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloomStrength;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb * u_bloomStrength;
    vec3 color = scene + bloom;
    color = color / (1.0 + color); // simple Reinhard tone mapping
    fragColor = vec4(color, 1.0);
}
"""


if __name__ == "__main__":  # pragma: no cover
    main()
