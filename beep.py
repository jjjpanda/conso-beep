#!/usr/bin/env python3
"""Usage: beep.py [--root HZ|NOTE | --auto-key [SECONDS] | --avoid-speech [SECONDS] | --mic [SECONDS]] [--pan PAN] [--muffle MUFFLE] [--ticks N] [--verbose] [--hook [--sync] | --melody NAME | -- semitones,vol,ms ...]
       beep.py --listen
  semitones: signed offset from root (0 = root note), -48 to 48
  vol:       0.0-1.0; 0 is a rest
  ms:        time until the next note in milliseconds, 0-10000; notes ring past it

  --root         Hz number (523.25) or note name (C5, F#4, Bb3). Default: 440 (A4). Max 20000.
                 Folded by octaves into 600-1200 Hz
  --auto-key     record SECONDS (default 2) of mic audio, use detected key's tonic as root
  --avoid-speech record SECONDS (default 1.5) of mic audio, use quietest region as root
  --mic          record SECONDS (default 2) of mic audio, auto-detect music vs talking
                 (by key-detection confidence) and pick --auto-key or --avoid-speech behavior.
                 Mic modes also scale volume to background loudness
  --pan          stereo pan from -1.0 (left) to 1.0 (right); overrides Windows detection
  --muffle       muffle amount from 0.0 (clear) to 1.0 (behind a wall); overrides Windows detection
  --ticks        0-3 clicks before the melody (virtual-desktop distance); overrides Windows detection
  --verbose      log resolved spatial settings to stderr
  --hook         read LLM harness hook JSON from stdin, trigger matching
                 melody, and emit hook JSON to stdout
  --sync         play audio synchronously in --hook mode instead of spawning detached
  --melody       play a named melody from HOOK_MELODIES (e.g. Stop, Notification:idle_prompt)
  --listen       run the daemon: play requests sent on localhost at once, with mic modes
                 read from a rolling mic buffer. Hooks start it on demand; it exits
                 after 10 idle minutes

Mic recording is capped at 30 seconds. Without a mic mode, volume is 0.5.

Note: positional notes must follow a `--` separator (negative offsets look like
flags to argparse otherwise), e.g. beep.py --root 440 -- -3,0.7,90
"""
import argparse
import json
import math
import pathlib
import re
import subprocess
import sys

SAMPLE_RATE = 44100
BELL_PARTIALS = ((1, 1.0, 0.275), (2, 0.22, 0.11), (3, 0.04, 0.04), (4, 0.15, 0.03))
TONE_LENGTH = 0.15
BELL_ATTACK_S = 0.0035
BELL_RELEASE_S = 0.013
NOTE_RING_S = 0.68
TEMPO = 0.93
LEAD_S = 0.001
STEREO_SPREAD = 0.24
STUTTER_COUNT = 1
STUTTER_GRAIN_S = 0.035
STUTTER_SPACING_S = 0.039
STUTTER_LEVEL = 0.5
STUTTER_DECAY_S = 0.008
AMBIENT_DB = (-55.0, -25.0)
AMBIENT_GAIN = (0.5, 1.0)
DEFAULT_GAIN = 0.5
AMBIENT_FRAME_S = 0.05
AMBIENT_PERCENTILE = 50
TICK_VOL = 1.0
TICK_MS = 60
CLICK_GAP_MS = 85
CLICK_LEAD_MS = 60

DEFAULT_ROOT_HZ = 440.0
DEFAULT_AUTO_KEY_SECONDS = 2.0
DEFAULT_AVOID_SPEECH_SECONDS = 1.5
DEFAULT_MIC_SECONDS = 2.0
ROOT_BAND_HZ = 1200.0
MAX_ROOT_HZ = 20000.0
MAX_RECORD_SECONDS = 30.0
NOTE_LIMITS = ((-48.0, 48.0, "semitones"), (0.0, 1.0, "vol"), (0.0, 10000.0, "ms"))
MUSIC_CONFIDENCE_THRESHOLD = 0.55
CHROMA_FMIN, CHROMA_FMAX = 65.0, 2100.0
QUIET_BAND = (200.0, 4000.0)
QUIET_SMOOTH_HZ = 30.0
DAEMON_PORT = 47392
DAEMON_IDLE_S = 600
DAEMON_TIMEOUT_S = 0.25
DAEMON_MIN_MIC_S = 0.5

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

NOTE_NAME_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

ANTIGRAVITY_FAILURE_REASONS = (
    "ERROR",
    "MAX_INVOCATIONS",
    "MAX_FORCED_INVOCATIONS",
    "MAX_TOKEN_BUDGET_EXCEEDED",
)

HOOK_MELODIES: dict[str, tuple[str, ...]] = {
    "SessionStart": ("0,0.9,72", "4,1,72", "7,1,99"),
    "Stop": ("0,0.9,99", "-4,0.9,131"),
    "StopFailure": ("0,0.9,110", "-3,0.95,110", "-6,1,185"),
    "PostToolUseFailure": ("5,0.7,90", "-1,0.7,220"),
    "PostCompact": ("0,0.9,61", "2,1,83"),
    "SubagentStop": ("0,0.9,72", "-3,0.95,94"),
    "Notification:permission_prompt": ("0,1,90", "7,1,120", "9,1,90", "14,1,160"),
    "Notification:idle_prompt": ("0,0.9,72", "0,0.9,110", "-5,0.95,72"),
    "Notification:auth_success": ("0,1,60", "4,1,60", "7,1,60", "12,1,60", "16,1,240"),
    "Notification:elicitation_dialog": ("4,1,80", "6,1,80", "2,1,80", "7,1,90", "11,1,260"),
    "RulesMissing": ("6,1,110", "1,1,110", "6,1,260"),
}


def parse_note_name(name):
    m = NOTE_NAME_RE.match(name)
    if not m:
        raise ValueError(f"not a note name: {name!r}")
    letter, accidental, octave = m.groups()
    semitone = NOTE_LETTER_SEMITONES[letter.upper()]
    if accidental == "#":
        semitone += 1
    elif accidental == "b":
        semitone -= 1
    midi = 12 * (int(octave) + 1) + semitone
    return 440.0 * 2 ** ((midi - 69) / 12)


def parse_root(root_arg):
    try:
        hz = float(root_arg)
    except ValueError:
        try:
            hz = parse_note_name(root_arg)
        except (ValueError, OverflowError):
            hz = math.nan
    if not 0 < hz <= MAX_ROOT_HZ:
        raise argparse.ArgumentTypeError(
            f"--root must be a Hz number in (0, {MAX_ROOT_HZ:g}] or note name (e.g. 523.25, C5, F#4): got {root_arg!r}"
        )
    return hz


def semitone_to_hz(root_hz, semitones):
    return root_hz * 2 ** (semitones / 12)


def fold_root(freq, band_hz=ROOT_BAND_HZ):
    while freq > band_hz:
        freq /= 2
    while freq <= band_hz / 2:
        freq *= 2
    return freq


def parse_note(arg):
    parts = arg.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected semitones,vol,ms: got {arg!r}")
    return tuple(parse_clamped_float(*limits)(p) for limits, p in zip(NOTE_LIMITS, parts))


def parse_clamped_float(lo, hi, name):
    def _parse(arg):
        try:
            val = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a number between {lo} and {hi}: got {arg!r}")
        if not math.isfinite(val) or val < lo or val > hi:
            raise argparse.ArgumentTypeError(f"{name} must be between {lo} and {hi}: got {arg!r}")
        return val

    return _parse


def apply_muffle(mono, amount, sample_rate=SAMPLE_RATE):
    import numpy as np

    if amount <= 0.0 or not mono.any():
        return mono

    cutoff = 1500.0 - 1000.0 * amount
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    out = np.fft.irfft(np.fft.rfft(mono) / (1 + (freqs / cutoff) ** 2), len(mono))
    return (out * (np.abs(mono).max() / np.abs(out).max())).astype(np.float32)


def pan_stereo(mono, pan):
    import numpy as np

    pan = max(-1.0, min(1.0, float(pan)))
    theta = (pan + 1.0) * (math.pi / 4.0)
    return np.column_stack((mono * math.cos(theta), mono * math.sin(theta))).astype(np.float32)


def supports_stereo_output():
    import sounddevice as sd

    try:
        dev = sd.query_devices(kind="output")
        if isinstance(dev, dict):
            return dev.get("max_output_channels", 2) >= 2
    except Exception:
        pass
    return True


def bell_partials():
    return tuple((r, w, tau * TONE_LENGTH) for r, w, tau in BELL_PARTIALS)


def bell(freq, num_samples, partials, attack_s):
    import numpy as np

    t = np.arange(num_samples) / SAMPLE_RATE
    body = sum(w * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / tau) for r, w, tau in partials)
    envelope = np.minimum(t / attack_s, 1)
    release = min(num_samples, int(SAMPLE_RATE * BELL_RELEASE_S))
    envelope[num_samples - release:] *= np.arange(release - 1, -1, -1) / max(release, 1)
    return body * envelope


def place(out, mono, onset, pan):
    onset = max(onset, 0)
    n = min(len(mono), len(out) - onset)
    if n > 0:
        out[onset:onset + n] += pan_stereo(mono[:n], pan)


def render_phrase(root_hz, notes):
    import numpy as np

    partials = bell_partials()
    ring = int(SAMPLE_RATE * NOTE_RING_S * TONE_LENGTH)
    lead = int(SAMPLE_RATE * LEAD_S)
    starts = [lead + round(SAMPLE_RATE * ms * TEMPO / 1000) for ms in np.cumsum([0] + [ms for *_, ms in notes])]
    voiced = [(semi, vol, start) for (semi, vol, _), start in zip(notes, starts) if vol > 0]
    if not voiced:
        return None

    out = np.zeros((starts[-1] + ring, 2))
    grain = int(SAMPLE_RATE * STUTTER_GRAIN_S)
    grain_window = np.exp(-np.arange(grain) / (SAMPLE_RATE * STUTTER_DECAY_S))
    for k, (semi, vol, onset) in enumerate(voiced):
        freq, pan = semitone_to_hz(root_hz, semi), STEREO_SPREAD * (-1) ** k
        place(out, vol * bell(freq, ring, partials, BELL_ATTACK_S), onset, pan)
        if k in (0, len(voiced) - 1):
            stutter = STUTTER_LEVEL * vol * bell(freq, grain, partials, BELL_ATTACK_S) * grain_window
            for j in range(2, STUTTER_COUNT + 2):
                place(out, stutter, onset - int(SAMPLE_RATE * STUTTER_SPACING_S * j), pan)
    return out


def silence(ms):
    import numpy as np

    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def record_audio(seconds, sample_rate=SAMPLE_RATE):
    import sounddevice as sd

    print(f"[beep] recording {seconds:.1f}s...", file=sys.stderr)
    try:
        buf = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError:
        print("[beep] no mic available; skipping recording", file=sys.stderr)
        return None
    return buf[:, 0]


def compute_chroma(buffer, sample_rate, fmin=CHROMA_FMIN, fmax=CHROMA_FMAX):
    import numpy as np

    windowed = buffer * np.hanning(len(buffer))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(buffer), 1 / sample_rate)
    mask = (freqs >= fmin) & (freqs <= fmax)
    f, m = freqs[mask], mag[mask]

    chroma = np.zeros(12)
    if len(f) == 0:
        return chroma

    midi = 69 + 12 * np.log2(f / 440.0)
    pc = np.round(midi).astype(int) % 12
    np.add.at(chroma, pc, m)

    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def krumhansl_key_candidates(chroma):
    import numpy as np

    scores = []
    for mode, profile in (("major", np.array(MAJOR_PROFILE)), ("minor", np.array(MINOR_PROFILE))):
        for t in range(12):
            rotated = np.roll(profile, t)
            corr = np.corrcoef(chroma, rotated)[0, 1]
            scores.append((t, mode, corr))
    return sorted(scores, key=lambda x: x[2], reverse=True)


def format_key_candidates(candidates, n=3):
    return ", ".join(f"{PITCH_CLASSES[t]} {m} ({c:.2f})" for t, m, c in candidates[:n])


def pitch_class_to_hz(pc):
    return 440.0 * 2 ** ((pc - 9) / 12)


def best_key(buffer, sample_rate):
    chroma = compute_chroma(buffer, sample_rate)
    if chroma.sum() == 0:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return None
    candidates = krumhansl_key_candidates(chroma)
    print(f"[beep] key candidates: {format_key_candidates(candidates)}", file=sys.stderr)
    return candidates[0]


def detect_key(buffer, sample_rate):
    key = best_key(buffer, sample_rate)
    if key is None:
        return DEFAULT_ROOT_HZ, "silence"
    pc, mode, _ = key
    return pitch_class_to_hz(pc), f"detected key={PITCH_CLASSES[pc]} {mode}"


def detect_quiet_region(buffer, sample_rate):
    return detect_quiet_freq(buffer, sample_rate), "quietest region"


def detect_quiet_freq(buffer, sample_rate, band=QUIET_BAND, smooth_hz=QUIET_SMOOTH_HZ):
    import numpy as np

    windowed = buffer * np.hanning(len(buffer))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(buffer), 1 / sample_rate)
    freq_res = sample_rate / len(buffer)
    k = max(1, round(smooth_hz / freq_res))
    smoothed = np.convolve(mag, np.ones(k) / k, mode="same")

    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_freqs, band_mag = freqs[mask], smoothed[mask]

    if len(band_mag) == 0 or band_mag.max() < 1e-6:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    if band_mag.min() > 0.8 * band_mag.mean():
        print("[beep] no clear spectral gap; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    return float(band_freqs[np.argmin(band_mag)])


def classify_mic(buffer, sample_rate):
    key = best_key(buffer, sample_rate)
    if key is None:
        return DEFAULT_ROOT_HZ, "silence"
    pc, mode, corr = key
    if corr >= MUSIC_CONFIDENCE_THRESHOLD:
        return pitch_class_to_hz(pc), f"music: {PITCH_CLASSES[pc]} {mode} (key confidence {corr:.2f})"
    return detect_quiet_freq(buffer, sample_rate), f"talking: no clear key (confidence {corr:.2f}), using quietest region"


MIC_ANALYZERS = {"--auto-key": detect_key, "--avoid-speech": detect_quiet_region, "--mic": classify_mic}


def analyze_mic(flag, buffer):
    import numpy as np

    root_hz, what = MIC_ANALYZERS[flag](buffer, SAMPLE_RATE)
    frame = int(SAMPLE_RATE * AMBIENT_FRAME_S)
    frames = buffer[: len(buffer) // frame * frame].reshape(-1, frame)
    floor_rms = np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), AMBIENT_PERCENTILE)
    level_db = 20 * math.log10(max(float(floor_rms), 1e-9))
    gain = float(np.interp(level_db, AMBIENT_DB, AMBIENT_GAIN))
    return root_hz, gain, f"{what}, ambient {level_db:.0f} dBFS"


def build_arg_parser():
    parser = argparse.ArgumentParser(prog="beep.py")
    root_group = parser.add_mutually_exclusive_group(required=False)
    root_group.add_argument("--root", type=parse_root, default=None,
                             help="Hz number (523.25) or note name (C5, F#4, Bb3)")
    root_group.add_argument("--auto-key", nargs="?", type=parse_clamped_float(0.0, MAX_RECORD_SECONDS, "--auto-key"),
                             const=DEFAULT_AUTO_KEY_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_AUTO_KEY_SECONDS}) of mic audio, "
                                  "use detected key's tonic as root")
    root_group.add_argument("--avoid-speech", nargs="?", type=parse_clamped_float(0.0, MAX_RECORD_SECONDS, "--avoid-speech"),
                             const=DEFAULT_AVOID_SPEECH_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_AVOID_SPEECH_SECONDS}) of mic audio, "
                                  "use quietest region as root")
    root_group.add_argument("--mic", nargs="?", type=parse_clamped_float(0.0, MAX_RECORD_SECONDS, "--mic"),
                             const=DEFAULT_MIC_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_MIC_SECONDS}) of mic audio, "
                                  "auto-detect music vs talking and use --auto-key or --avoid-speech behavior")
    parser.add_argument("--pan", type=parse_clamped_float(-1.0, 1.0, "--pan"), default=None,
                        help="stereo pan from -1.0 (left) to 1.0 (right); overrides Windows detection")
    parser.add_argument("--muffle", type=parse_clamped_float(0.0, 1.0, "--muffle"), default=None,
                        help="muffle amount from 0.0 (clear) to 1.0 (behind a wall); overrides Windows detection")
    parser.add_argument("--ticks", type=int, choices=range(4), default=None,
                        help="clicks before the melody (virtual-desktop distance); overrides Windows detection")
    parser.add_argument("--verbose", action="store_true",
                        help="log resolved spatial settings to stderr")
    parser.add_argument("--hook", action="store_true",
                        help="read hook event JSON from stdin and emit hook JSON to stdout")
    parser.add_argument("--sync", action="store_true",
                        help="play audio synchronously in --hook mode instead of spawning a detached subprocess")
    parser.add_argument("--melody", choices=HOOK_MELODIES, default=None,
                        help="play a named melody instead of positional notes")
    parser.add_argument("--listen", action="store_true",
                        help="run the background mic daemon that serves cached mic-mode roots")
    parser.add_argument("notes", nargs="*", type=parse_note, metavar="semitones,vol,ms")
    return parser


def resolve_spatial(args, cwd=None):
    if None in (args.pan, args.muffle, args.ticks):
        import spatial

        pan, muffle, ticks, desc = spatial.detect_spatial(cwd)
    else:
        pan, muffle, ticks, desc = 0.0, 0.0, 0, "manual"
    pan = pan if args.pan is None else args.pan
    muffle = muffle if args.muffle is None else args.muffle
    ticks = ticks if args.ticks is None else args.ticks
    if args.verbose:
        print(f"[beep] spatial pan={pan:.2f} muffle={muffle:.2f} ticks={ticks} ({desc})", file=sys.stderr)
    return pan, muffle, ticks


def send_to_daemon(argv):
    import socket

    try:
        with socket.create_connection(("127.0.0.1", DAEMON_PORT), timeout=DAEMON_TIMEOUT_S) as conn:
            conn.sendall(json.dumps(argv).encode())
        return True
    except OSError:
        return False


def listen(args):
    import socket

    server = socket.socket()
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        server.bind(("127.0.0.1", DAEMON_PORT))
    except OSError:
        if args.notes:
            play_notes(*resolve_mic(args), args.notes, *resolve_spatial(args))
        return 0
    server.listen()
    server.settimeout(DAEMON_IDLE_S)

    import threading
    import numpy as np
    import sounddevice as sd

    ring = np.zeros(int(MAX_RECORD_SECONDS * SAMPLE_RATE), dtype=np.float32)
    state = {"pos": 0, "filled": 0, "stream": None}
    lock = threading.Lock()

    def on_audio(indata, frames, _time, _status):
        with lock:
            ring[(state["pos"] + np.arange(frames)) % len(ring)] = indata[:, 0]
            state["pos"] = (state["pos"] + frames) % len(ring)
            state["filled"] = min(len(ring), state["filled"] + frames)

    def recent(seconds):
        if state["stream"] is None:
            try:
                state["stream"] = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio)
                state["stream"].start()
            except sd.PortAudioError:
                return None
        with lock:
            n = min(int(seconds * SAMPLE_RATE), state["filled"])
            return np.roll(ring, -state["pos"])[-n:] if n >= DAEMON_MIN_MIC_S * SAMPLE_RATE else None

    def serve(request):
        if request.notes:
            play_notes(*resolve_mic(request, recent), request.notes, *resolve_spatial(request), wait=False)

    parser = build_arg_parser()
    with server:
        serve(args)
        while True:
            try:
                conn, _ = server.accept()
            except TimeoutError:
                break
            with conn:
                try:
                    conn.settimeout(DAEMON_TIMEOUT_S)
                    serve(parser.parse_args(json.loads(b"".join(iter(lambda: conn.recv(4096), b"")))))
                except (Exception, SystemExit) as e:
                    print(f"[beep] daemon error: {e!r}", file=sys.stderr)
    if state["stream"] is not None:
        state["stream"].close()
    return 0


def resolve_mic(args, source=record_audio):
    mic_modes = (("--auto-key", args.auto_key), ("--avoid-speech", args.avoid_speech), ("--mic", args.mic))
    for flag, seconds in mic_modes:
        if seconds is None:
            continue
        if seconds <= 0:
            raise SystemExit(f"{flag} SECONDS must be > 0")
        buf = source(seconds)
        root_hz, gain, what = (DEFAULT_ROOT_HZ, DEFAULT_GAIN, "default (no mic)") if buf is None else analyze_mic(flag, buf)
        desc = f"{what} ({flag} {seconds:.1f}s)"
        break
    else:
        root_hz, desc = (args.root, "--root") if args.root is not None else (DEFAULT_ROOT_HZ, "default")
        gain = DEFAULT_GAIN

    root_hz = fold_root(root_hz)
    print(f"[beep] root={root_hz:.2f} Hz gain={gain:.2f} ({desc})", file=sys.stderr)
    return root_hz, gain


def play_notes(root_hz, gain, notes, pan=0.0, muffle=0.0, ticks=0, wait=True):
    import numpy as np
    import sounddevice as sd

    audio = render_phrase(root_hz, notes)
    if audio is None:
        return
    audio = np.stack([apply_muffle(channel, muffle) for channel in audio.T], axis=1)
    if ticks:
        pluck = TICK_VOL * bell(2 * root_hz, int(SAMPLE_RATE * TICK_MS / 1000), bell_partials(), BELL_ATTACK_S)
        click = np.concatenate([pluck, silence(CLICK_GAP_MS)] * ticks
                               + [silence(CLICK_LEAD_MS)])
        audio = np.concatenate([np.column_stack((click, click)), audio])
    audio = pan_stereo(np.ones(1), pan) * math.sqrt(2) * audio
    audio = gain * audio / max(np.abs(audio).max(), 1e-9)
    if not supports_stereo_output():
        audio = audio.mean(axis=1)
    sd.play(audio.astype(np.float32), SAMPLE_RATE, latency="low")
    if wait:
        sd.wait()


def resolve_hook_event(payload: dict) -> tuple[str | None, dict]:
    if "hookEventName" in payload:
        event = payload.get("hookEventName", "")
        if event == "SessionStart":
            return ("SessionStart", {})
        if event == "PostToolUse":
            return ("PostToolUseFailure", {}) if payload.get("error") else (None, {})
        if event == "Stop":
            reason = payload.get("terminationReason", "")
            if reason in ANTIGRAVITY_FAILURE_REASONS or payload.get("error"):
                return ("StopFailure", {"decision": "allow"})
            if reason == "USER_CANCELED":
                return (None, {"decision": "allow"})
            return ("Stop", {"decision": "allow"})
        return (None, {})

    if "hook_event_name" in payload:
        event = payload.get("hook_event_name", "")
        if event == "Notification":
            key = f"Notification:{payload.get('notification_type', '')}"
            return (key if key in HOOK_MELODIES else None, {})
        return (event if event in HOOK_MELODIES else None, {})

    return (None, {})


def build_root_flags(args) -> list[str]:
    if args.auto_key is not None:
        return ["--auto-key", str(args.auto_key)]
    if args.avoid_speech is not None:
        return ["--avoid-speech", str(args.avoid_speech)]
    if args.mic is not None:
        return ["--mic", str(args.mic)]
    if args.root is not None:
        return ["--root", str(args.root)]
    return []


def play_argv(args, placement, notes):
    pan, muffle, ticks = placement
    return [*build_root_flags(args), f"--pan={pan:.4f}", f"--muffle={muffle:.4f}", f"--ticks={ticks}",
            "--", *(f"{s:g},{v:g},{ms:g}" for s, v, ms in notes)]


def spawn_detached(beep_args: list[str]) -> None:
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), *beep_args]
    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def run_hook(args) -> int:
    response = {}
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8", "replace").strip() or "{}")
        if not isinstance(payload, dict):
            payload = {}
        melody_key, response = resolve_hook_event(payload)
        melody = HOOK_MELODIES.get(melody_key) if melody_key else None
        if melody:
            cwd = payload.get("cwd")
            placement = resolve_spatial(args, cwd if isinstance(cwd, str) else None)
            notes = [parse_note(m) for m in melody]
            if args.sync:
                play_notes(*resolve_mic(args), notes, *placement)
            else:
                argv = play_argv(args, placement, notes)
                if not send_to_daemon(argv):
                    spawn_detached(["--listen", *argv])
    except Exception as e:
        print(f"[beep] hook error: {e!r}", file=sys.stderr)

    print(json.dumps(response))
    return 0


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.hook and args.notes:
        parser.error("positional notes cannot be combined with --hook")
    if args.listen:
        return listen(args)
    if args.melody:
        if args.hook or args.notes:
            parser.error("--melody cannot be combined with --hook or positional notes")
        args.notes = [parse_note(m) for m in HOOK_MELODIES[args.melody]]
    if not args.hook and not args.notes:
        parser.error("at least one semitones,vol,ms note is required unless --hook or --listen is set")

    if args.hook:
        return run_hook(args)

    placement = resolve_spatial(args)
    if not send_to_daemon(play_argv(args, placement, args.notes)):
        play_notes(*resolve_mic(args), args.notes, *placement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
