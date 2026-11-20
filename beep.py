#!/usr/bin/env python3
"""Usage: beep.py [--root HZ|NOTE | --auto-key [SECONDS] | --avoid-speech [SECONDS] | --mic [SECONDS]] [--hook [--sync]] [-- semitones,vol,ms ...]
  semitones: signed offset from root (0 = root note)
  vol:       0.0-1.0
  ms:        duration in milliseconds

  --root         Hz number (523.25) or note name (C5, F#4, Bb3). Default: 440 (A4).
  --auto-key     record SECONDS (default 2) of mic audio, use detected key's tonic as root
  --avoid-speech record SECONDS (default 1.5) of mic audio, use quietest region as root
  --mic          record SECONDS (default 2) of mic audio, auto-detect music vs talking
                 (by key-detection confidence) and pick --auto-key or --avoid-speech behavior
  --hook         read LLM harness hook JSON from stdin, trigger matching
                 melody, and emit hook JSON to stdout
  --sync         play audio synchronously in --hook mode instead of spawning detached

All output is octave-folded to stay at or below 800 Hz.

Note: positional notes must follow a `--` separator (negative offsets look like
flags to argparse otherwise), e.g. beep.py --root 440 -- -3,0.7,90
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

SAMPLE_RATE = 44100
GAP_MS = 12
TAIL_MS = 150
DECAY_RATE = 5
ATTACK_S = 0.004

DEFAULT_ROOT_HZ = 440.0
DEFAULT_AUTO_KEY_SECONDS = 2.0
DEFAULT_AVOID_SPEECH_SECONDS = 1.5
DEFAULT_MIC_SECONDS = 2.0
MAX_HZ = 800.0
MUSIC_CONFIDENCE_THRESHOLD = 0.55
CHROMA_FMIN, CHROMA_FMAX = 65.0, 2100.0
QUIET_BAND = (200.0, 4000.0)
QUIET_SMOOTH_HZ = 30.0

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
    "SessionStart": ("0,0.9,55", "4,1.0,55", "7,1.0,80"),
    "Stop": ("0,0.9,80", "-4,0.9,110"),
    "StopFailure": ("0,0.9,90", "-3,0.95,90", "-6,1.0,160"),
    "PostToolUseFailure": ("0,0.9,70", "-3,1.0,120"),
    "PostCompact": ("0,0.9,45", "2,1.0,65"),
    "SubagentStop": ("0,0.9,55", "-3,0.95,75"),
    "Notification:permission_prompt": ("0,0.9,100", "3,1.0,80", "7,1.0,140"),
    "Notification:idle_prompt": ("0,0.9,55", "0,0.9,90", "-5,0.95,55"),
    "Notification:auth_success": ("0,0.9,45", "4,1.0,45", "7,1.0,45", "12,1.0,90"),
    "Notification:elicitation_dialog": ("0,0.9,55", "6,1.0,75", "3,0.9,55"),
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
        return float(root_arg)
    except ValueError:
        pass
    try:
        return parse_note_name(root_arg)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--root must be a Hz number or note name (e.g. 523.25, C5, F#4): got {root_arg!r}"
        )


def semitone_to_hz(root_hz, semitones):
    return root_hz * 2 ** (semitones / 12)


def octave_cap(freq, max_hz=MAX_HZ):
    while freq > max_hz:
        freq /= 2
    return freq


def parse_note_triple(arg):
    parts = arg.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected semitones,vol,ms: got {arg!r}")
    try:
        return tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected semitones,vol,ms (numbers): got {arg!r}")


def tone(freq, vol, ms):
    import numpy as np

    num_samples = int(SAMPLE_RATE * ms / 1000)
    if num_samples == 0 or vol <= 0:
        return np.zeros(num_samples, dtype=np.float32)

    duration_s = ms / 1000
    t = np.linspace(0, duration_s, num_samples, endpoint=False)

    envelope = np.exp(-DECAY_RATE * t / duration_s)

    attack_samples = min(int(SAMPLE_RATE * ATTACK_S), num_samples // 6)
    envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)

    return (vol * np.sin(2 * np.pi * freq * t) * envelope).astype(np.float32)


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


def krumhansl_best_key(chroma):
    return krumhansl_key_candidates(chroma)[0]


def format_key_candidates(candidates, n=3):
    return ", ".join(f"{PITCH_CLASSES[t]} {m} ({c:.2f})" for t, m, c in candidates[:n])


def pitch_class_to_hz(pc):
    return 440.0 * 2 ** ((pc - 9) / 12)


def detect_key(buffer, sample_rate):
    chroma = compute_chroma(buffer, sample_rate)
    if chroma.sum() == 0:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ, "silence"
    candidates = krumhansl_key_candidates(chroma)
    pc, mode, _ = candidates[0]
    print(f"[beep] key candidates: {format_key_candidates(candidates)}", file=sys.stderr)
    return pitch_class_to_hz(pc), f"{PITCH_CLASSES[pc]} {mode}"


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
    chroma = compute_chroma(buffer, sample_rate)
    if chroma.sum() == 0:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ, "silence"
    candidates = krumhansl_key_candidates(chroma)
    pc, mode, corr = candidates[0]
    print(f"[beep] key candidates: {format_key_candidates(candidates)}", file=sys.stderr)
    if corr >= MUSIC_CONFIDENCE_THRESHOLD:
        return pitch_class_to_hz(pc), f"music: {PITCH_CLASSES[pc]} {mode} (key confidence {corr:.2f})"
    return detect_quiet_freq(buffer, sample_rate), f"talking: no clear key (confidence {corr:.2f}), using quietest region"


def build_arg_parser():
    parser = argparse.ArgumentParser(prog="beep.py")
    root_group = parser.add_mutually_exclusive_group(required=False)
    root_group.add_argument("--root", type=parse_root, default=None,
                             help="Hz number (523.25) or note name (C5, F#4, Bb3)")
    root_group.add_argument("--auto-key", nargs="?", type=float,
                             const=DEFAULT_AUTO_KEY_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_AUTO_KEY_SECONDS}) of mic audio, "
                                  "use detected key's tonic as root")
    root_group.add_argument("--avoid-speech", nargs="?", type=float,
                             const=DEFAULT_AVOID_SPEECH_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_AVOID_SPEECH_SECONDS}) of mic audio, "
                                  "use quietest region as root")
    root_group.add_argument("--mic", nargs="?", type=float,
                             const=DEFAULT_MIC_SECONDS, default=None, metavar="SECONDS",
                             help=f"record SECONDS (default {DEFAULT_MIC_SECONDS}) of mic audio, "
                                  "auto-detect music vs talking and use --auto-key or --avoid-speech behavior")
    parser.add_argument("--hook", action="store_true",
                        help="read hook event JSON from stdin and emit hook JSON to stdout")
    parser.add_argument("--sync", action="store_true",
                        help="play audio synchronously in --hook mode instead of spawning a detached subprocess")
    parser.add_argument("notes", nargs="*", type=parse_note_triple, metavar="semitones,vol,ms")
    return parser


def resolve_root(args):
    if args.auto_key is not None:
        if args.auto_key <= 0:
            raise SystemExit("--auto-key SECONDS must be > 0")
        buf = record_audio(args.auto_key)
        if buf is None:
            root_hz, desc = DEFAULT_ROOT_HZ, "default (no mic)"
        else:
            root_hz, key_name = detect_key(buf, SAMPLE_RATE)
            desc = f"detected key={key_name} (--auto-key {args.auto_key:.1f}s)"
    elif args.avoid_speech is not None:
        if args.avoid_speech <= 0:
            raise SystemExit("--avoid-speech SECONDS must be > 0")
        buf = record_audio(args.avoid_speech)
        if buf is None:
            root_hz, desc = DEFAULT_ROOT_HZ, "default (no mic)"
        else:
            root_hz = detect_quiet_freq(buf, SAMPLE_RATE)
            desc = f"quietest region (--avoid-speech {args.avoid_speech:.1f}s)"
    elif args.mic is not None:
        if args.mic <= 0:
            raise SystemExit("--mic SECONDS must be > 0")
        buf = record_audio(args.mic)
        if buf is None:
            root_hz, desc = DEFAULT_ROOT_HZ, "default (no mic)"
        else:
            root_hz, classification = classify_mic(buf, SAMPLE_RATE)
            desc = f"{classification} (--mic {args.mic:.1f}s)"
    elif args.root is not None:
        root_hz, desc = args.root, "--root"
    else:
        root_hz, desc = DEFAULT_ROOT_HZ, "default"

    root_hz = octave_cap(root_hz)
    print(f"[beep] root={root_hz:.2f} Hz ({desc})", file=sys.stderr)
    return root_hz


def play_notes(root_hz, notes):
    import numpy as np
    import sounddevice as sd

    chunks = []
    for semitones, vol, ms in notes:
        freq = octave_cap(semitone_to_hz(root_hz, semitones))
        chunks.append(tone(freq, vol, ms))
        chunks.append(silence(GAP_MS))

    if chunks:
        sd.play(np.concatenate(chunks + [silence(TAIL_MS)]), SAMPLE_RATE)
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


def spawn_detached(root_flags: list[str], melody: tuple[str, ...]) -> None:
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), *root_flags, "--", *melody]
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
    raw = sys.stdin.read()
    payload = {}
    if raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            pass

    melody_key, response = resolve_hook_event(payload)
    melody = HOOK_MELODIES.get(melody_key) if melody_key else None
    if melody:
        if args.sync:
            play_notes(resolve_root(args), [parse_note_triple(m) for m in melody])
        else:
            spawn_detached(build_root_flags(args), melody)

    print(json.dumps(response))
    return 0


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.hook and args.notes:
        parser.error("positional notes cannot be combined with --hook")
    if not args.hook and not args.notes:
        parser.error("at least one semitones,vol,ms note is required unless --hook is set")

    if args.hook:
        return run_hook(args)

    play_notes(resolve_root(args), args.notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
