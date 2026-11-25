#!/usr/bin/env python3
"""Usage: beep.py [--root HZ|NOTE | --auto-key [SECONDS] | --avoid-speech [SECONDS] | --mic [SECONDS]] [--pan PAN] [--muffle MUFFLE] [--ticks N] [--verbose] [--sync] [--melody NAME | -- semitones,vol,ms ...]
       beep.py --listen
  semitones: signed offset from root (0 = root note), -48 to 48
  vol:       0.0-1.0; 0 is a rest
  ms:        time until the next note in milliseconds, 0-10000; notes ring past it

  --root         Hz number (523.25) or note name (C5, F#4, Bb3). Default: 440 (A4). Max 20000.
                 Folded by octaves into 400-800 Hz
  --auto-key     record SECONDS (default 2) of mic audio, use detected key's root (relative major for minor keys)
  --avoid-speech record SECONDS (default 1.5) of mic audio, use quietest region in 400-800 Hz as root
  --mic          record SECONDS (default 2) of mic audio, auto-detect music vs talking
                 (by key-detection confidence) and pick --auto-key or --avoid-speech behavior.
                 Mic modes also scale volume to background loudness
  --pan          stereo pan from -1.0 (left) to 1.0 (right); overrides Windows detection
  --muffle       muffle amount from 0.0 (clear) to 1.0 (behind a wall); overrides Windows detection
  --ticks        0-3 clicks before the melody (virtual-desktop distance); overrides Windows detection
  --verbose      log resolved spatial settings to stderr
  --sync         play audio synchronously in-process (with stderr logs) instead of via daemon
  --melody       play a named melody from MELODIES (done, fail, tool-fail, permission, question, idle, auth, alert)
  --listen       run the daemon: play requests sent on localhost sequentially, with mic modes
                 read from a rolling mic buffer. Requests start it on demand; it exits
                 after 10 idle minutes

Mic recording is capped at 30 seconds. Without a mic mode, volume is 0.25.

Note: positional notes must follow a `--` separator (negative offsets look like
flags to argparse otherwise), e.g. beep.py --root 440 -- -3,0.7,90
"""
import argparse
import json
import math
import pathlib
import re
import socket
import subprocess
import sys

DEFAULT_AUTO_KEY_SECONDS = 2.0
DEFAULT_AVOID_SPEECH_SECONDS = 1.5
DEFAULT_MIC_SECONDS = 2.0
MAX_ROOT_HZ = 20000.0
MAX_RECORD_SECONDS = 30.0
NOTE_LIMITS = ((-48.0, 48.0, "semitones"), (0.0, 1.0, "vol"), (0.0, 10000.0, "ms"))
DAEMON_PORT = 47392
DAEMON_TIMEOUT_S = 0.25

NOTE_LETTER_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
NOTE_NAME_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")

MELODIES: dict[str, tuple[str, ...]] = {
    "done": ("0,0.9,99",),
    "fail": ("0,0.9,110", "-3,0.95,110", "-6,1,185"),
    "tool-fail": ("5,0.7,90", "-1,0.7,220"),
    "permission": ("0,1,90", "7,1,120", "9,1,90", "14,1,160"),
    "question": ("4,1,80", "6,1,80", "2,1,80", "7,1,90", "11,1,260"),
    "idle": ("0,0.9,72", "0,0.9,110", "-5,0.95,72"),
    "auth": ("0,1,60", "4,1,60", "7,1,60", "12,1,60", "16,1,240"),
    "alert": ("6,1,110", "1,1,110", "6,1,260"),
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


def parse_positive_float(hi, name):
    def _parse(arg):
        try:
            val = float(arg)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{name} must be a number in (0, {hi}]: got {arg!r}")
        if not math.isfinite(val) or val <= 0.0 or val > hi:
            raise argparse.ArgumentTypeError(f"{name} must be in (0, {hi}]: got {arg!r}")
        return val

    return _parse


def parse_note(arg):
    parts = arg.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected semitones,vol,ms: got {arg!r}")
    return tuple(parse_clamped_float(*limits)(p) for limits, p in zip(NOTE_LIMITS, parts))


class BeepArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def build_arg_parser():
    parser = BeepArgumentParser(prog="beep.py")
    root_group = parser.add_mutually_exclusive_group(required=False)
    root_group.add_argument("--root", type=parse_root, default=None,
                            help="Hz number (523.25) or note name (C5, F#4, Bb3)")
    root_group.add_argument("--auto-key", nargs="?", type=parse_positive_float(MAX_RECORD_SECONDS, "--auto-key"),
                            const=DEFAULT_AUTO_KEY_SECONDS, default=None, metavar="SECONDS",
                            help=f"record SECONDS (default {DEFAULT_AUTO_KEY_SECONDS}) of mic audio, "
                                 "use detected key's root (relative major for minor keys)")
    root_group.add_argument("--avoid-speech", nargs="?", type=parse_positive_float(MAX_RECORD_SECONDS, "--avoid-speech"),
                            const=DEFAULT_AVOID_SPEECH_SECONDS, default=None, metavar="SECONDS",
                            help=f"record SECONDS (default {DEFAULT_AVOID_SPEECH_SECONDS}) of mic audio, "
                                 "use quietest region in 400-800 Hz as root")
    root_group.add_argument("--mic", nargs="?", type=parse_positive_float(MAX_RECORD_SECONDS, "--mic"),
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
    parser.add_argument("--sync", action="store_true",
                        help="play audio synchronously in-process (with stderr logs) instead of via daemon")
    parser.add_argument("--melody", choices=MELODIES, default=None,
                        help="play a named melody instead of positional notes")
    parser.add_argument("--listen", action="store_true",
                        help="run the background mic daemon that plays requests from a rolling mic buffer")
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


def send_to_daemon(argv, port=DAEMON_PORT):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=DAEMON_TIMEOUT_S) as conn:
            conn.settimeout(DAEMON_TIMEOUT_S)
            conn.sendall(json.dumps(argv).encode())
            conn.shutdown(socket.SHUT_WR)
            return conn.recv(16).strip() == b"ok"
    except OSError:
        return False


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


def dispatch_play(args, placement, notes, port=DAEMON_PORT):
    if args.sync:
        import audio

        audio.play_notes(*audio.resolve_mic(args), notes, *placement)
        return
    argv = play_argv(args, placement, notes)
    if not send_to_daemon(argv, port=port):
        spawn_detached(["--listen", *argv])


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.listen and args.sync:
        parser.error("--listen cannot be combined with --sync")
    if args.melody:
        if args.notes:
            parser.error("--melody cannot be combined with positional notes")
        args.notes = [parse_note(m) for m in MELODIES[args.melody]]
    if args.listen:
        import audio

        return audio.listen(args, parser.parse_args, resolve_spatial)
    if not args.notes:
        parser.error("at least one semitones,vol,ms note is required unless --listen is set")

    placement = resolve_spatial(args)
    dispatch_play(args, placement, args.notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
