# conso-beep

Ambient-aware tone generator and lifecycle hook runner for LLM agent harnesses.

See the module docstring at the top of `beep.py` for CLI flags, options, and defaults.

## Dependencies

Python 3.10+.

```bash
pip install sounddevice numpy
```

## Usage

**Direct CLI:**

```bash
python beep.py --root C5 -- 0,0.4,55 4,0.5,55 7,0.55,80
```

On Windows, the beep pans toward the monitor or virtual desktop it came from. Beeps from other desktops sound muffled, with one pluck per desktop away.

**Hook runner (`--hook`):**

Reads hook JSON from `stdin`, plays the melody for that event in a detached process, and prints valid hook JSON to `stdout`. Pass `--sync` to block until playback finishes.

## Installation

Point the harness lifecycle hooks at `python <path>/beep.py --hook --mic`. For harnesses that load plugins from a folder, clone the repo there; `plugin.json` and `hooks.json` are the manifest.

`--mic` picks a pitch that fits the room and a volume that fits its background noise. The first call starts a background mic daemon (`--listen`) and records for 2 seconds. Later calls read the daemon's rolling buffer and play at once. The daemon exits after 10 idle minutes.
