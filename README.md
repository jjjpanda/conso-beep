# conso-beep

Ambient-aware sine-wave tone generator and lifecycle hook runner for LLM agent harnesses.

See the module docstring at the top of `beep.py` for CLI flags, options, and defaults.

## Dependencies

```bash
pip install sounddevice numpy
```

## Usage

**Direct CLI:**

```bash
python beep.py --root C5 -- 0,0.4,55 4,0.5,55 7,0.55,80
```

**Hook runner (`--hook`):**

Reads hook JSON from `stdin`, plays the melody for that event in a detached process, and prints valid hook JSON to `stdout`. Pass `--sync` to block until playback finishes.

## Installation

Point the harness lifecycle hooks at `python <path>/beep.py --hook --mic`. For harnesses that load plugins from a folder, clone the repo there; `plugin.json` and `hooks.json` are the manifest.
