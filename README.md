# conso-beep

Ambient-aware audio tones, live mic dashboard, and permission inbox. Flags are in the `beep.py` docstring.

## Install

Python 3.10+.

```bash
pip install sounddevice numpy
```

## Layout

- `beep.py` — stdlib-only CLI, `--ask` client, and daemon dispatcher (`python -S` fast path).
- `audio.py` — DSP bell synthesis, mic analysis, and rolling `MicRing` telemetry.
- `daemon.py` — localhost HTTP server (`http://127.0.0.1:47392`), playback worker, and permission inbox.
- `dashboard.html` — live room status, quick controls, and multi-permission queue UI.
- `spatial.py` — Windows monitor panning and virtual-desktop muffling/ticks.
- `tests/` — `unittest` suite (`python -m unittest discover -s tests -v`).

## How it works

- Requests send their melody to the background HTTP daemon on `http://127.0.0.1:47392`, starting it on demand (`--listen`). The daemon exits after 10 idle minutes unless `Keep Alive` is toggled in the dashboard. Pass `--sync` to play in-process with stderr logs.
- `--melody NAME` plays a built-in tone (`done`, `fail`, `tool-fail`, `permission`, `question`, `idle`, `auth`, `alert`).
- `--ask TITLE` posts a permission request to the dashboard inbox, plays an alert, and prints `{"decision": "allow"|"deny"|"timeout", "reason": "..."}` once resolved.
- `--mic` matches pitch (or avoids speech in `400-800 Hz`) and scales volume to room loudness.
- On Windows, the beep pans toward the active window's monitor. Beeps from other virtual desktops sound muffled, with one tick per desktop away.

## CLI

```bash
python beep.py --root C5 -- 0,0.4,55 4,0.5,55 7,0.55,80
python -S beep.py --mic --melody done
python -S beep.py --ask "Run tests?" --detail "pytest -q" --source cli
```
