# conso-beep

Ambient-aware beeps for LLM agent harness hooks. Flags are in the `beep.py` docstring.

## Install

Python 3.10+.

```bash
pip install sounddevice numpy
```

Point the harness hooks at `python -S <path>/beep.py --hook --mic`, or clone the repo into the harness plugin folder (`plugin.json` and `hooks.json` are the manifest).

## How it works

- `--hook` reads hook JSON from `stdin`, prints hook JSON to `stdout`, and sends the event's melody to a background daemon. It starts the daemon if none runs. The daemon exits after 10 idle minutes.
- `-S` makes the hook start fast. `--sync` plays in-process instead, and needs plain `python`.
- `--mic` matches pitch and volume to the room.
- On Windows, the beep pans toward the source monitor. Beeps from other virtual desktops sound muffled, with one pluck per desktop away.

## CLI

```bash
python beep.py --root C5 -- 0,0.4,55 4,0.5,55 7,0.55,80
```
