# conso-beep

Ambient-aware sine-wave tone generator and unified lifecycle hook runner for Antigravity and Claude Code.

See the module docstring at the top of `beep.py` for the authoritative CLI flags, options, and defaults.

## Dependencies

```bash
pip install sounddevice numpy
```

## Usage

**Direct CLI:**

```bash
python beep.py --root C5 -- 0,0.4,55 4,0.5,55 7,0.55,80
```

**Unified hook runner (`--hook`):**

Reads hook JSON from `stdin` (auto-detecting Antigravity camelCase vs. Claude Code snake_case payloads), launches matching tone playback detached so the agent loop does not stall on `--mic` sampling, and prints valid hook JSON to `stdout`. Pass `--sync` to block until playback finishes.

## Installation

- **Antigravity:** Clone into `~/.gemini/config/plugins/conso-beep/` (`plugin.json` and `hooks.json` are loaded automatically).
- **Claude Code:** Add as a submodule (e.g. `~/.claude/scripts/conso-beep/`) and point hooks in `settings.json` to `python <path>/beep.py --hook --mic`.
