import json
import math
import queue
import socket
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
BELL_PARTIALS = ((1, 1.0, 0.275), (2, 0.22, 0.275), (3, 0.12, 0.2))
MUD_HZ = 500.0
TONE_LENGTH = 0.15
BELL_ATTACK_S = 0.015
OVERTONE_RISE_S = 0.03
BELL_RELEASE_S = 0.013
NOTE_RING_S = 0.68
TEMPO = 0.93
LEAD_S = 0.001
STEREO_SPREAD = 0.24
AMBIENT_DB = (-55.0, -25.0)
AMBIENT_GAIN = (0.25, 1.0)
DEFAULT_GAIN = 0.25
AMBIENT_FRAME_S = 0.05
AMBIENT_PERCENTILE = 50
TICK_HZ = 2200.0
TICK_VOL = 1.0
TICK_MS = 8
LOW_CUT_HZ = 500.0
CLICK_GAP_MS = 85
CLICK_LEAD_MS = 60

DEFAULT_ROOT_HZ = 440.0
ROOT_BAND_HZ = 800.0
MAX_RECORD_SECONDS = 30.0
MUSIC_CONFIDENCE_THRESHOLD = 0.55
CHROMA_FMIN, CHROMA_FMAX = 65.0, 2100.0
QUIET_SMOOTH_HZ = 30.0
DAEMON_PORT = 47392
DAEMON_IDLE_S = 600
DAEMON_TIMEOUT_S = 0.25
DAEMON_MIN_MIC_S = 0.5
DAEMON_COLD_WAIT_S = 0.6
DAEMON_MAX_MSG_BYTES = 16384
PLAYBACK_MUTE_TAIL_S = 0.15

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def semitone_to_hz(root_hz, semitones):
    return root_hz * 2 ** (semitones / 12)


def fold_root(freq, band_hz=ROOT_BAND_HZ):
    while freq > band_hz:
        freq /= 2
    while freq <= band_hz / 2:
        freq *= 2
    return freq


def spectral_filter(mono, response, sample_rate=SAMPLE_RATE):
    freqs = np.fft.rfftfreq(len(mono), 1 / sample_rate)
    return np.fft.irfft(np.fft.rfft(mono) * response(freqs), len(mono))


def low_cut(mono, cutoff=LOW_CUT_HZ):
    return spectral_filter(mono, lambda f: (f / cutoff) ** 2 / (1 + (f / cutoff) ** 2))


def apply_muffle(mono, amount):
    if amount <= 0.0 or not mono.any():
        return mono

    cutoff = 1500.0 - 1000.0 * amount
    out = spectral_filter(mono, lambda f: 1 / (1 + (f / cutoff) ** 2))
    out_peak = np.abs(out).max()
    if out_peak <= 0.0:
        return np.zeros_like(mono, dtype=np.float32)
    return (out * (np.abs(mono).max() / out_peak)).astype(np.float32)


def pan_gains(pan):
    clamped = max(-1.0, min(1.0, float(pan)))
    theta = (clamped + 1.0) * (math.pi / 4.0)
    return (math.cos(theta), math.sin(theta))


def pan_stereo(mono, pan):
    gl, gr = pan_gains(pan)
    return np.column_stack((mono * gl, mono * gr)).astype(np.float32)


def supports_stereo_output():
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
    t = np.arange(num_samples) / SAMPLE_RATE
    rise = 1 - np.exp(-t / OVERTONE_RISE_S)
    weights = [w / (1 + (MUD_HZ / (freq * r)) ** 8) for r, w, _ in partials]
    scale = sum(w for _, w, _ in partials) / sum(weights)
    body = scale * sum(
        w * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / tau) * (rise if r > 1 else 1)
        for (r, _, tau), w in zip(partials, weights)
    )
    envelope = np.minimum(t / attack_s, 1)
    release = min(num_samples, int(SAMPLE_RATE * BELL_RELEASE_S))
    envelope[num_samples - release:] *= np.arange(release - 1, -1, -1) / max(release, 1)
    return body * envelope


def place(out, mono, onset, pan):
    n = min(len(mono), len(out) - onset)
    if n > 0:
        out[onset:onset + n] += pan_stereo(mono[:n], pan)


def render_phrase(root_hz, notes):
    partials = bell_partials()
    ring = int(SAMPLE_RATE * NOTE_RING_S * TONE_LENGTH)
    lead = int(SAMPLE_RATE * LEAD_S)
    starts = [lead + round(SAMPLE_RATE * ms * TEMPO / 1000) for ms in np.cumsum([0] + [ms for *_, ms in notes])]
    voiced = [(semi, vol, start) for (semi, vol, _), start in zip(notes, starts) if vol > 0]
    if not voiced:
        return None

    total = max(starts[-1], voiced[-1][2] + ring)
    out = np.zeros((total, 2))
    spread = STEREO_SPREAD if len(voiced) > 1 else 0.0
    for k, (semi, vol, onset) in enumerate(voiced):
        freq, pan = semitone_to_hz(root_hz, semi), spread * (-1) ** k
        place(out, vol * bell(freq, ring, partials, BELL_ATTACK_S), onset, pan)
    return out


def silence(ms):
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def record_audio(seconds, sample_rate=SAMPLE_RATE):
    print(f"[beep] recording {seconds:.1f}s...", file=sys.stderr)
    try:
        buf = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except sd.PortAudioError:
        print("[beep] no mic available; skipping recording", file=sys.stderr)
        return None
    return buf[:, 0]


def compute_chroma(buffer, sample_rate, fmin=CHROMA_FMIN, fmax=CHROMA_FMAX):
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
    counts = np.zeros(12)
    np.add.at(chroma, pc, m)
    np.add.at(counts, pc, 1)
    chroma /= np.maximum(counts, 1)

    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def krumhansl_key_candidates(chroma):
    if np.std(chroma) == 0:
        return [(t, mode, 0.0) for mode in ("major", "minor") for t in range(12)]
    scores = []
    for mode, profile in (("major", np.array(MAJOR_PROFILE)), ("minor", np.array(MINOR_PROFILE))):
        for t in range(12):
            rotated = np.roll(profile, t)
            corr = float(np.corrcoef(chroma, rotated)[0, 1])
            scores.append((t, mode, corr))
    return sorted(scores, key=lambda x: x[2], reverse=True)


def format_key_candidates(candidates, n=3):
    return ", ".join(f"{PITCH_CLASSES[t]} {m} ({c:.2f})" for t, m, c in candidates[:n])


def key_root_hz(pc, mode):
    return 440.0 * 2 ** ((pc + (3 if mode == "minor" else 0) - 9) / 12)


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
    return key_root_hz(pc, mode), f"detected key={PITCH_CLASSES[pc]} {mode}"


def detect_quiet_region(buffer, sample_rate):
    return detect_quiet_freq(buffer, sample_rate), "quietest region"


def detect_quiet_freq(buffer, sample_rate, band=(ROOT_BAND_HZ / 2, ROOT_BAND_HZ), smooth_hz=QUIET_SMOOTH_HZ):
    windowed = buffer * np.hanning(len(buffer))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(buffer), 1 / sample_rate)
    freq_res = sample_rate / len(buffer)
    k = max(1, round(smooth_hz / freq_res))
    smoothed = np.convolve(mag, np.ones(k) / k, mode="same")

    mask = (freqs > band[0]) & (freqs <= band[1])
    band_freqs = freqs[mask]
    if len(band_freqs) == 0:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ

    score = sum(w * np.interp(band_freqs * r, freqs, smoothed) for r, w, _ in BELL_PARTIALS)
    if score.max() < 1e-6:
        print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    if score.min() > 0.8 * score.mean():
        print("[beep] no clear spectral gap; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    return float(band_freqs[np.argmin(score)])


def classify_mic(buffer, sample_rate):
    key = best_key(buffer, sample_rate)
    if key is None:
        return DEFAULT_ROOT_HZ, "silence"
    pc, mode, corr = key
    if corr >= MUSIC_CONFIDENCE_THRESHOLD:
        return key_root_hz(pc, mode), f"music: {PITCH_CLASSES[pc]} {mode} (key confidence {corr:.2f})"
    return detect_quiet_freq(buffer, sample_rate), f"talking: no clear key (confidence {corr:.2f}), using quietest region"


MIC_ANALYZERS = {"--auto-key": detect_key, "--avoid-speech": detect_quiet_region, "--mic": classify_mic}


def analyze_mic(flag, buffer):
    root_hz, what = MIC_ANALYZERS[flag](buffer, SAMPLE_RATE)
    frame = int(SAMPLE_RATE * AMBIENT_FRAME_S)
    if len(buffer) < frame:
        floor_rms = float(np.sqrt(np.mean(buffer ** 2))) if len(buffer) else 0.0
    else:
        frames = buffer[: len(buffer) // frame * frame].reshape(-1, frame)
        floor_rms = float(np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), AMBIENT_PERCENTILE))
    level_db = 20 * math.log10(max(floor_rms, 1e-9))
    gain = float(np.interp(level_db, AMBIENT_DB, AMBIENT_GAIN))
    return root_hz, gain, f"{what}, ambient {level_db:.0f} dBFS"


def resolve_mic(args, source=record_audio):
    mic_modes = (("--auto-key", args.auto_key), ("--avoid-speech", args.avoid_speech), ("--mic", args.mic))
    for flag, seconds in mic_modes:
        if seconds is None:
            continue
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
    audio = render_phrase(root_hz, notes)
    if audio is None:
        return
    audio = np.stack([apply_muffle(low_cut(channel), muffle) for channel in audio.T], axis=1)
    audio /= max(np.abs(audio).max(), 1e-9)
    if ticks:
        tick_s = TICK_MS / 1000
        tick = TICK_VOL * bell(TICK_HZ, int(SAMPLE_RATE * tick_s), ((1, 1.0, tick_s / 5),), tick_s / 6)
        click = np.concatenate([tick, silence(CLICK_GAP_MS)] * ticks
                               + [silence(CLICK_LEAD_MS)])
        audio = np.concatenate([np.column_stack((click, click)), audio])
    if supports_stereo_output():
        gl, gr = pan_gains(pan)
        audio = np.clip(audio * (np.array([gl, gr], dtype=np.float32) * (math.sqrt(2) * gain)), -1.0, 1.0)
    else:
        audio = audio.mean(axis=1)
        audio = gain * audio / max(np.abs(audio).max(), 1e-9)
    sd.play(audio.astype(np.float32), SAMPLE_RATE, latency="low")
    if wait:
        sd.wait()


def listen(args, parse_request, resolve_spatial_fn, port=DAEMON_PORT):
    """Run the rolling-buffer mic daemon and sequential playback worker."""
    server = socket.socket()
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        server.bind(("127.0.0.1", port))
    except OSError:
        if args.notes:
            play_notes(*resolve_mic(args), args.notes, *resolve_spatial_fn(args))
        return 0
    server.listen()
    server.settimeout(DAEMON_IDLE_S)

    ring = np.zeros(int(MAX_RECORD_SECONDS * SAMPLE_RATE), dtype=np.float32)
    state = {"pos": 0, "filled": 0, "stream": None, "mute_until": 0.0}
    lock = threading.Lock()
    play_queue: queue.Queue = queue.Queue()

    def close_stream():
        stream = state["stream"]
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
            state["stream"] = None

    def on_audio(indata, frames, _time, _status):
        if time.monotonic() < state["mute_until"]:
            return
        src = indata[:, 0]
        if frames >= len(ring):
            src = src[-len(ring):]
            frames = len(ring)
        with lock:
            pos = state["pos"]
            first = min(frames, len(ring) - pos)
            ring[pos:pos + first] = src[:first]
            if first < frames:
                ring[:frames - first] = src[first:]
            state["pos"] = (pos + frames) % len(ring)
            state["filled"] = min(len(ring), state["filled"] + frames)

    def ensure_stream():
        stream = state["stream"]
        if stream is not None and stream.active:
            return True
        close_stream()
        with lock:
            state["pos"] = 0
            state["filled"] = 0
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio)
            stream.start()
            state["stream"] = stream
            return True
        except sd.PortAudioError:
            return False

    def recent(seconds):
        if not ensure_stream():
            return None
        min_samples = max(1, int(min(seconds, DAEMON_MIN_MIC_S) * SAMPLE_RATE))
        deadline = time.monotonic() + DAEMON_COLD_WAIT_S
        while state["filled"] < min_samples and time.monotonic() < deadline:
            if state["stream"] is None or not state["stream"].active:
                return None
            time.sleep(0.025)
        with lock:
            filled = state["filled"]
            if filled < min_samples:
                return None
            n = min(int(seconds * SAMPLE_RATE), filled)
            if n <= 0:
                return None
            pos = state["pos"]
            if pos >= n:
                return ring[pos - n:pos].copy()
            head = n - pos
            return np.concatenate((ring[len(ring) - head:], ring[:pos]))

    def worker_loop():
        while True:
            request = play_queue.get()
            if request is None:
                break
            try:
                root_hz, gain = resolve_mic(request, recent)
                placement = resolve_spatial_fn(request)
                state["mute_until"] = float("inf")
                try:
                    play_notes(root_hz, gain, request.notes, *placement, wait=True)
                except sd.PortAudioError:
                    close_stream()
                    sd._terminate()
                    sd._initialize()
                    play_notes(root_hz, gain, request.notes, *placement, wait=True)
                finally:
                    state["mute_until"] = time.monotonic() + PLAYBACK_MUTE_TAIL_S
            except Exception as e:
                print(f"[beep] daemon playback error: {e!r}", file=sys.stderr)

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    with server:
        if args.notes:
            play_queue.put(args)
        while True:
            try:
                conn, _ = server.accept()
            except TimeoutError:
                break
            with conn:
                try:
                    conn.settimeout(DAEMON_TIMEOUT_S)
                    chunks = []
                    total = 0
                    while True:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > DAEMON_MAX_MSG_BYTES:
                            raise ValueError("payload too large")
                        chunks.append(chunk)
                    raw = json.loads(b"".join(chunks))
                    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
                        raise ValueError("expected JSON list of strings")
                    request = parse_request(raw)
                    if request.notes:
                        play_queue.put(request)
                    conn.sendall(b"ok\n")
                except (Exception, SystemExit) as e:
                    print(f"[beep] daemon error: {e!r}", file=sys.stderr)

    play_queue.put(None)
    worker.join(timeout=5.0)
    close_stream()
    return 0
