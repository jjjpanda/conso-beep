import math
import sys
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
TIMBRES = {
    "sine": (((1, 1.0, 0.15), (2, 0.05, 0.035)), 0.005, 0.34),
    "marimba": (((1, 1.0, 0.075), (2, 0.06, 0.022), (4, 0.22, 0.012)), 0.002, 0.22),
}
DEFAULT_TIMBRE = "sine"
RELEASE_S = 0.025
TAIL_S = 0.04
TEMPO = 0.93
LEAD_S = 0.002
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
MUSIC_CONFIDENCE_THRESHOLD = 0.35
CHROMA_FMIN, CHROMA_FMAX = 130.0, 2100.0
CHROMA_FFT = 8192
CHROMA_HOP = 4096
TONAL_SMOOTH_HZ = 150.0
TONAL_NOISE_GATE = 1.2
KEY_WINDOW_S = 15.0
MUSIC_EXIT_THRESHOLD = 0.22
KEY_SWITCH_MARGIN = 0.06
QUIET_SMOOTH_HZ = 30.0
DAEMON_MIN_MIC_S = 0.5
DAEMON_COLD_WAIT_S = 0.6
PLAYBACK_MUTE_TAIL_S = 0.15
STATUS_CACHE_TTL_S = 1.0

KEY_SIGNATURES = [
    "C / Am",
    "Db / Bbm",
    "D / Bm",
    "Eb / Cm",
    "E / C#m",
    "F / Dm",
    "F# / D#m",
    "G / Em",
    "Ab / Fm",
    "A / F#m",
    "Bb / Gm",
    "B / G#m",
]
DIATONIC_MASK = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
DIATONIC_WEIGHTS = np.array([1.04, 0.0, 0.96, 0.0, 1.00, 0.98, 0.0, 1.02, 0.0, 1.01, 0.0, 0.95], dtype=float)


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


def bell(freq, num_samples, partials, attack_s):
    t = np.arange(num_samples) / SAMPLE_RATE
    body = sum(w * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / tau) for r, w, tau in partials)
    att_n = max(1, min(num_samples, int(SAMPLE_RATE * attack_s)))
    env = np.ones(num_samples, dtype=np.float64)
    env[:att_n] = 0.5 * (1 - np.cos(np.pi * np.arange(att_n) / att_n))
    rel_n = min(num_samples, int(SAMPLE_RATE * RELEASE_S))
    env[num_samples - rel_n:] *= 0.5 * (1 + np.cos(np.pi * np.arange(rel_n) / rel_n))
    return body * env


def place(out, mono, onset, pan):
    n = min(len(mono), len(out) - onset)
    if n > 0:
        out[onset:onset + n] += pan_stereo(mono[:n], pan)


def render_phrase(root_hz, notes, timbre=DEFAULT_TIMBRE):
    partials, attack_s, ring_s = TIMBRES.get(timbre, TIMBRES[DEFAULT_TIMBRE])
    ring = int(SAMPLE_RATE * ring_s)
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
        place(out, vol * bell(freq, ring, partials, attack_s), onset, pan)
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
    if len(buffer) == 0:
        return np.zeros(12)
    buf = buffer.astype(np.float64, copy=False)
    sig_rms = float(np.sqrt(np.mean(buf ** 2)))
    if sig_rms < 1e-7:
        return np.zeros(12)
    buf = buf / sig_rms

    if len(buf) < CHROMA_FFT:
        frames = buf[None, :]
        win = np.hanning(len(buf))
    else:
        n_frames = 1 + (len(buf) - CHROMA_FFT) // CHROMA_HOP
        idx = np.arange(CHROMA_FFT)[None, :] + (np.arange(n_frames) * CHROMA_HOP)[:, None]
        frames = buf[idx]
        win = np.hanning(CHROMA_FFT)

    mag = np.abs(np.fft.rfft(frames * win, axis=1))
    freqs = np.fft.rfftfreq(frames.shape[1], 1 / sample_rate)
    df = sample_rate / frames.shape[1]
    k_smooth = max(3, min(mag.shape[1], int(round(TONAL_SMOOTH_HZ / df)) | 1))
    pad_left = (k_smooth + 1) // 2
    pad_right = k_smooth - pad_left
    cs = np.cumsum(np.pad(mag, ((0, 0), (pad_left, pad_right)), mode="edge"), axis=1)
    baseline = (cs[:, k_smooth:] - cs[:, :-k_smooth]) / k_smooth
    excess = np.maximum(mag - TONAL_NOISE_GATE * baseline, 0.0) / (baseline + 1e-9)
    tonal_log = np.log1p(2.0 * excess)

    mask = (freqs >= fmin) & (freqs <= fmax)
    f, v = freqs[mask], tonal_log[:, mask]
    if len(f) == 0:
        return np.zeros(12)

    pc = np.round(69 + 12 * np.log2(f / 440.0)).astype(int) % 12
    w = 1.0 / f
    frame_chroma = np.zeros((v.shape[0], 12), dtype=np.float64)
    for k in range(12):
        cols = pc == k
        if cols.any():
            wk = w[cols]
            frame_chroma[:, k] = (v[:, cols] @ wk) / wk.sum()

    totals = frame_chroma.sum(axis=1)
    med = float(np.median(totals))
    active = frame_chroma[totals > max(med * 0.2, 1e-12)]
    if len(active) == 0:
        return np.zeros(12)

    normed = active / active.sum(axis=1, keepdims=True)
    chroma = normed.mean(axis=0)
    total = chroma.sum()
    return chroma / total if total > 0 else chroma


def key_signature_candidates(chroma):
    if chroma.sum() == 0:
        return [(t, 0.0) for t in range(12)]
    results = []
    for t in range(12):
        in_scale = float(np.dot(chroma, np.roll(DIATONIC_MASK, t)))
        conf = max(0.0, min(1.0, (in_scale - 7.0 / 12.0) / (5.0 / 12.0)))
        tie = float(np.dot(chroma, np.roll(DIATONIC_WEIGHTS, t)))
        results.append((t, conf, tie))
    results.sort(key=lambda x: (round(x[1], 4), x[2]), reverse=True)
    return [(t, conf) for t, conf, _ in results]


def format_key_candidates(candidates, n=3):
    return ", ".join(f"{KEY_SIGNATURES[t]} ({c:.2f})" for t, c in candidates[:n])


def key_root_hz(pc):
    return 440.0 * 2 ** ((pc - 9) / 12)


def best_key(buffer, sample_rate, quiet=False):
    chroma = compute_chroma(buffer, sample_rate)
    if chroma.sum() == 0:
        if not quiet:
            print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return None
    candidates = key_signature_candidates(chroma)
    if not quiet:
        print(f"[beep] key signatures: {format_key_candidates(candidates)}", file=sys.stderr)
    return candidates[0]


def detect_key(buffer, sample_rate):
    key = best_key(buffer, sample_rate)
    if key is None:
        return DEFAULT_ROOT_HZ, DEFAULT_TIMBRE, "silence"
    pc, _ = key
    return key_root_hz(pc), "sine", f"detected key={KEY_SIGNATURES[pc]}"


def detect_quiet_region(buffer, sample_rate):
    return detect_quiet_freq(buffer, sample_rate), "marimba", "quietest region"


def detect_quiet_freq(buffer, sample_rate, band=(ROOT_BAND_HZ / 2, ROOT_BAND_HZ), smooth_hz=QUIET_SMOOTH_HZ, quiet=False):
    windowed = buffer * np.hanning(len(buffer))
    mag = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(buffer), 1 / sample_rate)
    freq_res = sample_rate / len(buffer)
    k = max(1, round(smooth_hz / freq_res))
    smoothed = np.convolve(mag, np.ones(k) / k, mode="same")

    mask = (freqs > band[0]) & (freqs <= band[1])
    band_freqs = freqs[mask]
    if len(band_freqs) == 0:
        if not quiet:
            print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ

    score = sum(w * np.interp(band_freqs * r, freqs, smoothed) for r, w, _ in TIMBRES["marimba"][0])
    if score.max() < 1e-6:
        if not quiet:
            print("[beep] no signal detected; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    if score.min() > 0.8 * score.mean():
        if not quiet:
            print("[beep] no clear spectral gap; falling back to 440 Hz", file=sys.stderr)
        return DEFAULT_ROOT_HZ
    return float(band_freqs[np.argmin(score)])


def classify_mic(buffer, sample_rate):
    key = best_key(buffer, sample_rate)
    if key is None:
        return DEFAULT_ROOT_HZ, DEFAULT_TIMBRE, "silence"
    pc, conf = key
    if conf >= MUSIC_CONFIDENCE_THRESHOLD:
        return key_root_hz(pc), "sine", f"music: {KEY_SIGNATURES[pc]} (confidence {conf:.2f})"
    return detect_quiet_freq(buffer, sample_rate), "marimba", f"talking: no clear key signature (confidence {conf:.2f}), using quietest region"


MIC_ANALYZERS = {"--auto-key": detect_key, "--avoid-speech": detect_quiet_region, "--mic": classify_mic}


def ambient_level(buffer):
    frame = int(SAMPLE_RATE * AMBIENT_FRAME_S)
    if len(buffer) < frame:
        floor_rms = float(np.sqrt(np.mean(buffer ** 2))) if len(buffer) else 0.0
    else:
        frames = buffer[: len(buffer) // frame * frame].reshape(-1, frame)
        floor_rms = float(np.percentile(np.sqrt(np.mean(frames ** 2, axis=1)), AMBIENT_PERCENTILE))
    level_db = 20 * math.log10(max(floor_rms, 1e-9))
    gain = float(np.interp(level_db, AMBIENT_DB, AMBIENT_GAIN))
    return level_db, gain


def analyze_mic(flag, buffer):
    root_hz, timbre, what = MIC_ANALYZERS[flag](buffer, SAMPLE_RATE)
    level_db, gain = ambient_level(buffer)
    return root_hz, gain, timbre, f"{what}, ambient {level_db:.0f} dBFS"


def resolve_mic(args, source=record_audio, gain_override=None, timbre_override=None):
    mic_modes = (("--auto-key", args.auto_key), ("--avoid-speech", args.avoid_speech), ("--mic", args.mic))
    for flag, seconds in mic_modes:
        if seconds is None:
            continue
        buf = source(seconds)
        if buf is None:
            root_hz, gain, timbre, what = DEFAULT_ROOT_HZ, DEFAULT_GAIN, DEFAULT_TIMBRE, "default (no mic)"
        else:
            root_hz, gain, timbre, what = analyze_mic(flag, buf)
        desc = f"{what} ({flag} {seconds:.1f}s)"
        break
    else:
        root_hz, desc = (args.root, "--root") if args.root is not None else (DEFAULT_ROOT_HZ, "default")
        gain, timbre = DEFAULT_GAIN, DEFAULT_TIMBRE

    if gain_override is not None:
        gain = max(0.0, min(1.0, float(gain_override)))
        desc = f"{desc}, override"
    if timbre_override in TIMBRES:
        timbre = timbre_override

    root_hz = fold_root(root_hz)
    print(f"[beep] root={root_hz:.2f} Hz gain={gain:.2f} tone={timbre} ({desc})", file=sys.stderr)
    return root_hz, gain, timbre


def play_notes(root_hz, gain, notes, pan=0.0, muffle=0.0, ticks=0, wait=True, timbre=DEFAULT_TIMBRE):
    audio = render_phrase(root_hz, notes, timbre=timbre)
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
    tail_n = int(SAMPLE_RATE * TAIL_S)
    if audio.ndim == 2:
        audio = np.concatenate([audio, np.zeros((tail_n, audio.shape[1]), dtype=np.float32)])
    else:
        audio = np.concatenate([audio, np.zeros(tail_n, dtype=np.float32)])
    sd.play(audio.astype(np.float32), SAMPLE_RATE, latency="low")
    if wait:
        sd.wait()


class MicRing:
    """Rolling microphone ring buffer and cached ambient telemetry."""

    def __init__(self):
        self.ring = np.zeros(int(MAX_RECORD_SECONDS * SAMPLE_RATE), dtype=np.float32)
        self.pos = 0
        self.filled = 0
        self.stream = None
        self.mute_until = 0.0
        self.lock = threading.Lock()
        self._status_cache = None
        self._status_ts = 0.0
        self._mode = "idle"
        self._locked_key = None

    def close(self):
        stream = self.stream
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
            self.stream = None

    def _on_audio(self, indata, frames, _time, _status):
        if time.monotonic() < self.mute_until:
            return
        src = indata[:, 0]
        if frames >= len(self.ring):
            src = src[-len(self.ring):]
            frames = len(self.ring)
        with self.lock:
            pos = self.pos
            first = min(frames, len(self.ring) - pos)
            self.ring[pos:pos + first] = src[:first]
            if first < frames:
                self.ring[:frames - first] = src[first:]
            self.pos = (pos + frames) % len(self.ring)
            self.filled = min(len(self.ring), self.filled + frames)

    def ensure_stream(self):
        stream = self.stream
        if stream is not None and stream.active:
            return True
        self.close()
        with self.lock:
            self.pos = 0
            self.filled = 0
        self._mode = "idle"
        self._locked_key = None
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=self._on_audio)
            stream.start()
            self.stream = stream
            return True
        except sd.PortAudioError:
            return False

    def mute_playback(self):
        self.mute_until = float("inf")

    def unmute_playback(self):
        self.mute_until = time.monotonic() + PLAYBACK_MUTE_TAIL_S

    def _slice_tail(self, seconds):
        with self.lock:
            filled = self.filled
            n = min(int(seconds * SAMPLE_RATE), filled)
            if n <= 0:
                return None
            pos = self.pos
            if pos >= n:
                return self.ring[pos - n:pos].copy()
            head = n - pos
            return np.concatenate((self.ring[len(self.ring) - head:], self.ring[:pos]))

    def recent(self, seconds):
        if not self.ensure_stream():
            return None
        min_samples = max(1, int(min(seconds, DAEMON_MIN_MIC_S) * SAMPLE_RATE))
        deadline = time.monotonic() + DAEMON_COLD_WAIT_S
        while self.filled < min_samples and time.monotonic() < deadline:
            if self.stream is None or not self.stream.active:
                return None
            time.sleep(0.025)
        if self.filled < min_samples:
            return None
        return self._slice_tail(seconds)

    def _analyze_key_state(self, key_buf):
        chroma = compute_chroma(key_buf, SAMPLE_RATE)
        if chroma.sum() == 0:
            self._mode = "silence"
            self._locked_key = None
            return ("silence", None, 0.0)

        candidates = key_signature_candidates(chroma)
        top_pc, top_conf = candidates[0]
        if self._mode == "music":
            self._mode = "music" if top_conf >= MUSIC_EXIT_THRESHOLD else "talking"
        else:
            self._mode = "music" if top_conf >= MUSIC_CONFIDENCE_THRESHOLD else "talking"

        if self._mode == "music":
            if self._locked_key is None:
                self._locked_key = top_pc
                return ("music", self._locked_key, top_conf)
            locked_conf = next((c for pc, c in candidates if pc == self._locked_key), top_conf)
            if top_conf >= locked_conf + KEY_SWITCH_MARGIN:
                self._locked_key = top_pc
                return ("music", self._locked_key, top_conf)
            return ("music", self._locked_key, locked_conf)

        self._locked_key = None
        return (self._mode, top_pc, top_conf)

    def resolve_for_args(self, args, gain_override=None, timbre_override=None):
        if args.avoid_speech is not None:
            return resolve_mic(args, self.recent, gain_override=gain_override, timbre_override=timbre_override)

        for flag, seconds in (("--auto-key", args.auto_key), ("--mic", args.mic)):
            if seconds is None:
                continue
            short_buf = self.recent(seconds)
            if short_buf is None:
                root_hz, gain, timbre, desc = (
                    DEFAULT_ROOT_HZ,
                    DEFAULT_GAIN,
                    DEFAULT_TIMBRE,
                    f"default (no mic) ({flag} {seconds:.1f}s)",
                )
                break
            key_buf = self._slice_tail(max(seconds, KEY_WINDOW_S))
            mode, pc, conf = self._analyze_key_state(key_buf if key_buf is not None else short_buf)
            level_db, gain = ambient_level(short_buf)
            if pc is None:
                root_hz, timbre, what = DEFAULT_ROOT_HZ, DEFAULT_TIMBRE, "silence"
            elif flag == "--auto-key" or mode == "music":
                root_hz = key_root_hz(pc)
                timbre = "sine"
                label = "detected key=" if flag == "--auto-key" else "music: "
                what = f"{label}{KEY_SIGNATURES[pc]} (confidence {conf:.2f})"
            else:
                root_hz = detect_quiet_freq(short_buf, SAMPLE_RATE)
                timbre = "marimba"
                what = f"talking: no clear key signature (confidence {conf:.2f}), using quietest region"
            desc = f"{what}, ambient {level_db:.0f} dBFS ({flag} {seconds:.1f}s)"
            break
        else:
            root_hz, desc = (args.root, "--root") if args.root is not None else (DEFAULT_ROOT_HZ, "default")
            gain, timbre = DEFAULT_GAIN, DEFAULT_TIMBRE

        if gain_override is not None:
            gain = max(0.0, min(1.0, float(gain_override)))
            desc = f"{desc}, override"
        if timbre_override in TIMBRES:
            timbre = timbre_override

        root_hz = fold_root(root_hz)
        print(f"[beep] root={root_hz:.2f} Hz gain={gain:.2f} tone={timbre} ({desc})", file=sys.stderr)
        return root_hz, gain, timbre

    def snapshot_status(self, seconds=2.0):
        if self.stream is None or not self.stream.active:
            return {
                "active": False,
                "ambient_db": None,
                "key": None,
                "key_confidence": None,
                "quiet_hz": None,
                "root_hz": None,
                "gain": DEFAULT_GAIN,
                "mode": "idle",
            }
        now = time.monotonic()
        if self._status_cache is not None and now - self._status_ts < STATUS_CACHE_TTL_S:
            return dict(self._status_cache)

        short_buf = self._slice_tail(seconds)
        if short_buf is None or len(short_buf) == 0:
            return {
                "active": True,
                "ambient_db": None,
                "key": None,
                "key_confidence": None,
                "quiet_hz": None,
                "root_hz": DEFAULT_ROOT_HZ,
                "gain": DEFAULT_GAIN,
                "mode": "silence",
            }

        level_db, gain = ambient_level(short_buf)
        quiet_hz = detect_quiet_freq(short_buf, SAMPLE_RATE, quiet=True)
        key_buf = self._slice_tail(max(seconds, KEY_WINDOW_S))
        mode, pc, conf = self._analyze_key_state(key_buf if key_buf is not None else short_buf)
        if pc is None:
            key_str, key_conf, raw_root = None, 0.0, DEFAULT_ROOT_HZ
        else:
            key_str = KEY_SIGNATURES[pc]
            key_conf = round(float(conf), 3)
            raw_root = key_root_hz(pc) if mode == "music" else quiet_hz

        snapshot = {
            "active": True,
            "ambient_db": round(float(level_db), 1),
            "key": key_str,
            "key_confidence": key_conf,
            "quiet_hz": round(float(quiet_hz), 1),
            "root_hz": round(float(fold_root(raw_root)), 2),
            "gain": round(float(gain), 3),
            "mode": mode,
        }
        self._status_cache = snapshot
        self._status_ts = now
        return dict(snapshot)
