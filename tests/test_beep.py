import contextlib
import http.client
import http.server
import io
import json
import pathlib
import re
import sys
import threading
import time
import unittest
import warnings
from unittest import mock

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import audio
import beep
import daemon
import spatial


class BeepCliTest(unittest.TestCase):
    def test_parse_note_name_and_root(self):
        self.assertAlmostEqual(beep.parse_note_name("A4"), 440.0)
        self.assertAlmostEqual(beep.parse_note_name("C5"), 523.2511306, places=4)
        self.assertAlmostEqual(beep.parse_root("Bb3"), 233.08188, places=4)
        self.assertEqual(beep.parse_root("500"), 500.0)

    def test_parse_positive_float_validation(self):
        parser = beep.build_arg_parser()
        for bad in ("0", "-1", "31", "nan", "inf"):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    parser.parse_args(["--mic", bad, "--melody", "done"])
            self.assertEqual(ctx.exception.code, 1)

    def test_cli_mutual_exclusion(self):
        for argv in (
            ["--listen", "--sync", "--melody", "done"],
            ["--melody", "done", "--", "0,0.5,50"],
            ["--melody", "nonexistent"],
            ["--ask", "Approve?", "--listen"],
            ["--ask", "Approve?", "--sync"],
            ["--ask", "Approve?", "--", "0,0.5,50"],
            ["--ask", "   "],
            [],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    beep.main(argv)
            self.assertEqual(ctx.exception.code, 1)

    def test_melodies_parse_cleanly(self):
        for name, raw in beep.MELODIES.items():
            notes = [beep.parse_note(m) for m in raw]
            self.assertGreater(len(notes), 0, name)


class AudioDspTest(unittest.TestCase):
    def test_fold_root(self):
        self.assertAlmostEqual(audio.fold_root(110.0), 880.0)
        self.assertAlmostEqual(audio.fold_root(900.0), 900.0)
        self.assertAlmostEqual(audio.fold_root(1800.0), 900.0)
        self.assertAlmostEqual(audio.fold_root(450.0), 900.0)
        self.assertAlmostEqual(audio.fold_root(400.0), 800.0)

    def test_compute_chroma_unbiased_and_sine(self):
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(audio.SAMPLE_RATE * 2).astype(np.float32)
        noise_chroma = audio.compute_chroma(noise, audio.SAMPLE_RATE)
        self.assertLess(noise_chroma.max() / noise_chroma.min(), 1.35)

        t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
        sine = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sine_chroma = audio.compute_chroma(sine, audio.SAMPLE_RATE)
        self.assertEqual(int(np.argmax(sine_chroma)), 9)

    def test_key_signature_uniform_chroma_no_nan(self):
        uniform = np.ones(12) / 12.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidates = audio.key_signature_candidates(uniform)
        self.assertEqual(len(caught), 0)
        self.assertTrue(all(score == 0.0 for _, score in candidates))

    def test_detect_quiet_freq_in_band_and_avoids_tone(self):
        t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 500.0 * t) + 0.01 * np.sin(2 * np.pi * 700.0 * t)).astype(np.float32)
        freq = audio.detect_quiet_freq(tone, audio.SAMPLE_RATE)
        self.assertGreater(freq, audio.MAX_ROOT_HZ / 2)
        self.assertLessEqual(freq, audio.MAX_ROOT_HZ)
        self.assertGreater(abs(freq - 500.0), 50.0)

    def test_analyze_mic_short_buffer(self):
        short = np.full(441, 0.05, dtype=np.float32)
        root_hz, gain, timbre, _ = audio.analyze_mic("--mic", short)
        self.assertTrue(0 < root_hz <= 20000)
        self.assertTrue(0.25 <= gain <= 1.0)
        self.assertIn(timbre, audio.TIMBRES)

    def test_render_phrase_single_note_centered_and_compact(self):
        out = audio.render_phrase(440.0, [(0.0, 0.9, 99.0)])
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out[:, 0], out[:, 1])
        ring = int(audio.SAMPLE_RATE * audio.TIMBRES["sine"][2])
        lead = int(audio.SAMPLE_RATE * audio.LEAD_S)
        self.assertEqual(len(out), lead + ring)

        marimba_out = audio.render_phrase(440.0, [(0.0, 0.9, 99.0)], timbre="marimba")
        self.assertIsNotNone(marimba_out)
        self.assertLess(len(marimba_out), len(out))

    def test_play_notes_constant_power_pan_and_mono_fallback(self):
        played = []
        notes = [(0.0, 0.9, 99.0)]
        with mock.patch.object(audio, "supports_stereo_output", return_value=True), \
                mock.patch.object(audio.sd, "play", side_effect=lambda data, sr, latency=None: played.append(data)):
            for pan in (-1.0, 0.0, 1.0):
                audio.play_notes(440.0, 0.25, notes, pan=pan, muffle=0.0, ticks=0, wait=False)
        energies = [float(np.sum(buf ** 2)) for buf in played]
        self.assertAlmostEqual(energies[0], energies[1], delta=energies[1] * 0.02)
        self.assertAlmostEqual(energies[2], energies[1], delta=energies[1] * 0.02)
        self.assertAlmostEqual(float(np.max(np.abs(played[0][:, 1]))), 0.0, places=5)
        self.assertAlmostEqual(float(np.max(np.abs(played[2][:, 0]))), 0.0, places=5)

        mono_played = []
        with mock.patch.object(audio, "supports_stereo_output", return_value=False), \
                mock.patch.object(audio.sd, "play", side_effect=lambda data, sr, latency=None: mono_played.append(data)):
            audio.play_notes(440.0, 0.25, notes, pan=-1.0, muffle=0.0, ticks=0, wait=False)
        self.assertEqual(mono_played[0].ndim, 1)
        self.assertAlmostEqual(float(np.max(np.abs(mono_played[0]))), 0.25, places=5)

    def test_play_notes_ticks_peak_and_length(self):
        played = []
        notes = [(0.0, 0.9, 99.0)]
        with mock.patch.object(audio, "supports_stereo_output", return_value=True), \
                mock.patch.object(audio.sd, "play", side_effect=lambda data, sr, latency=None: played.append(data)):
            audio.play_notes(440.0, 0.5, notes, ticks=0, wait=False)
            audio.play_notes(440.0, 0.5, notes, ticks=2, wait=False)
        tick_n = int(audio.SAMPLE_RATE * audio.TICK_MS / 1000)
        gap_n = len(audio.silence(audio.CLICK_GAP_MS))
        lead_n = len(audio.silence(audio.CLICK_LEAD_MS))
        self.assertEqual(len(played[1]) - len(played[0]), 2 * (tick_n + gap_n) + lead_n)
        np.testing.assert_allclose(played[1][:, 0], played[1][:, 1])
        self.assertAlmostEqual(float(np.max(np.abs(played[1][:tick_n]))), 0.5 * audio.TICK_VOL, places=4)
        self.assertEqual(float(np.max(np.abs(played[1][tick_n:tick_n + gap_n]))), 0.0)

    def test_mic_ring_resolve_for_args_modes(self):
        parser = beep.build_arg_parser()
        sr = audio.SAMPLE_RATE
        rng = np.random.default_rng(3)
        ring = audio.MicRing()
        ring._on_audio((0.05 * rng.standard_normal((sr * 20, 1))).astype(np.float32), sr * 20, None, None)
        short = ring._slice_tail(1.0)
        quiet_root = audio.fold_root(audio.detect_quiet_freq(short, sr))
        _, ambient_gain = audio.ambient_level(short)
        analyzed = []

        def resolve(argv, state, **kw):
            def fake_state(buf):
                analyzed.append(len(buf))
                return state
            with mock.patch.object(ring, "ensure_stream", return_value=True), \
                    mock.patch.object(ring, "_analyze_key_state", side_effect=fake_state), \
                    contextlib.redirect_stderr(io.StringIO()):
                return ring.resolve_for_args(parser.parse_args(argv + ["--melody", "done"]), **kw)

        self.assertEqual(resolve(["--mic", "1"], ("talking", 0, 0.1)), (quiet_root, ambient_gain, "marimba"))
        self.assertEqual(analyzed[-1], int(audio.KEY_WINDOW_S * sr))
        self.assertEqual(resolve(["--mic", "1"], ("music", 9, 0.8)), (880.0, ambient_gain, "sine"))
        self.assertEqual(
            resolve(["--auto-key", "1"], ("talking", 7, 0.1)),
            (audio.fold_root(audio.key_root_hz(7)), ambient_gain, "sine"),
        )
        self.assertEqual(resolve(["--mic", "1"], ("silence", None, 0.0)), (880.0, ambient_gain, "sine"))
        self.assertEqual(
            resolve(["--mic", "1"], ("music", 9, 0.8), gain_override=0.6, timbre_override="marimba"),
            (880.0, 0.6, "marimba"),
        )

        root_hz, gain, timbre, _ = audio.analyze_mic("--avoid-speech", ring._slice_tail(1.5))
        self.assertEqual(resolve(["--avoid-speech", "1.5"], None), (audio.fold_root(root_hz), gain, timbre))

        with mock.patch.object(ring, "ensure_stream", return_value=False), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(
                ring.resolve_for_args(parser.parse_args(["--mic", "1", "--melody", "done"])),
                (880.0, audio.DEFAULT_GAIN, audio.DEFAULT_TIMBRE),
            )

    def test_framed_chroma_song_progression_and_noise_rejection(self):
        sr = audio.SAMPLE_RATE
        rng = np.random.default_rng(7)

        def chord(freqs, dur):
            t = np.arange(int(sr * dur)) / sr
            return sum(np.sin(2 * np.pi * f * t) + 0.5 * np.sin(4 * np.pi * f * t) for f in freqs) / len(freqs)

        song = np.concatenate([
            chord([261.63, 329.63, 392.00], 3.0),
            chord([196.00, 246.94, 293.66], 3.0),
            chord([220.00, 261.63, 329.63], 3.0),
            chord([174.61, 220.00, 261.63], 3.0),
        ]) + 0.5 * rng.standard_normal(sr * 12)
        for beat in range(24):
            i = int(beat * 0.5 * sr)
            song[i:i + 1500] += 2.5 * rng.standard_normal(1500) * np.exp(-np.arange(1500) / 350)

        pc, conf = audio.key_signature_candidates(audio.compute_chroma(song.astype(np.float32), sr))[0]
        self.assertEqual(audio.KEY_SIGNATURES[pc], "C / Am")
        self.assertGreaterEqual(conf, 0.55)

        noise = rng.standard_normal(sr * 12).astype(np.float32)
        _, noise_conf = audio.key_signature_candidates(audio.compute_chroma(noise, sr))[0]
        self.assertLess(noise_conf, 0.15)

    def test_mic_ring_hysteresis_mode_and_key(self):
        ring = audio.MicRing()
        dummy = np.ones(1024, dtype=np.float32)

        seq = [
            [(0, 0.70), (7, 0.65)],
            [(7, 0.28), (0, 0.25)],
            [(7, 0.73), (0, 0.70)],
            [(7, 0.80), (0, 0.65)],
            [(7, 0.18), (0, 0.12)],
        ]
        with mock.patch.object(audio, "compute_chroma", return_value=np.ones(12) / 12), \
                mock.patch.object(audio, "key_signature_candidates", side_effect=seq):
            self.assertEqual(ring._analyze_key_state(dummy), ("music", 0, 0.70))
            self.assertEqual(ring._analyze_key_state(dummy), ("music", 0, 0.25))
            self.assertEqual(ring._analyze_key_state(dummy), ("music", 0, 0.70))
            self.assertEqual(ring._analyze_key_state(dummy), ("music", 7, 0.80))
            self.assertEqual(ring._analyze_key_state(dummy)[0], "talking")


class DaemonHttpTest(unittest.TestCase):
    @contextlib.contextmanager
    def running_daemon(self, play_errors=()):
        played = []
        play_errors = list(play_errors)
        self.spatial_calls = []
        played_event = threading.Event()
        ready = threading.Event()
        ports = []
        parser = beep.build_arg_parser()
        args = parser.parse_args(["--listen"])

        class FakeStream:
            active = True

            def start(self):
                pass

            def close(self):
                pass

        def fake_play(data, sr, latency=None):
            if play_errors:
                raise play_errors.pop(0)
            played.append(data)
            played_event.set()

        def fake_spatial(req):
            placement = (req.pan or 0.0, req.muffle or 0.0, req.ticks or 0)
            self.spatial_calls.append((placement, list(req.notes)))
            return placement

        with mock.patch.object(daemon.sd, "InputStream", return_value=FakeStream(), create=True), \
                mock.patch.object(audio.sd, "InputStream", return_value=FakeStream()), \
                mock.patch.object(audio.sd, "play", side_effect=fake_play), \
                mock.patch.object(audio.sd, "wait"), \
                mock.patch.object(audio, "supports_stereo_output", return_value=True):
            t = threading.Thread(
                target=daemon.serve,
                args=(args, parser.parse_args, fake_spatial, beep.MELODIES),
                kwargs={"port": 0, "ready_event": ready, "bound_port_out": ports},
                daemon=True,
            )
            t.start()
            self.daemon_thread = t
            self.assertTrue(ready.wait(timeout=2.0))
            port = ports[0]
            status, html = self._http("GET", port, "/")
            self.assertEqual(status, 200)
            m = re.search(r'const TOKEN = "([0-9a-f]+)";', html.decode("utf-8"))
            self.assertIsNotNone(m)
            token = m.group(1)
            try:
                yield port, token, played, played_event
            finally:
                with contextlib.suppress(OSError):
                    self._http("POST", port, "/api/stop", {}, {"X-Conso-Token": token})
                t.join(timeout=3.0)

    def _http(self, method, port, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
        hdrs = dict(headers or {})
        raw = None
        if body is not None:
            raw = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        conn.request(method, path, body=raw, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        conn.close()
        return resp.status, data

    def test_dashboard_status_and_controls(self):
        with self.running_daemon() as (port, token, played, played_event):
            status, body = self._http("GET", port, "/api/status")
            self.assertEqual(status, 200)
            data = json.loads(body)
            self.assertIn("mic", data)
            self.assertIn("controls", data)
            self.assertEqual(data["permissions"], [])
            self.assertEqual(data["melodies"], list(beep.MELODIES.keys()))

            self.assertTrue(
                beep.send_to_daemon(["--root", "440", "--pan=0", "--muffle=0", "--ticks=0", "--", "0,0.8,50"], port=port)
            )
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(len(played), 1)

            played_event.clear()
            st, _ = self._http("POST", port, "/api/controls", {"muted": True, "keep_alive": True}, {"X-Conso-Token": token})
            self.assertEqual(st, 200)
            self.assertTrue(
                beep.send_to_daemon(["--root", "440", "--pan=0", "--muffle=0", "--ticks=0", "--", "0,0.8,50"], port=port)
            )
            self.assertFalse(played_event.wait(timeout=0.15))
            self.assertEqual(len(played), 1)

            st, ctrl_raw = self._http(
                "POST", port, "/api/controls", {"muted": False, "gain_override": 0.75}, {"X-Conso-Token": token}
            )
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(ctrl_raw)["controls"]["gain_override"], 0.75)
            self.assertTrue(
                beep.send_to_daemon(["--root", "440", "--pan=0", "--muffle=0", "--ticks=0", "--", "0,0.8,50"], port=port)
            )
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(len(played), 2)
            self.assertAlmostEqual(float(np.max(np.abs(played[1]))), 0.75, places=4)

            played_event.clear()
            st, play_res = self._http("POST", port, "/api/play", {"melody": "question", "use_mic": False, "timbre": "marimba"})
            self.assertEqual(st, 200)
            self.assertTrue(json.loads(play_res)["ok"])
            self.assertTrue(played_event.wait(timeout=2.0))
            played_event.clear()

            st, _ = self._http("POST", port, "/api/play", {"notes": "0,0.8,50 4,0.8,50", "use_mic": False})
            self.assertEqual(st, 200)
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(len(played), 4)

            st, _ = self._http("POST", port, "/api/play", {"notes": "badnote", "use_mic": False})
            self.assertEqual(st, 400)

    def test_multi_permission_batch_decide_and_ask(self):
        with self.running_daemon() as (port, token, _played, _played_event):
            results = {}

            def ask_req(key, title, source):
                results[key] = beep.http_post_json(
                    "/api/permissions",
                    {"title": title, "detail": "echo hi", "source": source, "timeout": 5.0},
                    port=port,
                    timeout=5.0,
                )

            t1 = threading.Thread(target=ask_req, args=("a", "Run echo", "claude"))
            t2 = threading.Thread(target=ask_req, args=("b", "Edit foo.py", "opencode"))
            t1.start()
            t2.start()

            ids = []
            for _ in range(40):
                _, raw = self._http("GET", port, "/api/status")
                perms = json.loads(raw)["permissions"]
                if len(perms) == 2:
                    ids = [p["id"] for p in perms]
                    break
                time.sleep(0.025)
            self.assertEqual(len(ids), 2)

            st, decide_raw = self._http(
                "POST",
                port,
                "/api/permissions/decide",
                {"ids": ids, "decision": "allow", "reason": "batch"},
                {"X-Conso-Token": token},
            )
            self.assertEqual(st, 200)
            self.assertEqual(set(json.loads(decide_raw)["decided"]), set(ids))

            t1.join(timeout=2.0)
            t2.join(timeout=2.0)
            self.assertEqual(results["a"], (200, {"decision": "allow", "reason": "batch"}))
            self.assertEqual(results["b"], (200, {"decision": "allow", "reason": "batch"}))

            ask_args = beep.build_arg_parser().parse_args(
                ["--ask", "Deploy", "--detail", "git push", "--source", "cli", "--pan", "0", "--muffle", "0", "--ticks", "0"]
            )
            out_buf = io.StringIO()

            def run_cli_ask():
                with contextlib.redirect_stdout(out_buf):
                    beep.run_ask(ask_args, port=port)

            t3 = threading.Thread(target=run_cli_ask)
            t3.start()
            ask_id = None
            for _ in range(40):
                _, raw = self._http("GET", port, "/api/status")
                perms = json.loads(raw)["permissions"]
                if len(perms) == 1:
                    ask_id = perms[0]["id"]
                    break
                time.sleep(0.025)
            self.assertIsNotNone(ask_id)
            self._http(
                "POST",
                port,
                "/api/permissions/decide",
                {"ids": [ask_id], "decision": "deny", "reason": "nope"},
                {"X-Conso-Token": token},
            )
            t3.join(timeout=2.0)
            self.assertEqual(json.loads(out_buf.getvalue()), {"decision": "deny", "reason": "nope"})

    def test_idle_exit_unless_keep_alive_or_pending_permission(self):
        with mock.patch.object(daemon, "DAEMON_IDLE_S", 1.0), self.running_daemon() as (port, token, _played, _ev):
            auth = {"X-Conso-Token": token}
            self._http("POST", port, "/api/controls", {"keep_alive": True}, auth)
            time.sleep(1.6)
            self.assertTrue(self.daemon_thread.is_alive())

            self._http("POST", port, "/api/controls", {"keep_alive": False}, auth)
            ask = threading.Thread(
                target=beep.http_post_json,
                args=("/api/permissions", {"title": "Idle test", "timeout": 2.5}),
                kwargs={"port": port, "timeout": 5.0},
            )
            ask.start()
            for _ in range(40):
                _, raw = self._http("GET", port, "/api/status")
                if json.loads(raw)["permissions"]:
                    break
                time.sleep(0.025)
            time.sleep(1.6)
            self.assertTrue(self.daemon_thread.is_alive())

            ask.join(timeout=3.0)
            self.daemon_thread.join(timeout=3.0)
            self.assertFalse(self.daemon_thread.is_alive())

    def test_dashboard_play_spatial_reaches_playback(self):
        with self.running_daemon() as (port, _token, _played, played_event):
            st, _ = self._http(
                "POST", port, "/api/play",
                {"notes": "0,0.8,50", "use_mic": False, "pan": -0.5, "muffle": 0.7, "ticks": 2},
            )
            self.assertEqual(st, 200)
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(self.spatial_calls[-1][0], (-0.5, 0.7, 2))

    def test_permission_request_plays_permission_melody(self):
        expected = [beep.parse_note(m) for m in beep.MELODIES["permission"]]
        with self.running_daemon() as (port, _token, played, played_event):
            st, _ = beep.http_post_json(
                "/api/permissions",
                {"title": "Melody test", "timeout": 0.3, "root_flags": ["--root", "440"], "pan": 0.4, "muffle": 0.2, "ticks": 1},
                port=port,
                timeout=2.0,
            )
            self.assertEqual(st, 200)
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(len(played), 1)
            self.assertEqual(self.spatial_calls[-1], ((0.4, 0.2, 1), expected))

    def test_alert_shows_plays_and_dismisses(self):
        expected = [beep.parse_note(m) for m in beep.MELODIES["alert"]]
        with self.running_daemon() as (port, token, _played, played_event):
            alert_args = beep.build_arg_parser().parse_args(
                ["--alert", "Which DB?", "--detail", "postgres or sqlite", "--pan", "0", "--muffle", "0", "--ticks", "0"]
            )
            self.assertEqual(beep.run_alert(alert_args, port=port), 0)
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(self.spatial_calls[-1][1], expected)

            _, raw = self._http("GET", port, "/api/status")
            data = json.loads(raw)
            self.assertEqual(data["permissions"], [])
            self.assertEqual([(a["title"], a["detail"]) for a in data["alerts"]], [("Which DB?", "postgres or sqlite")])
            alert_id = data["alerts"][0]["id"]

            st, _ = self._http("POST", port, "/api/alerts/dismiss", {"ids": [alert_id]})
            self.assertEqual(st, 403)
            st, raw = self._http("POST", port, "/api/alerts/dismiss", {"ids": [alert_id]}, {"X-Conso-Token": token})
            self.assertEqual(st, 200)
            self.assertEqual(json.loads(raw)["dismissed"], [alert_id])
            _, raw = self._http("GET", port, "/api/status")
            self.assertEqual(json.loads(raw)["alerts"], [])

    def test_port_audio_error_reinitializes_and_retries(self):
        with mock.patch.object(daemon.sd, "_terminate") as term, mock.patch.object(daemon.sd, "_initialize") as init, \
                self.running_daemon(play_errors=[daemon.sd.PortAudioError("device lost")]) as (port, _token, played, played_event):
            self.assertTrue(beep.send_to_daemon(["--root", "440", "--", "0,0.8,50"], port=port))
            self.assertTrue(played_event.wait(timeout=2.0))
            self.assertEqual(len(played), 1)
            term.assert_called_once()
            init.assert_called_once()

    def test_bind_failure_plays_in_process(self):
        parser = beep.build_arg_parser()
        args = parser.parse_args(["--listen", "--root", "440", "--", "0,0.8,50"])
        holder = daemon._DaemonHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        try:
            with mock.patch.object(audio, "play_notes") as play:
                rc = daemon.serve(args, parser.parse_args, lambda req: (0.25, 0.0, 0), beep.MELODIES, port=holder.server_port)
        finally:
            holder.server_close()
        self.assertEqual(rc, 0)
        play.assert_called_once()
        self.assertEqual(play.call_args.args[2], args.notes)
        self.assertEqual(play.call_args.args[3:6], (0.25, 0.0, 0))

    def test_permission_timeout_and_security(self):
        with self.running_daemon() as (port, token, _played, _played_event):
            st, body = beep.http_post_json(
                "/api/permissions",
                {"title": "Timeout test", "timeout": 0.15},
                port=port,
                timeout=2.0,
            )
            self.assertEqual(st, 200)
            self.assertEqual(body, {"decision": "timeout", "reason": ""})

            st, _ = self._http("GET", port, "/api/status", headers={"Host": "evil.example:80"})
            self.assertEqual(st, 403)

            st, _ = self._http("GET", port, "/api/status", headers={"Origin": "http://evil.example"})
            self.assertEqual(st, 403)

            for protected in ("/api/permissions/decide", "/api/controls", "/api/stop"):
                st, _ = self._http("POST", port, protected, {})
                self.assertEqual(st, 403)
                st, _ = self._http("POST", port, protected, {}, {"X-Conso-Token": "wrong"})
                self.assertEqual(st, 403)

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2.0)
            conn.putrequest("POST", "/api/play")
            conn.putheader("Content-Type", "application/json")
            conn.putheader("Content-Length", str(daemon.DAEMON_MAX_MSG_BYTES + 100))
            conn.endheaders()
            resp = conn.getresponse()
            resp.read()
            conn.close()
            self.assertEqual(resp.status, 413)


class DispatchPlayTest(unittest.TestCase):
    def test_fallback_spawns_listener_when_daemon_unavailable(self):
        args = beep.build_arg_parser().parse_args(["--root", "440", "--", "0,0.8,50"])
        placement = (0.5, 0.0, 0)
        expected = ["--listen", *beep.play_argv(args, placement, args.notes)]
        for status in (None, 500):
            with mock.patch.object(beep, "http_post_json", return_value=(status, None)), \
                    mock.patch.object(beep, "spawn_detached") as spawn:
                beep.dispatch_play(args, placement, args.notes)
            spawn.assert_called_once_with(expected)

        with mock.patch.object(beep, "http_post_json", return_value=(200, {"ok": True})), \
                mock.patch.object(beep, "spawn_detached") as spawn:
            beep.dispatch_play(args, placement, args.notes)
        spawn.assert_not_called()


class SpatialTest(unittest.TestCase):
    def test_compute_monitor_pan_layouts(self):
        self.assertEqual(spatial.compute_monitor_pan([(1, 0, 0, 1920, 1080)], 1), (0.0, 1, 1))

        side_by_side = [(1, -1920, 0, 0, 1080), (2, 0, 0, 1920, 1080), (3, 1920, 0, 3840, 1080)]
        self.assertEqual(spatial.compute_monitor_pan(side_by_side, 1), (-0.6, 1, 3))
        self.assertEqual(spatial.compute_monitor_pan(side_by_side, 2), (0.0, 2, 3))
        self.assertEqual(spatial.compute_monitor_pan(side_by_side, 3), (0.6, 3, 3))

        stacked = [(1, 0, 0, 1920, 1080), (2, 0, 1080, 3440, 2520)]
        self.assertEqual(spatial.compute_monitor_pan(stacked, 1), (0.0, 1, 1))
        self.assertEqual(spatial.compute_monitor_pan(stacked, 2), (0.0, 1, 1))

    def test_pick_best_window(self):
        candidates = [(10, "README.md - foo - Visual Studio Code"), (20, "index.ts - bar - Visual Studio Code")]
        self.assertEqual(spatial.pick_best_window(candidates, "C:/work/foo"), 10)
        tie = [(10, "foo - Code"), (20, "foo - Code")]
        self.assertIsNone(spatial.pick_best_window(tie, "C:/work/foo"))

    def test_resolve_spatial_values(self):
        self.assertEqual(spatial.resolve_spatial_values(-0.6, 0, True, "monitor 1/3"), (-0.6, 0.0, 0, "monitor 1/3"))
        self.assertEqual(spatial.resolve_spatial_values(0.0, -2, False), (-1.0, 0.85, 2, "desktop -2"))
        self.assertEqual(spatial.resolve_spatial_values(0.3, None, False), (0.3, 0.70, 1, "other desktop"))


if __name__ == "__main__":
    unittest.main()
