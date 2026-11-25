import contextlib
import io
import socket
import threading
import unittest
import warnings
from unittest import mock

import numpy as np

import audio
import beep
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
                with self.assertRaises(SystemExit):
                    parser.parse_args(["--mic", bad, "--melody", "done"])

    def test_cli_mutual_exclusion(self):
        for argv in (
            ["--listen", "--sync", "--melody", "done"],
            ["--melody", "done", "--", "0,0.5,50"],
            [],
        ):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    beep.main(argv)

    def test_melodies_parse_cleanly(self):
        for name, raw in beep.MELODIES.items():
            notes = [beep.parse_note(m) for m in raw]
            self.assertGreater(len(notes), 0, name)


class AudioDspAndDaemonTest(unittest.TestCase):
    def test_fold_root(self):
        self.assertAlmostEqual(audio.fold_root(110.0), 440.0)
        self.assertAlmostEqual(audio.fold_root(1600.0), 800.0)
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

    def test_krumhansl_uniform_chroma_no_nan(self):
        uniform = np.ones(12) / 12.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidates = audio.krumhansl_key_candidates(uniform)
        self.assertEqual(len(caught), 0)
        self.assertTrue(all(score == 0.0 for _, _, score in candidates))

    def test_detect_quiet_freq_in_band_and_avoids_tone(self):
        t = np.arange(audio.SAMPLE_RATE) / audio.SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 500.0 * t) + 0.01 * np.sin(2 * np.pi * 700.0 * t)).astype(np.float32)
        freq = audio.detect_quiet_freq(tone, audio.SAMPLE_RATE)
        self.assertGreater(freq, 400.0)
        self.assertLessEqual(freq, 800.0)
        self.assertGreater(abs(freq - 500.0), 50.0)

    def test_analyze_mic_short_buffer(self):
        short = np.full(441, 0.05, dtype=np.float32)
        root_hz, gain, _ = audio.analyze_mic("--mic", short)
        self.assertTrue(0 < root_hz <= 20000)
        self.assertTrue(0.25 <= gain <= 1.0)

    def test_render_phrase_single_note_centered_and_compact(self):
        out = audio.render_phrase(440.0, [(0.0, 0.9, 99.0)])
        self.assertIsNotNone(out)
        np.testing.assert_allclose(out[:, 0], out[:, 1])
        ring = int(audio.SAMPLE_RATE * audio.NOTE_RING_S * audio.TONE_LENGTH)
        lead = int(audio.SAMPLE_RATE * audio.LEAD_S)
        self.assertEqual(len(out), lead + ring)

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

    def test_daemon_roundtrip_and_oversized_rejection(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        played = []
        done = threading.Event()
        parser = beep.build_arg_parser()
        args = parser.parse_args(["--listen"])

        class FakeStream:
            active = True

            def start(self):
                pass

            def close(self):
                pass

        def fake_play(data, sr, latency=None):
            played.append(data)
            done.set()

        with mock.patch.object(audio, "DAEMON_IDLE_S", 0.35), \
                mock.patch.object(audio.sd, "InputStream", return_value=FakeStream()), \
                mock.patch.object(audio.sd, "play", side_effect=fake_play), \
                mock.patch.object(audio.sd, "wait"), \
                mock.patch.object(audio, "supports_stereo_output", return_value=True):
            t = threading.Thread(
                target=audio.listen,
                args=(args, parser.parse_args, lambda req: (0.0, 0.0, 0)),
                kwargs={"port": port},
                daemon=True,
            )
            t.start()

            for _ in range(20):
                if beep.send_to_daemon(["--root", "440", "--pan=0", "--muffle=0", "--ticks=0", "--", "0,0.8,50"], port=port):
                    break
                threading.Event().wait(0.02)
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(played), 1)

            with socket.create_connection(("127.0.0.1", port), timeout=0.5) as big_conn:
                big_conn.sendall(b"x" * (audio.DAEMON_MAX_MSG_BYTES + 100))
                big_conn.shutdown(socket.SHUT_WR)
                self.assertEqual(big_conn.recv(16), b"")

            done.clear()
            self.assertTrue(
                beep.send_to_daemon(["--root", "440", "--pan=0", "--muffle=0", "--ticks=0", "--", "0,0.8,50"], port=port)
            )
            self.assertTrue(done.wait(timeout=2.0))
            self.assertEqual(len(played), 2)
            t.join(timeout=2.0)


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
