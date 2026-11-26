import sys
import unittest
from unittest import mock

import typing_idle
from typing_idle import required_pause, wait_for_idle


class WaitForIdleTest(unittest.TestCase):
    def run_wait(self, idle, cancelled=lambda: False, since=0.0):
        clock = [0.0]

        def sleep(seconds):
            clock[0] += seconds

        with mock.patch.object(typing_idle, "idle_seconds", side_effect=lambda: idle(clock[0])), \
                mock.patch("typing_idle.time.monotonic", side_effect=lambda: clock[0]), \
                mock.patch("typing_idle.time.sleep", side_effect=sleep):
            return wait_for_idle(cancelled, since=since), clock[0]

    def test_requires_more_than_2_5_seconds(self):
        ok, now = self.run_wait(lambda t: 2.4 + t)
        self.assertTrue(ok)
        self.assertGreater(2.4 + now, 2.5)
        self.assertLess(now, 0.2)

    def test_new_input_restarts_wait(self):
        ok, now = self.run_wait(lambda t: t if t < 1.0 else t - 1.0)
        self.assertTrue(ok)
        self.assertGreater(now, 3.5)

    def test_pause_shrinks_with_request_age_and_has_a_floor(self):
        for age, pause in ((0, 2.5), (4.99, 2.5), (5, 1.25), (10, 0.625),
                           (20, 0.15625), (25, 0.078125), (30, 0.05), (600, 0.05)):
            with self.subTest(age=age):
                self.assertEqual(required_pause(age), pause)

    def test_continuous_input_eventually_plays_in_a_small_gap(self):
        ok, now = self.run_wait(lambda t: 0.06)
        self.assertTrue(ok)
        self.assertGreaterEqual(now, 30.0)
        self.assertLess(now, 30.2)

    def test_held_key_plays_after_max_wait(self):
        ok, now = self.run_wait(lambda t: 0.033)
        self.assertTrue(ok)
        self.assertGreaterEqual(now, typing_idle.MAX_WAIT_S)
        self.assertLess(now, typing_idle.MAX_WAIT_S + 0.1)

    def test_queue_age_shortens_wait(self):
        ok, now = self.run_wait(lambda t: 0.0, since=-typing_idle.MAX_WAIT_S)
        self.assertTrue(ok)
        self.assertEqual(now, 0.0)

    def test_cancel_stops_pending_sound(self):
        calls = []
        ok, _ = self.run_wait(lambda t: 0.0, cancelled=lambda: calls.append(1) or len(calls) > 3)
        self.assertFalse(ok)

    def test_unknown_idle_time_does_not_wait(self):
        with mock.patch.object(typing_idle, "idle_seconds", return_value=None):
            self.assertTrue(wait_for_idle())

    @unittest.skipUnless(sys.platform == "win32", "Windows input timing")
    def test_windows_reports_idle_time(self):
        self.assertGreaterEqual(typing_idle._windows_idle_seconds()(), 0.0)
