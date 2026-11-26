"""Wait for a pause in Windows keyboard and mouse input before playback."""
import sys
import time

TYPING_IDLE_S = 2.5
RELAX_EVERY_S = 5.0
MIN_TYPING_IDLE_S = 0.05
MAX_WAIT_S = 30.0
POLL_S = 0.02


def required_pause(waited_s):
    """Halve the desired gap every five seconds, down to 50 milliseconds."""
    steps = max(0.0, waited_s) // RELAX_EVERY_S
    return max(MIN_TYPING_IDLE_S, TYPING_IDLE_S * 0.5 ** steps)


def _windows_idle_seconds():
    import ctypes
    from ctypes import wintypes as wt

    class LastInputInfo(ctypes.Structure):
        _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]

    user32 = ctypes.WinDLL("user32")
    kernel32 = ctypes.WinDLL("kernel32")
    user32.GetLastInputInfo.argtypes = [ctypes.POINTER(LastInputInfo)]
    user32.GetLastInputInfo.restype = wt.BOOL
    kernel32.GetTickCount.restype = wt.DWORD

    def idle_seconds():
        info = LastInputInfo(ctypes.sizeof(LastInputInfo), 0)
        if not user32.GetLastInputInfo(ctypes.byref(info)):
            return None
        return ((kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF) / 1000.0

    return idle_seconds


idle_seconds = _windows_idle_seconds() if sys.platform == "win32" else (lambda: None)


def wait_for_idle(cancelled=lambda: False, *, since=None):
    """Wait for a progressively shorter input pause, at most MAX_WAIT_S; False if cancelled."""
    started = time.monotonic() if since is None else since
    while not cancelled():
        waited = time.monotonic() - started
        idle = idle_seconds()
        if idle is None or waited >= MAX_WAIT_S or idle > required_pause(waited):
            return True
        time.sleep(POLL_S)
    return False
