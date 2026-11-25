import os
import pathlib
import re
import sys
import threading

MONITOR_PAN_WIDTH = 0.6
MAX_TICKS = 3
DETECT_TIMEOUT_S = 1.0


def compute_monitor_pan(
    monitors: list[tuple[int, int, int, int, int]],
    target_hmon: int,
) -> tuple[float, int, int]:
    if not monitors:
        return (0.0, 1, 1)

    ordered = sorted(
        monitors,
        key=lambda m: (m[1], m[2]),
    )
    col_map: dict[int, int] = {}
    col_idx = 0
    col_left, col_right = ordered[0][1], ordered[0][3]

    for hmon, left, _top, right, _bottom in ordered:
        overlap = min(right, col_right) - max(left, col_left)
        min_width = min(max(1, right - left), max(1, col_right - col_left))
        if overlap < 0.5 * min_width:
            col_idx += 1
            col_left, col_right = left, right
        else:
            col_left = min(col_left, left)
            col_right = max(col_right, right)
        col_map[hmon] = col_idx

    num_cols = col_idx + 1
    if target_hmon not in col_map:
        return (0.0, 1, num_cols)

    target_col = col_map[target_hmon]
    if num_cols <= 1:
        return (0.0, 1, 1)

    pan = MONITOR_PAN_WIDTH * (-1.0 + 2.0 * target_col / (num_cols - 1))
    return (pan, target_col + 1, num_cols)


def pick_best_window(
    candidates: list[tuple[int, str]],
    cwd: str | pathlib.Path | None,
) -> int | None:
    if len(candidates) == 1:
        return candidates[0][0]
    if not candidates or not cwd:
        return None

    components: list[str] = []
    cur = pathlib.Path(os.path.abspath(str(cwd)))
    for _ in range(4):
        if cur.name:
            components.append(cur.name.lower())
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    best_hwnd = None
    best_score = 0

    for hwnd, title in candidates:
        segments = [
            re.sub(r"(?:\s*(?:\(Workspace\)|\[[^\]]+\]))+$", "", seg, flags=re.I).strip().lower()
            for seg in re.split(r" - | \u2014 ", title)
        ]
        title_lower = title.lower()
        win_score = 0
        for d, comp in enumerate(components):
            if comp in segments:
                win_score = max(win_score, 100 - d)
            elif len(comp) >= 2 and re.search(rf"(?<![a-z0-9_]){re.escape(comp)}(?![a-z0-9_])", title_lower):
                win_score = max(win_score, 50 - d)
        if win_score > best_score:
            best_score = win_score
            best_hwnd = hwnd
        elif win_score and win_score == best_score:
            best_hwnd = None

    return best_hwnd


def resolve_spatial_values(
    monitor_pan: float,
    desktop_delta: int | None,
    on_current_desktop: bool | None,
    mon_label: str = "monitor",
) -> tuple[float, float, int, str]:
    if on_current_desktop is False:
        if desktop_delta is not None and desktop_delta != 0:
            pan = -1.0 if desktop_delta < 0 else 1.0
            muffle = min(1.0, 0.70 + 0.15 * (abs(desktop_delta) - 1))
            return (pan, muffle, min(MAX_TICKS, abs(desktop_delta)), f"desktop {desktop_delta:+d}")
        return (monitor_pan, 0.70, 1, "other desktop")
    return (monitor_pan, 0.0, 0, mon_label)


if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes as wt
    import uuid
    import winreg

    _STOP_PROCS = frozenset({
        "explorer.exe",
        "services.exe",
        "svchost.exe",
        "wininit.exe",
        "winlogon.exe",
        "csrss.exe",
        "smss.exe",
        "sihost.exe",
        "system",
        "idle",
    })
    _VSCODE_EXES = frozenset({
        "code.exe",
        "code - insiders.exe",
        "cursor.exe",
        "windsurf.exe",
        "codium.exe",
    })
    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GW_OWNER = 4
    _GWL_EXSTYLE = -20
    _WS_EX_TOOLWINDOW = 0x00000080
    _MONITOR_DEFAULTTONEAREST = 2
    _TH32CS_SNAPPROCESS = 0x00000002
    _COINIT_APARTMENTTHREADED = 0x2
    _CLSCTX_ALL = 0x17
    _CLSID_VDA = "{AA509086-5CA9-4C25-8F95-589D3C07B48A}"
    _IID_VDA = "{A5CD92FF-29BE-454C-8D04-D82879FB3F1B}"
    _VD_REG_PATH = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\VirtualDesktops"
    _SESSION_INFO_PATH = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\SessionInfo"

    class _PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD),
            ("cntUsage", wt.DWORD),
            ("th32ProcessID", wt.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wt.DWORD),
            ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD),
            ("pcPriClassBase", wt.LONG),
            ("dwFlags", wt.DWORD),
            ("szExeFile", wt.WCHAR * 260),
        ]

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", wt.LONG),
            ("top", wt.LONG),
            ("right", wt.LONG),
            ("bottom", wt.LONG),
        ]

    class _MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", wt.DWORD),
            ("rcMonitor", _RECT),
            ("rcWork", _RECT),
            ("dwFlags", wt.DWORD),
        ]

    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wt.DWORD),
            ("Data2", wt.WORD),
            ("Data3", wt.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _ole32 = ctypes.WinDLL("ole32", use_last_error=True)

    _WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    _MONITORENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HMONITOR, wt.HDC, ctypes.POINTER(_RECT), wt.LPARAM)

    _kernel32.CreateToolhelp32Snapshot.argtypes = [wt.DWORD, wt.DWORD]
    _kernel32.CreateToolhelp32Snapshot.restype = wt.HANDLE
    _kernel32.Process32FirstW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _kernel32.Process32FirstW.restype = wt.BOOL
    _kernel32.Process32NextW.argtypes = [wt.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    _kernel32.Process32NextW.restype = wt.BOOL
    _kernel32.CloseHandle.argtypes = [wt.HANDLE]
    _kernel32.CloseHandle.restype = wt.BOOL
    _kernel32.GetConsoleWindow.argtypes = []
    _kernel32.GetConsoleWindow.restype = wt.HWND

    _user32.EnumWindows.argtypes = [_WNDENUMPROC, wt.LPARAM]
    _user32.EnumWindows.restype = wt.BOOL
    _user32.IsWindowVisible.argtypes = [wt.HWND]
    _user32.IsWindowVisible.restype = wt.BOOL
    _user32.GetWindow.argtypes = [wt.HWND, wt.UINT]
    _user32.GetWindow.restype = wt.HWND
    _user32.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
    _user32.GetWindowLongW.restype = wt.LONG
    _user32.GetClassNameW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    _user32.GetClassNameW.restype = ctypes.c_int
    _user32.GetWindowTextLengthW.argtypes = [wt.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    _user32.GetWindowThreadProcessId.restype = wt.DWORD
    _user32.EnumDisplayMonitors.argtypes = [wt.HDC, ctypes.POINTER(_RECT), _MONITORENUMPROC, wt.LPARAM]
    _user32.EnumDisplayMonitors.restype = wt.BOOL
    _user32.GetMonitorInfoW.argtypes = [wt.HMONITOR, ctypes.POINTER(_MONITORINFO)]
    _user32.GetMonitorInfoW.restype = wt.BOOL
    _user32.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
    _user32.MonitorFromWindow.restype = wt.HMONITOR

    _ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
    _ole32.CoInitializeEx.restype = ctypes.c_long
    _ole32.CoUninitialize.argtypes = []
    _ole32.CoUninitialize.restype = None
    _ole32.CoCreateInstance.argtypes = [
        ctypes.POINTER(_GUID),
        ctypes.c_void_p,
        wt.DWORD,
        ctypes.POINTER(_GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _ole32.CoCreateInstance.restype = ctypes.c_long

    def _snapshot_processes() -> dict[int, tuple[int, str]]:
        snap = _kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
        if not snap or snap == _INVALID_HANDLE_VALUE:
            return {}
        procs: dict[int, tuple[int, str]] = {}
        try:
            pe = _PROCESSENTRY32W()
            pe.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
            if _kernel32.Process32FirstW(snap, ctypes.byref(pe)):
                while True:
                    procs[pe.th32ProcessID] = (pe.th32ParentProcessID, pe.szExeFile)
                    if not _kernel32.Process32NextW(snap, ctypes.byref(pe)):
                        break
        finally:
            _kernel32.CloseHandle(snap)
        return procs

    def _ancestor_chain(start_pid: int, procs: dict[int, tuple[int, str]]) -> list[tuple[int, str]]:
        chain: list[tuple[int, str]] = []
        seen: set[int] = set()
        pid = start_pid
        while pid and pid in procs and pid not in seen:
            seen.add(pid)
            ppid, exe = procs[pid]
            if exe.lower() in _STOP_PROCS:
                break
            chain.append((pid, exe))
            pid = ppid
        return chain

    def _window_class(hwnd: int) -> str:
        buf = ctypes.create_unicode_buffer(256)
        n = _user32.GetClassNameW(hwnd, buf, 256)
        return buf.value[:n] if n > 0 else ""

    def _enum_app_windows() -> list[tuple[int, int, str, str]]:
        windows: list[tuple[int, int, str, str]] = []

        def cb(hwnd: int, _lparam: int) -> bool:
            if not _user32.IsWindowVisible(hwnd):
                return True
            if _user32.GetWindow(hwnd, _GW_OWNER):
                return True
            if _user32.GetWindowLongW(hwnd, _GWL_EXSTYLE) & _WS_EX_TOOLWINDOW:
                return True
            cls = _window_class(hwnd)
            if cls == "PseudoConsoleWindow":
                return True
            title_len = _user32.GetWindowTextLengthW(hwnd)
            if title_len <= 0:
                return True
            title_buf = ctypes.create_unicode_buffer(title_len + 1)
            _user32.GetWindowTextW(hwnd, title_buf, title_len + 1)
            pid = wt.DWORD()
            _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            windows.append((hwnd, pid.value, cls, title_buf.value))
            return True

        _user32.EnumWindows(_WNDENUMPROC(cb), 0)
        return windows

    def _find_host_hwnd(cwd: str | pathlib.Path | None) -> int | None:
        procs = _snapshot_processes()
        windows = _enum_app_windows()
        if not windows:
            return None

        by_pid: dict[int, list[tuple[int, str]]] = {}
        for hwnd, pid, _cls, title in windows:
            by_pid.setdefault(pid, []).append((hwnd, title))

        for pid, _exe in _ancestor_chain(os.getpid(), procs):
            if pid in by_pid:
                hwnd = pick_best_window(by_pid[pid], cwd)
                if hwnd is not None:
                    return hwnd

        con_hwnd = _kernel32.GetConsoleWindow()
        if con_hwnd:
            if _window_class(con_hwnd) != "PseudoConsoleWindow" and _user32.IsWindowVisible(con_hwnd):
                return con_hwnd
            con_pid = wt.DWORD()
            _user32.GetWindowThreadProcessId(con_hwnd, ctypes.byref(con_pid))
            for pid, _exe in _ancestor_chain(con_pid.value, procs):
                if pid in by_pid:
                    hwnd = pick_best_window(by_pid[pid], cwd)
                    if hwnd is not None:
                        return hwnd

        if os.environ.get("TERM_PROGRAM") == "vscode" or any(k.startswith("VSCODE_") for k in os.environ):
            vsc_windows = [
                (h, t) for h, p, _c, t in windows if procs.get(p, (0, ""))[1].lower() in _VSCODE_EXES
            ]
            if vsc_windows:
                return pick_best_window(vsc_windows, cwd)
        if "WT_SESSION" in os.environ:
            wt_windows = [(h, t) for h, _p, c, t in windows if c == "CASCADIA_HOSTING_WINDOW_CLASS"]
            if wt_windows:
                return pick_best_window(wt_windows, cwd)

        return pick_best_window([(h, t) for h, _p, _c, t in windows], cwd)

    def _get_monitor_pan(hwnd: int) -> tuple[float, int, int]:
        monitors: list[tuple[int, int, int, int, int]] = []

        def cb(hmon: int, _hdc: int, _lprect: object, _lparam: int) -> bool:
            mi = _MONITORINFO()
            mi.cbSize = ctypes.sizeof(_MONITORINFO)
            if _user32.GetMonitorInfoW(hmon, ctypes.byref(mi)):
                r = mi.rcMonitor
                monitors.append((hmon, r.left, r.top, r.right, r.bottom))
            return True

        _user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(cb), 0)
        target_hmon = _user32.MonitorFromWindow(hwnd, _MONITOR_DEFAULTTONEAREST)
        return compute_monitor_pan(monitors, target_hmon)

    def _read_registry_desktops() -> tuple[list[uuid.UUID], uuid.UUID | None]:
        ids: list[uuid.UUID] = []
        cur: uuid.UUID | None = None
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _VD_REG_PATH) as key:
                try:
                    ids_raw, _ = winreg.QueryValueEx(key, "VirtualDesktopIDs")
                    ids = [uuid.UUID(bytes_le=ids_raw[i : i + 16]) for i in range(0, len(ids_raw) - 15, 16)]
                except OSError:
                    pass
                try:
                    cur_raw, _ = winreg.QueryValueEx(key, "CurrentVirtualDesktop")
                    if len(cur_raw) == 16:
                        cur = uuid.UUID(bytes_le=cur_raw)
                except OSError:
                    pass
        except OSError:
            return ([], None)

        if cur is None:
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SESSION_INFO_PATH) as si_key:
                    idx = 0
                    while True:
                        try:
                            sub = winreg.EnumKey(si_key, idx)
                            idx += 1
                        except OSError:
                            break
                        try:
                            with winreg.OpenKey(si_key, f"{sub}\\VirtualDesktops") as vd_key:
                                cur_raw, _ = winreg.QueryValueEx(vd_key, "CurrentVirtualDesktop")
                                if len(cur_raw) == 16:
                                    cur = uuid.UUID(bytes_le=cur_raw)
                                    break
                        except OSError:
                            continue
            except OSError:
                pass

        return (ids, cur)

    def _get_desktop_info(hwnd: int) -> tuple[int | None, bool | None]:
        desktop_ids, cur_guid = _read_registry_desktops()
        hr_init = _ole32.CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
        did_init = hr_init >= 0
        ptr = ctypes.c_void_p()
        try:
            clsid = _GUID.from_buffer_copy(uuid.UUID(_CLSID_VDA).bytes_le)
            iid = _GUID.from_buffer_copy(uuid.UUID(_IID_VDA).bytes_le)
            hr = _ole32.CoCreateInstance(ctypes.byref(clsid), None, _CLSCTX_ALL, ctypes.byref(iid), ctypes.byref(ptr))
            if hr < 0 or not ptr.value:
                return (None, None)

            vtable = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            is_cur_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wt.HWND, ctypes.POINTER(wt.BOOL))(vtable[3])
            get_id_fn = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wt.HWND, ctypes.POINTER(_GUID))(vtable[4])

            on_cur_val = wt.BOOL()
            hr_cur = is_cur_fn(ptr, hwnd, ctypes.byref(on_cur_val))
            on_current: bool | None = bool(on_cur_val.value) if hr_cur >= 0 else None
            if on_current is True:
                return (0, True)

            win_guid_raw = _GUID()
            hr_id = get_id_fn(ptr, hwnd, ctypes.byref(win_guid_raw))
            if hr_id >= 0 and cur_guid is not None:
                win_guid = uuid.UUID(bytes_le=bytes(win_guid_raw))
                if win_guid in desktop_ids and cur_guid in desktop_ids:
                    delta = desktop_ids.index(win_guid) - desktop_ids.index(cur_guid)
                    return (delta, on_current)
            return (None, on_current)
        finally:
            if ptr.value:
                vt = ctypes.cast(ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
                ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)(vt[2])(ptr)
            if did_init:
                _ole32.CoUninitialize()


def detect_spatial(cwd: str | pathlib.Path | None = None) -> tuple[float, float, int, str]:
    if sys.platform != "win32":
        return (0.0, 0.0, 0, "non-windows")
    result = [(0.0, 0.0, 0, "timeout")]
    worker = threading.Thread(target=lambda: result.__setitem__(0, _detect_windows(cwd or os.getcwd())), daemon=True)
    worker.start()
    worker.join(DETECT_TIMEOUT_S)
    return result[0]


def _detect_windows(cwd: str | pathlib.Path) -> tuple[float, float, int, str]:
    try:
        hwnd = _find_host_hwnd(cwd)
        if not hwnd:
            return (0.0, 0.0, 0, "no window")
        mon_pan, col, total = _get_monitor_pan(hwnd)
        delta, on_current = _get_desktop_info(hwnd)
        return resolve_spatial_values(mon_pan, delta, on_current, f"monitor {col}/{total}")
    except Exception:
        return (0.0, 0.0, 0, "fallback")
