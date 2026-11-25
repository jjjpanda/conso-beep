import hmac
import http.server
import json
import math
import pathlib
import queue
import secrets
import socket
import sys
import threading
import time

import sounddevice as sd

import audio

DAEMON_PORT = 47392
DAEMON_IDLE_S = 600
DAEMON_MAX_MSG_BYTES = 65536
DEFAULT_ASK_TIMEOUT_S = 120.0
MAX_ASK_TIMEOUT_S = 600.0
DASHBOARD_HTML = pathlib.Path(__file__).with_name("dashboard.html")


class _DaemonHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def serve(args, parse_request, resolve_spatial_fn, melodies, port=DAEMON_PORT, ready_event=None, bound_port_out=None):
    """Run the localhost HTTP dashboard, permission inbox, and playback daemon."""
    token = secrets.token_hex(16)
    controls = {"muted": False, "gain_override": None, "keep_alive": False}
    permissions: dict[str, dict] = {}
    state_lock = threading.Lock()
    activity = {"ts": time.monotonic()}
    stop_event = threading.Event()
    mic = audio.MicRing()
    play_queue: queue.Queue = queue.Queue()

    def wake_all_permissions():
        with state_lock:
            for entry in permissions.values():
                if entry["decision"] is None:
                    entry["decision"] = "timeout"
                    entry["reason"] = "daemon stopped"
                entry["event"].set()

    def worker_loop():
        while True:
            request = play_queue.get()
            if request is None:
                break
            try:
                with state_lock:
                    if controls["muted"]:
                        continue
                    gain_override = controls["gain_override"]
                root_hz, gain, timbre = mic.resolve_for_args(
                    request,
                    gain_override=gain_override,
                    timbre_override=getattr(request, "timbre_override", None),
                )
                placement = resolve_spatial_fn(request)
                mic.mute_playback()
                try:
                    audio.play_notes(root_hz, gain, request.notes, *placement, wait=True, timbre=timbre)
                except sd.PortAudioError:
                    mic.close()
                    sd._terminate()
                    sd._initialize()
                    audio.play_notes(root_hz, gain, request.notes, *placement, wait=True, timbre=timbre)
                finally:
                    mic.unmute_playback()
            except Exception as e:
                print(f"[beep] daemon playback error: {e!r}", file=sys.stderr)

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _valid_origin_and_host(self):
            bound = self.server.server_port
            host = (self.headers.get("Host") or "").strip().lower()
            allowed_hosts = {f"127.0.0.1:{bound}", f"localhost:{bound}", "127.0.0.1", "localhost"}
            if host not in allowed_hosts:
                return False
            origin = self.headers.get("Origin")
            if origin is not None:
                allowed_origins = {f"http://127.0.0.1:{bound}", f"http://localhost:{bound}"}
                if origin.strip().lower() not in allowed_origins:
                    return False
            return True

        def _valid_token(self):
            supplied = self.headers.get("X-Conso-Token", "")
            return hmac.compare_digest(supplied, token)

        def _send_bytes(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status, payload):
            self._send_bytes(status, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

        def _read_json_body(self):
            raw_len = self.headers.get("Content-Length")
            if raw_len is None:
                self._send_json(400, {"error": "missing Content-Length"})
                return None
            try:
                length = int(raw_len)
            except ValueError:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length < 0:
                self._send_json(400, {"error": "invalid Content-Length"})
                return None
            if length > DAEMON_MAX_MSG_BYTES:
                self._send_json(413, {"error": "payload too large"})
                return None
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, {"error": "invalid JSON"})
                return None

        def do_GET(self):
            if not self._valid_origin_and_host():
                self._send_json(403, {"error": "forbidden"})
                return
            path = self.path.split("?", 1)[0]
            if path == "/":
                html = DASHBOARD_HTML.read_text(encoding="utf-8").replace("{{TOKEN}}", token)
                self._send_bytes(200, "text/html; charset=utf-8", html.encode("utf-8"))
                return
            if path == "/api/status":
                mic_status = mic.snapshot_status()
                with state_lock:
                    ctrl = dict(controls)
                    if ctrl["keep_alive"] or permissions:
                        idle_remaining = None
                    else:
                        idle_remaining = max(0, round(DAEMON_IDLE_S - (time.monotonic() - activity["ts"])))
                    ctrl["idle_remaining_s"] = idle_remaining
                    perms = [
                        {
                            "id": e["id"],
                            "title": e["title"],
                            "detail": e["detail"],
                            "cwd": e["cwd"],
                            "source": e["source"],
                            "created_at": e["created_at"],
                        }
                        for e in permissions.values()
                    ]
                self._send_json(
                    200,
                    {
                        "mic": mic_status,
                        "controls": ctrl,
                        "permissions": perms,
                        "melodies": list(melodies.keys()),
                        "melody_notes": {k: " ".join(v) for k, v in melodies.items()},
                    },
                )
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            if not self._valid_origin_and_host():
                self._send_json(403, {"error": "forbidden"})
                return
            path = self.path.split("?", 1)[0]

            if path in ("/api/permissions/decide", "/api/controls", "/api/stop") and not self._valid_token():
                self._send_json(403, {"error": "invalid token"})
                return

            body = self._read_json_body()
            if body is None:
                return

            if path == "/api/play":
                if isinstance(body, dict) and "argv" not in body:
                    melody_name = body.get("melody")
                    raw_notes = body.get("notes")
                    use_mic = body.get("use_mic", True)
                    if not isinstance(use_mic, bool):
                        self._send_json(400, {"error": "use_mic must be boolean"})
                        return
                    if melody_name is not None:
                        if not isinstance(melody_name, str) or melody_name not in melodies:
                            self._send_json(400, {"error": "unknown melody"})
                            return
                        note_tokens = list(melodies[melody_name])
                    elif isinstance(raw_notes, str):
                        note_tokens = raw_notes.split()
                    elif isinstance(raw_notes, list) and all(isinstance(x, str) for x in raw_notes):
                        note_tokens = raw_notes
                    else:
                        self._send_json(400, {"error": "expected melody or notes"})
                        return
                    if not note_tokens:
                        self._send_json(400, {"error": "at least one note is required"})
                        return
                    try:
                        pan = float(body.get("pan", 0.0))
                        muffle = float(body.get("muffle", 0.0))
                        ticks = int(body.get("ticks", 0))
                    except (TypeError, ValueError):
                        self._send_json(400, {"error": "invalid spatial parameter"})
                        return
                    argv = (["--mic"] if use_mic else []) + [
                        f"--pan={pan:.4f}",
                        f"--muffle={muffle:.4f}",
                        f"--ticks={ticks}",
                        "--",
                        *note_tokens,
                    ]
                    timbre_req = body.get("timbre")
                else:
                    argv = body.get("argv") if isinstance(body, dict) else body
                    timbre_req = None
                if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
                    self._send_json(400, {"error": "expected list of strings"})
                    return
                try:
                    request = parse_request(argv)
                except (Exception, SystemExit) as e:
                    self._send_json(400, {"error": repr(e)})
                    return
                request.timbre_override = timbre_req if timbre_req in audio.TIMBRES else None
                with state_lock:
                    activity["ts"] = time.monotonic()
                    muted = controls["muted"]
                if request.notes and not muted:
                    play_queue.put(request)
                self._send_json(200, {"ok": True, "muted": muted})
                return

            if path == "/api/permissions":
                if not isinstance(body, dict):
                    self._send_json(400, {"error": "expected JSON object"})
                    return
                title = body.get("title")
                if not isinstance(title, str) or not title.strip():
                    self._send_json(400, {"error": "title is required"})
                    return
                detail = body.get("detail", "")
                cwd = body.get("cwd", "")
                source = body.get("source", "")
                if not all(isinstance(v, str) for v in (detail, cwd, source)):
                    self._send_json(400, {"error": "detail, cwd, and source must be strings"})
                    return
                timeout = body.get("timeout", DEFAULT_ASK_TIMEOUT_S)
                if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0 or timeout > MAX_ASK_TIMEOUT_S:
                    self._send_json(400, {"error": "invalid timeout"})
                    return

                entry = {
                    "id": secrets.token_hex(8),
                    "title": title.strip(),
                    "detail": detail,
                    "cwd": cwd,
                    "source": source,
                    "created_at": time.time(),
                    "event": threading.Event(),
                    "decision": None,
                    "reason": "",
                }
                with state_lock:
                    permissions[entry["id"]] = entry
                    activity["ts"] = time.monotonic()
                    muted = controls["muted"]

                melody_name = body.get("melody", "permission")
                if isinstance(melody_name, str) and melody_name in melodies and not muted:
                    try:
                        root_flags = body.get("root_flags") or []
                        if isinstance(root_flags, list) and all(isinstance(x, str) for x in root_flags):
                            pan = float(body.get("pan", 0.0))
                            muffle = float(body.get("muffle", 0.0))
                            ticks = int(body.get("ticks", 0))
                            argv = [*root_flags, f"--pan={pan:.4f}", f"--muffle={muffle:.4f}", f"--ticks={ticks}", "--", *melodies[melody_name]]
                            req = parse_request(argv)
                            if req.notes:
                                play_queue.put(req)
                    except (Exception, SystemExit):
                        pass

                try:
                    entry["event"].wait(timeout=float(timeout))
                finally:
                    with state_lock:
                        permissions.pop(entry["id"], None)
                        activity["ts"] = time.monotonic()

                self._send_json(200, {"decision": entry["decision"] or "timeout", "reason": entry["reason"]})
                return

            if path == "/api/permissions/decide":
                if not isinstance(body, dict):
                    self._send_json(400, {"error": "expected JSON object"})
                    return
                ids = body.get("ids")
                decision = body.get("decision")
                reason = body.get("reason", "")
                if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
                    self._send_json(400, {"error": "ids must be a list of strings"})
                    return
                if decision not in ("allow", "deny") or not isinstance(reason, str):
                    self._send_json(400, {"error": "invalid decision or reason"})
                    return
                decided = []
                with state_lock:
                    activity["ts"] = time.monotonic()
                    for pid in ids:
                        entry = permissions.get(pid)
                        if entry is not None and entry["decision"] is None:
                            entry["decision"] = decision
                            entry["reason"] = reason
                            entry["event"].set()
                            decided.append(pid)
                self._send_json(200, {"decided": decided})
                return

            if path == "/api/controls":
                if not isinstance(body, dict):
                    self._send_json(400, {"error": "expected JSON object"})
                    return
                if "muted" in body and not isinstance(body["muted"], bool):
                    self._send_json(400, {"error": "muted must be boolean"})
                    return
                if "keep_alive" in body and not isinstance(body["keep_alive"], bool):
                    self._send_json(400, {"error": "keep_alive must be boolean"})
                    return
                if "gain_override" in body:
                    go = body["gain_override"]
                    if go is not None and (not isinstance(go, (int, float)) or isinstance(go, bool) or not math.isfinite(go) or go < 0.0 or go > 1.0):
                        self._send_json(400, {"error": "gain_override must be null or in [0, 1]"})
                        return
                with state_lock:
                    activity["ts"] = time.monotonic()
                    if "muted" in body:
                        controls["muted"] = body["muted"]
                    if "keep_alive" in body:
                        controls["keep_alive"] = body["keep_alive"]
                    if "gain_override" in body:
                        go = body["gain_override"]
                        controls["gain_override"] = None if go is None else float(go)
                    out = dict(controls)
                self._send_json(200, {"controls": out})
                return

            if path == "/api/stop":
                stop_event.set()
                wake_all_permissions()
                self._send_json(200, {"ok": True})
                return

            self._send_json(404, {"error": "not found"})

    try:
        server = _DaemonHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        if args.notes:
            root_hz, gain, timbre = audio.resolve_mic(args)
            audio.play_notes(root_hz, gain, args.notes, *resolve_spatial_fn(args), timbre=timbre)
        return 0

    server.timeout = 0.5
    if bound_port_out is not None:
        bound_port_out.append(server.server_port)
    if ready_event is not None:
        ready_event.set()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    try:
        if args.notes:
            play_queue.put(args)
        while not stop_event.is_set():
            server.handle_request()
            with state_lock:
                idle_expired = (
                    not controls["keep_alive"]
                    and not permissions
                    and (time.monotonic() - activity["ts"]) >= DAEMON_IDLE_S
                )
            if idle_expired:
                break
    finally:
        wake_all_permissions()
        server.server_close()
        play_queue.put(None)
        worker.join(timeout=5.0)
        mic.close()

    return 0
