#!/usr/bin/env python3
"""Standalone RPC Tracker (no popen_pool dependency)"""
import sys, os, time, socket, struct, json, threading, logging
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TVM_ROOT = SCRIPT_DIR.parent.parent / "third_party" / "tvm"
sys.path.insert(0, str(TVM_ROOT / "python"))

from tvm._ffi.base import RPC_TRACKER_MAGIC

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("tracker")

TRACKER_CODE_STOP = 0
TRACKER_CODE_PUT = 1
TRACKER_CODE_UPDATE_INFO = 2
TRACKER_CODE_GET_PENDING_MATCHKEYS = 3
TRACKER_CODE_SUCCESS = 4
TRACKER_CODE_FAILURE = 5
TRACKER_CODE_PING = 6

class TrackerServer:
    def __init__(self, host="0.0.0.0", port=9190, port_end=9199, timeout=None):
        self.host = host
        self.timeout = timeout
        self._queue = {}           # key -> list of pending connections
        self._info = {}            # key -> addr info
        self._stop_key = os.urandom(16).hex()

        # Find available port
        self.port = port
        for p in range(port, port_end + 1):
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self._sock.bind((host, p))
                self._sock.listen(10)
                self.port = p
                break
            except OSError:
                if p == port_end:
                    raise RuntimeError(f"Cannot bind to any port in {port}-{port_end}")

        log.info(f"Tracker listening on {host}:{self.port}")
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while self._running:
            try:
                self._sock.settimeout(1.0)
                conn, addr = self._sock.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    log.error(f"Accept error: {e}")

    def _handle(self, sock, addr):
        try:
            if self.timeout:
                sock.settimeout(self.timeout)

            # Read magic
            magic = struct.unpack("<i", sock.recv(4))[0]
            if magic != RPC_TRACKER_MAGIC:
                log.warning(f"Bad magic from {addr}")
                sock.close()
                return

            sock.sendall(struct.pack("<i", RPC_TRACKER_MAGIC))

            # Read request
            data = _recv_json(sock)
            code = data[0]

            if code == TRACKER_CODE_PUT:
                key, conn_info = data[1], data[2]
                log.info(f"PUT key={key} from {addr}")
                self._info[key] = conn_info
                if key in self._queue and self._queue[key]:
                    pending = self._queue[key].pop(0)
                    _send_json(sock, [TRACKER_CODE_SUCCESS, pending])
                else:
                    _send_json(sock, [TRACKER_CODE_SUCCESS])

            elif code == TRACKER_CODE_GET_PENDING_MATCHKEYS:
                keys = list(self._queue.keys())
                _send_json(sock, [TRACKER_CODE_SUCCESS, keys])

            elif code == TRACKER_CODE_UPDATE_INFO:
                key, conn_info = data[1], data[2]
                self._info[key] = conn_info
                _send_json(sock, [TRACKER_CODE_SUCCESS])

            elif code == TRACKER_CODE_PING:
                # Queue this request and notify
                key = data[1]
                if key not in self._queue:
                    self._queue[key] = []
                self._queue[key].append(data)
                _send_json(sock, [TRACKER_CODE_SUCCESS, "waiting"])

            elif code == TRACKER_CODE_STOP:
                if len(data) > 1 and data[1] == self._stop_key:
                    log.info("Stop received")
                    self._running = False
                _send_json(sock, [TRACKER_CODE_SUCCESS])

            else:
                log.warning(f"Unknown code {code}")
                _send_json(sock, [TRACKER_CODE_FAILURE])

        except Exception as e:
            log.error(f"Handler error: {e}")
        finally:
            try: sock.close()
            except: pass

    def stop(self):
        self._running = False
        try: self._sock.close()
        except: pass

    def wait(self):
        while self._running:
            time.sleep(0.5)

def _recv_json(sock):
    """Receive length-prefixed JSON"""
    length = struct.unpack("<i", sock.recv(4))[0]
    data = b""
    while len(data) < length:
        chunk = sock.recv(length - len(data))
        if not chunk: break
        data += chunk
    return json.loads(data.decode())

def _send_json(sock, obj):
    """Send length-prefixed JSON"""
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("<i", len(data)) + data)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Standalone TVM RPC Tracker")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9190)
    parser.add_argument("--port-end", type=int, default=9199)
    args = parser.parse_args()

    t = TrackerServer(host=args.host, port=args.port, port_end=args.port_end)
    try:
        t.wait()
    except KeyboardInterrupt:
        log.info("Shutting down...")
        t.stop()
