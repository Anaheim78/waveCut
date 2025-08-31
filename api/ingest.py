from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def _json(self, obj, status=200, headers=None):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        # CORS 預檢（如果你要跨網域）
        self._json({}, 204, {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        })

    def do_POST(self):
        if self.path != "/api/ingest":
            self._json({"ok": False, "error": "Not Found"}, 404); return

        # 讀取 JSON body
        try:
            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception:
            self._json({"ok": False, "error": "invalid JSON"}, 400); return

        values = data.get("values", [])
        if not isinstance(values, list):
            self._json({"ok": False, "error": "`values` must be a list"}, 400); return

        if len(values) > 5000:
            self._json({"ok": False, "error": "too many values"}, 413); return

        try:
            nums = [float(x) for x in values]
        except Exception:
            self._json({"ok": False, "error": "values must be numbers"}, 400); return

        s = sum(nums)
        n = len(nums)
        avg = s / n if n else 0.0

        self._json(
            {"ok": True, "count": n, "sum": s, "avg": avg},
            200,
            {"Access-Control-Allow-Origin": "*"}
        )
