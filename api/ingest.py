from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            # 讀取 body
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            print("📥 Received from Android:", data)

            # 回應 Android
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            resp = {"message": "APITEST OK", "receivedCount": len(data.get("lines", []))}
            self.wfile.write(json.dumps(resp).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            resp = {"message": "Server error", "error": str(e)}
            self.wfile.write(json.dumps(resp).encode("utf-8"))
