import json
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

from configuration import ROOT, browser_settings, settings


class WebHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path == "/config.js":
            body = (
                "window.AISERVER_CONFIG = "
                + json.dumps(browser_settings(), ensure_ascii=True)
                + ";\n"
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif urlparse(self.path).path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


def main():
    handler = partial(WebHandler, directory=str(ROOT / "web"))
    print(f"Starting web server on {settings.web.host}:{settings.web.port}")
    HTTPServer((settings.web.host, settings.web.port), handler).serve_forever()


if __name__ == "__main__":
    main()
