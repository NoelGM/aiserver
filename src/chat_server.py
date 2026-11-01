import json
import logging
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.error import HTTPError
from urllib.parse import urlparse

from active_models import CHAT_ROUTE, chat_request, read_selections, reset_active_models
from model_management_server import ModelManagementMixin
from configuration import ROOT, browser_settings, configure_logging, settings

logger = logging.getLogger(__name__)


class ChatHandler(ModelManagementMixin, SimpleHTTPRequestHandler):
    def do_POST(self):
        """Proxy chat completions using a single captured runtime selection.

        Args:
            None. Reads an OpenAI-compatible JSON body. Any supplied model,
            adapters or draft model are overridden by the active selection.

        Returns:
            None: Relays JSON or SSE bytes; malformed input returns 400 and
            unavailable backends return 502. In-flight selections do not change.
        """
        if urlparse(self.path).path != CHAT_ROUTE:
            self.send_json({"error": "Unknown endpoint"}, status=404)
            return
        try:
            selection = read_selections()["chat"]
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4 * 1024 * 1024:
                raise ValueError("A JSON body of at most 4 MiB is required.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must be a JSON object.")
        except (ValueError, UnicodeError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        except (OSError, KeyError, TypeError) as exc:
            self.send_json({"error": str(exc)}, status=500)
            return
        try:
            try:
                response = chat_request(data, selection)
            except HTTPError as exc:
                response = exc
        except (OSError, ValueError) as exc:
            self.send_json({"error": f"Chat backend unavailable: {exc}"}, status=502)
            return
        with response:
            self.send_response(response.status)
            self.send_header("Content-Type", response.headers.get(
                "Content-Type", "application/json"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.close_connection = True
            self.end_headers()
            try:
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except OSError:
                logger.warning("Chat relay connection closed before completion")

    def log_message(self, format, *args):
        """Record an HTTP request message at INFO level.

        Args:
            format (str): Percent-style log message template.
            *args: Values interpolated into the template.

        Returns:
            None: Logs the formatted message with the client's address.
        """
        logger.info("%s - %s", self.address_string(), format % args)

    def log_error(self, format, *args):
        """Record an HTTP error message at ERROR level.

        Args:
            format (str): Percent-style log message template.
            *args: Values interpolated into the template.

        Returns:
            None: Logs the formatted error with the client's address.
        """
        logger.error("%s - %s", self.address_string(), format % args)

    def do_GET(self):
        """Serve browser configuration, health and delegated GET routes.

        Args:
            None. Reads the request path from this handler instance.

        Returns:
            None: Sends JavaScript configuration or JSON health status, or
            delegates to model management and static-file handlers.
        """
        route = urlparse(self.path).path
        if route == "/config.js":
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
        elif route == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


def main():
    """Start the chat interface and model management HTTP listener.

    Args:
        None. Reads the listener host, port and web directory from configuration.

    Returns:
        None: Runs the HTTP serving loop until interrupted or stopped.
    """
    configure_logging()
    reset_active_models()
    handler = partial(ChatHandler, directory=str(ROOT / "web"))
    logger.info("Starting chat server on %s:%s", settings.web.host, settings.web.port)
    ThreadingHTTPServer((settings.web.host, settings.web.port), handler).serve_forever()


if __name__ == "__main__":
    main()
