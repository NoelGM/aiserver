import json
import logging
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

from model_management_server import ModelManagementMixin
from configuration import ROOT, browser_settings, configure_logging, settings

logger = logging.getLogger(__name__)


class ChatHandler(ModelManagementMixin, SimpleHTTPRequestHandler):
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
    handler = partial(ChatHandler, directory=str(ROOT / "web"))
    logger.info("Starting chat server on %s:%s", settings.web.host, settings.web.port)
    HTTPServer((settings.web.host, settings.web.port), handler).serve_forever()


if __name__ == "__main__":
    main()
