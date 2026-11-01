import argparse
import logging
from pathlib import Path
from urllib.parse import urlparse

from dynaconf import Dynaconf, Validator

ROOT = Path(__file__).resolve().parent.parent
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("configuration")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_DIR / "configuration.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)


def configure_logging():
    """Configure application logging with the shared level and format.

    Args:
        None.

    Returns:
        None: Configures the root logger when it has no existing handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
    )


SETTINGS_FILE = ROOT / "config" / "settings.yaml"
logger.info("Loading configuration from %s", SETTINGS_FILE)
if not SETTINGS_FILE.is_file():
    logger.error("Configuration not found: %s", SETTINGS_FILE)
    raise FileNotFoundError(f"Configuration not found: {SETTINGS_FILE}")

settings = Dynaconf(
    settings_files=[str(SETTINGS_FILE)],
    environments=False,
    envvar_prefix="AISERVER",
    validators=[
        Validator(f"{service}.host", must_exist=True, is_type_of=str, len_min=1)
        for service in ("web", "chat", "images")
    ] + [
        Validator(f"{service}.port", must_exist=True, is_type_of=int, gte=1, lte=65535)
        for service in ("web", "chat", "images")
    ] + [
        Validator(name, must_exist=True, is_type_of=str, len_min=1)
        for name in (
            "chat.model", "chat.api_url", "embedding.model", "images.generate_model",
            "images.edit_model", "images.generate_command", "images.edit_command",
            "images.generate_url", "images.edit_url",
        )
    ] + [
        Validator(name, must_exist=True, is_type_of=int, gte=1)
        for name in (
            "chat.max_tokens", "images.width", "images.height",
            "images.steps", "images.timeout", "images.quantize",
            "images.min_dimension", "images.max_dimension",
        )
    ] + [Validator("images.low_ram", must_exist=True, is_type_of=bool)],
)
settings.validators.validate()
if not (
    settings.images.min_dimension <= settings.images.width <= settings.images.max_dimension
    and settings.images.min_dimension <= settings.images.height <= settings.images.max_dimension
):
    raise ValueError("Image dimensions must be within the configured dimension limits.")
if settings.images.min_dimension % 16 or settings.images.max_dimension % 16:
    raise ValueError("Image dimension limits must be multiples of 16.")
if settings.images.quantize not in (3, 4, 5, 6, 8):
    raise ValueError("Image quantization must be 3, 4, 5, 6 or 8.")
if settings.images.steps > 50:
    raise ValueError("Image steps must not exceed 50.")
if len({settings.web.port, settings.chat.port, settings.images.port}) != 3:
    raise ValueError("Web, chat and image services must use different ports.")


def service_url(template, service, hostname):
    """Expand and validate a service URL, enclosing IPv6 hosts in brackets.

    Args:
        template (str): URL containing optional hostname and port placeholders.
        service: Service settings providing the port.
        hostname (str): Hostname, IP address or deferred hostname placeholder.

    Returns:
        str: Expanded HTTP or HTTPS URL.

    Raises:
        ValueError: If the URL lacks a valid scheme, hostname or path.
    """
    host = f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname
    url = template.format(hostname=host, port=service.port)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or not parsed.path:
        raise ValueError(f"Invalid API URL: {url}")
    return url


def image_route(template):
    """Extract the path of a configured image endpoint.

    Args:
        template (str): Image API URL with optional hostname and port placeholders.

    Returns:
        str: Endpoint path using the configured image service port.

    Raises:
        ValueError: If the URL is invalid or contains a query or fragment.
    """
    parsed = urlparse(service_url(template, settings.images, "localhost"))
    if parsed.query or parsed.fragment:
        raise ValueError("Image endpoint URLs must not contain query strings or fragments.")
    return parsed.path


GENERATE_ROUTE = image_route(settings.images.generate_url)
EDIT_ROUTE = image_route(settings.images.edit_url)
if GENERATE_ROUTE == EDIT_ROUTE or "/health" in (GENERATE_ROUTE, EDIT_ROUTE):
    raise ValueError("Image generation, editing and health routes must be different.")


def browser_settings():
    """Collect the chat and image settings exposed to the browser.

    Args:
        None.

    Returns:
        dict: Same-origin active-model chat proxy URL and generation defaults;
        image URLs retain the hostname placeholder for browser resolution.
    """
    return {
        "chat": {
            "api_url": "/v1/chat/completions",
            "max_tokens": settings.chat.max_tokens,
        },
        "images": {
            "generate_url": service_url(settings.images.generate_url, settings.images, "{hostname}"),
            "edit_url": service_url(settings.images.edit_url, settings.images, "{hostname}"),
            "width": settings.images.width,
            "height": settings.images.height,
            "min_dimension": settings.images.min_dimension,
            "max_dimension": settings.images.max_dimension,
            "steps": settings.images.steps,
        },
    }


browser_settings()
logger.info("Configuration loaded and validated successfully.")


def main():
    """Print a configuration field selected through command-line arguments.

    Args:
        None. Reads the service and field from the process command line.

    Returns:
        None: Writes the selected value or computed health URL to stdout.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("web", "chat", "images", "embedding"))
    parser.add_argument(
        "field",
        choices=("host", "port", "model", "generate_command", "edit_command", "health_url"),
    )
    args = parser.parse_args()
    service = settings[args.service]
    if args.field == "health_url":
        host = "127.0.0.1" if service.host == "0.0.0.0" else service.host
        if host == "::":
            host = "::1"
        value = service_url("http://{hostname}:{port}/health", service, host)
    else:
        value = service[args.field]
    logger.info("%s.%s = %s", args.service, args.field, value)
    print(value)


if __name__ == "__main__":
    main()
