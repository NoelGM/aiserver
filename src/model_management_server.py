import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

from api_docs import openapi_spec
from active_models import SelectionError, active_models, switch_model
from configuration import ROOT, settings

logger = logging.getLogger(__name__)


def model_metadata(snapshot):
    """Infer a local model's type and capabilities from its metadata.

    Args:
        snapshot (Path): Model directory or Hugging Face snapshot directory.

    Returns:
        tuple: Model type and list of inferred capabilities. Unrecognized
        metadata produces type ``unknown`` and an empty capability list.

    Raises:
        OSError: If metadata files cannot be read.
        ValueError: If JSON metadata or its architectures field is invalid.
    """
    configs = []
    for relative_path in (
        "model_index.json", "config.json", "transformer/config.json",
        "unet/config.json", "vae/config.json",
    ):
        path = snapshot / relative_path
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError(f"Invalid model metadata: {path}")
            architectures = config.get("architectures", [])
            if not isinstance(architectures, list):
                raise ValueError(f"Invalid model architectures: {path}")
            configs.append(config)

    classes = [
        name
        for config in configs
        for name in [config.get("_class_name", ""), *config.get("architectures", [])]
        if isinstance(name, str)
    ]
    if any(
        any(marker in name for marker in ("Flux", "StableDiffusion", "PixArt", "SanaPipeline"))
        or "Transformer2DModel" in name
        or "UNet2D" in name or "AutoencoderKL" in name
        for name in classes
    ):
        capabilities = ["generate"]
        if any("Flux2" in name or "Kontext" in name or "Img2Img" in name for name in classes):
            capabilities.append("edit")
        return "image", capabilities
    if (snapshot / "modules.json").is_file() or (snapshot / "sentence_bert_config.json").is_file():
        return "embedding", ["embedding"]
    if any("ForCausalLM" in name or "ForConditionalGeneration" in name for name in classes):
        return "chat", ["chat"]
    return "unknown", []


def model_catalog():
    """Discover models in the Hugging Face cache and configured local paths.

    Args:
        None. Reads the cache location and current model configuration.

    Returns:
        dict: List envelope containing model identifiers, inferred types,
        capabilities, sources and cache warnings. Cached entries without
        weight files are excluded; download completeness is not verified.

    Raises:
        OSError: If cache or metadata files cannot be accessed.
        ValueError: If model metadata is invalid.
    """
    models = []
    warnings = []
    cache = Path(HF_HUB_CACHE)
    if cache.exists():
        info = scan_cache_dir(cache)
        for warning in info.warnings:
            logger.warning("Model cache warning: %s", warning)
            warnings.append(str(warning))
        for repo in sorted(info.repos, key=lambda item: item.repo_id):
            if repo.repo_type != "model":
                continue
            revisions = sorted(repo.revisions, key=lambda item: item.last_modified, reverse=True)
            snapshots = [
                revision.snapshot_path for revision in revisions
                if any(
                    file.file_name.endswith((".safetensors", ".bin", ".gguf", ".npz"))
                    for file in revision.files
                )
            ]
            if not snapshots:
                continue
            model_type, capabilities = model_metadata(snapshots[0])
            models.append({
                "id": repo.repo_id,
                "object": "model",
                "type": model_type,
                "capabilities": capabilities,
                "source": "huggingface_cache",
            })
    local_paths = set()
    for model in (
        settings.chat.model, settings.embedding.model,
        settings.images.generate_model, settings.images.edit_model,
    ):
        path = Path(model).expanduser()
        if not path.is_dir():
            continue
        path = path.resolve()
        if path in local_paths:
            continue
        local_paths.add(path)
        model_type, capabilities = model_metadata(path)
        models.append({
            "id": str(path),
            "object": "model",
            "type": model_type,
            "capabilities": capabilities,
            "source": "local",
        })
    return {"object": "list", "data": models, "warnings": warnings}


def configured_models():
    """Read the models assigned to each configured service.

    Args:
        None. Reads the current application settings.

    Returns:
        dict: Startup chat and embedding identifiers plus image generation and
        editing defaults, independent of runtime selections and readiness.
    """
    return {
        "chat": settings.chat.model,
        "embedding": settings.embedding.model,
        "images": {
            "generate": settings.images.generate_model,
            "edit": settings.images.edit_model,
        },
    }


class ModelManagementMixin:
    def do_PUT(self):
        """Validate and publish a runtime model selection without modifying YAML.

        Args:
            None. Reads a JSON object with service and model from the HTTP body.

        Returns:
            None: Sends active selections on success, 400 for malformed input,
            422 for rejected candidates or 500 for state access failures.
        """
        if urlparse(self.path).path != "/models/active":
            self.send_json({"error": "Unknown endpoint"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("A JSON body of at most 4096 bytes is required.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict) or set(data) != {"service", "model"}:
                raise ValueError("Request must contain exactly service and model.")
        except (ValueError, UnicodeError) as exc:
            self.send_json({"error": str(exc)}, status=400)
            return
        try:
            self.send_json(switch_model(data["service"], data["model"]))
        except SelectionError as exc:
            self.send_json({"error": str(exc)}, status=422)
        except (OSError, ValueError) as exc:
            logger.exception("Unable to update active model state")
            self.send_json({"error": str(exc)}, status=500)

    def send_json(self, data, status=200):
        """Send a JSON HTTP response without caching.

        Args:
            data: JSON-serializable response payload.
            status (int): HTTP status code, defaulting to 200.

        Returns:
            None: Writes headers and the UTF-8 JSON body to the client.
        """
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_document(body, "application/json; charset=utf-8", status)

    def send_document(self, body, content_type, status=200):
        """Send a document with its content type and length.

        Args:
            body (bytes): Encoded response body.
            content_type (str): MIME type, optionally including a charset.
            status (int): HTTP status code, defaulting to 200.

        Returns:
            None: Writes a non-cacheable HTTP response to the client.
        """
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Serve model management and API documentation GET requests.

        Args:
            None. Reads the request path from this handler instance.

        Returns:
            None: Sends configured models, the local catalog, OpenAPI JSON or
            Swagger HTML. Catalog read failures produce HTTP 500; unmatched
            routes are delegated to the next handler in the inheritance chain.
        """
        route = urlparse(self.path).path
        if route == "/models/configured":
            self.send_json(configured_models())
        elif route == "/models/active":
            try:
                self.send_json(active_models())
            except (OSError, ValueError, KeyError, TypeError) as exc:
                logger.exception("Unable to read active model state")
                self.send_json({"error": str(exc)}, status=500)
        elif route in ("/models/list", "/openapi.json"):
            try:
                catalog = model_catalog()
            except (OSError, ValueError) as exc:
                logger.exception("Unable to read local model catalog")
                self.send_json({"error": str(exc)}, status=500)
                return
            self.send_json(
                catalog if route == "/models/list"
                else openapi_spec(catalog, configured_models())
            )
        elif route in ("/docs", "/docs/"):
            self.send_document(
                (ROOT / "web" / "swagger.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        else:
            super().do_GET()
