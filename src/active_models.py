"""Runtime selections shared by the web and image processes; never edits YAML."""

import fcntl
import json
import os
import re
import struct
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

from configuration import ROOT, service_url, settings

STATE_FILE = ROOT / "data" / "models" / "active.json"
SWITCH_LOCK = threading.Lock()
SERVICES = ("chat", "image_generate", "image_edit", "embedding")
CHAT_ROUTE = "/v1/chat/completions"


class SelectionError(ValueError):
    """A candidate cannot safely be selected."""


def default_selections():
    """Build selections from startup settings without loading any weights.

    Args:
        None.

    Returns:
        dict: Internal selections, marked as unvalidated startup defaults.
    """
    models = (settings.chat.model, settings.images.generate_model,
              settings.images.edit_model, settings.embedding.model)
    return {service: {"model": model, "path": None, "alias": None,
                      "validation": "startup_default"}
            for service, model in zip(SERVICES, models)}


@contextmanager
def _state_lock():
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _write_state(state):
    # The sibling file stays on the same filesystem, so readers see one complete version.
    pending = STATE_FILE.with_suffix(".pending")
    try:
        with pending.open("w", encoding="utf-8") as output:
            json.dump(state, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, STATE_FILE)
    finally:
        pending.unlink(missing_ok=True)


def reset_active_models():
    """Reset shared selections to YAML/environment defaults at application startup.

    Args:
        None.

    Returns:
        None: Atomically replaces runtime state; does not change configuration.
    """
    with _state_lock():
        _write_state(default_selections())


def read_selections():
    """Capture one consistent version of the cross-process active selections.

    Args:
        None.

    Returns:
        dict: Internal selections including pinned paths, or defaults before startup.

    Raises:
        OSError: If existing state cannot be read.
        ValueError: If existing state is corrupt (never silently resets a selection).
    """
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default_selections()
    if not isinstance(state, dict) or set(state) != set(SERVICES):
        raise ValueError("Invalid runtime model state.")
    for selection in state.values():
        if (not isinstance(selection, dict)
                or set(selection) != {"model", "path", "alias", "validation"}
                or not isinstance(selection.get("model"), str)
                or not selection["model"]
                or selection.get("validation") not in (
                    "startup_default", "local_files_checked", "backend_loaded")
                or any(selection.get(key) is not None and not isinstance(selection[key], str)
                       for key in ("path", "alias"))):
            raise ValueError("Invalid runtime model selection.")
    return state


def active_models():
    """Describe active identifiers and their readiness without exposing local paths.

    Args:
        None.

    Returns:
        dict: Four selections, validation levels and explicit embedding unavailability.
    """
    state = read_selections()
    return {
        **{service: state[service]["model"] for service in SERVICES},
        "validation": {service: state[service]["validation"] for service in SERVICES},
        "embedding_inference_available": False,
    }


def backend_url():
    """Resolve the configured chat backend URL for server-to-server requests.

    Args:
        None.

    Returns:
        str: Backend URL, using loopback for wildcard listener addresses.
    """
    host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(settings.chat.host, settings.chat.host)
    return service_url(settings.chat.api_url, settings.chat, host)


def chat_request(data, selection, timeout=600):
    """Forward a captured chat selection to MLX-LM, including streamed responses.

    Args:
        data (dict): OpenAI-compatible completion request.
        selection (dict): Internal active selection captured when the request starts.
        timeout (int): Socket timeout in seconds.

    Returns:
        HTTPResponse: Open backend response; the caller must close it.
    """
    payload = dict(data)
    payload["model"] = selection["path"] or selection["model"]
    # Runtime model selection must not introduce arbitrary adapter/draft file access.
    payload["adapters"] = None
    payload["draft_model"] = None
    request = Request(backend_url(), json.dumps(payload).encode("utf-8"),
                      {"Content-Type": "application/json"}, method="POST")
    return urlopen(request, timeout=timeout)


def image_registry():
    """Read the installed MFLUX registry for the supported FLUX.2 Klein commands.

    Args:
        None.

    Returns:
        dict: Canonical repository IDs and aliases mapped to (CLI alias, repository).
              Other pipelines and custom repositories are deliberately not accepted.
    """
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    registry = {}
    for alias, config in AVAILABLE_MODELS.items():
        if alias.startswith("flux2-klein"):
            value = (alias, config.model_name)
            for name in (alias, config.model_name, *config.aliases):
                registry[name] = value
    return registry


def _snapshot(model):
    # Only catalogued cache repositories and configured local directories are selectable.
    for configured in (settings.chat.model, settings.embedding.model,
                       settings.images.generate_model, settings.images.edit_model):
        path = Path(configured).expanduser()
        if path.is_dir() and model in (configured, str(path.resolve())):
            return path.resolve()
    if Path(HF_HUB_CACHE).exists():
        for repo in scan_cache_dir(HF_HUB_CACHE).repos:
            if repo.repo_type == "model" and repo.repo_id == model:
                revisions = sorted(repo.revisions, key=lambda revision: revision.last_modified,
                                   reverse=True)
                if revisions:
                    return revisions[0].snapshot_path
    raise SelectionError("Model is not downloaded in the HF cache or a configured local directory.")


def _json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectionError(f"Invalid metadata: {path.name}")
    return value


def _weights(directory):
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        raise SelectionError("Complete safetensors weights are required.")
    tensors = {}
    for file in files:
        with file.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SelectionError("Truncated safetensors file.")
            size = struct.unpack("<Q", prefix)[0]
            if not 2 <= size <= 16 * 1024 * 1024:
                raise SelectionError("Invalid safetensors header size.")
            header = json.loads(stream.read(size))
        data_size = file.stat().st_size - 8 - size
        if not isinstance(header, dict):
            raise SelectionError("Invalid safetensors header.")
        names = set()
        intervals = []
        end = 0
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or not all(isinstance(offset, int) for offset in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= data_size):
                raise SelectionError("Truncated or invalid weight tensor.")
            sizes = {"BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
                     "U32": 4, "I32": 4, "U64": 8, "I64": 8,
                     "F16": 2, "BF16": 2, "F32": 4, "F64": 8,
                     "F8_E4M3": 1, "F8_E5M2": 1}
            shape = tensor.get("shape")
            size = sizes.get(tensor.get("dtype"))
            if (size is None or not isinstance(shape, list)
                    or not all(type(dimension) is int and dimension >= 0 for dimension in shape)):
                raise SelectionError("Invalid tensor shape or dtype.")
            for dimension in shape:
                size *= dimension
            if size != offsets[1] - offsets[0]:
                raise SelectionError("Tensor payload size does not match its shape.")
            names.add(name)
            intervals.append(tuple(offsets))
            end = max(end, offsets[1])
        if not names or end != data_size:
            raise SelectionError("Incomplete safetensors payload.")
        position = 0
        for start, stop in sorted(intervals):
            if start != position:
                raise SelectionError("Invalid overlapping or missing tensor data.")
            position = stop
        tensors[file.name] = names
        match = re.search(r"-(\d+)-of-(\d+)\.safetensors$", file.name)
        if match:
            if not 1 <= int(match[1]) <= int(match[2]) <= 10000:
                raise SelectionError("Invalid shard count.")
            for shard in range(1, int(match[2]) + 1):
                sibling = re.sub(r"-\d+-of-", f"-{shard:0{len(match[1])}d}-of-", file.name)
                if not (directory / sibling).is_file():
                    raise SelectionError("Missing weight shard.")
    for index in directory.glob("*.safetensors.index.json"):
        mapping = _json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise SelectionError("Invalid shard index.")
        for tensor, filename in mapping.items():
            if tensor not in tensors.get(filename, set()):
                raise SelectionError("Missing indexed weight tensor or shard.")
    return set().union(*tensors.values())


def _image_weights(path, transformer, encoder):
    from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

    for component in Flux2KleinWeightDefinition.get_components():
        directory = path / component.hf_subdir
        _json(directory / "config.json")
        names = _weights(directory)
        for target in component.mapping_getter():
            if not target.required:
                continue
            for pattern in target.from_pattern:
                patterns = [pattern]
                if pattern.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                    field = ("num_single_layers" if pattern.startswith("single_") else "num_layers")
                    patterns = [pattern.replace("{block}", str(block))
                                for block in range(transformer[field])]
                elif pattern.startswith("model.layers."):
                    count = encoder.get("num_hidden_layers")
                    if type(count) is not int or not 1 <= count <= 128:
                        raise SelectionError("Invalid text encoder layer count.")
                    patterns = [pattern.replace("{layer}", str(layer)) for layer in range(count)]
                for required in patterns:
                    regex = "".join(
                        "[0-9]+" if part.startswith("{") else re.escape(part)
                        for part in re.split(r"(\{[^}]+\})", required)
                    )
                    if not any(re.fullmatch(regex, name) for name in names):
                        raise SelectionError(f"Missing required {component.name} weight: {required}")


def _tokenizer(directory):
    config = _json(directory / "tokenizer_config.json")
    if config.get("auto_map"):
        raise SelectionError("Tokenizers requiring remote code are not supported.")
    from tokenizers import Tokenizer

    try:
        Tokenizer.from_file(str(directory / "tokenizer.json"))
    except Exception as exc:
        raise SelectionError(f"Invalid or missing local tokenizer: {exc}") from exc


def prepare_selection(service, model):
    """Validate local candidate files and compatibility without downloading or loading.

    Args:
        service (str): chat, image_generate, image_edit or embedding.
        model (str): Cached HF repository, supported image alias or configured local path.

    Returns:
        dict: Selection with pinned local snapshot and local_files_checked readiness.

    Raises:
        SelectionError: If the service, identifier, files or architecture are unsupported.
    """
    if service not in SERVICES or not isinstance(model, str) or not model.strip() or len(model) > 1024:
        raise SelectionError("Specify a supported service and a nonempty model identifier.")
    model = model.strip()
    alias = None
    try:
        if service.startswith("image_"):
            command = (settings.images.generate_command if service == "image_generate"
                       else settings.images.edit_command)
            expected = ("mflux-generate-flux2" if service == "image_generate"
                        else "mflux-generate-flux2-edit")
            if command != expected:
                raise SelectionError("Runtime switching only supports the FLUX.2 Klein commands.")
            if not os.access(Path(sys.executable).parent / command, os.X_OK):
                raise SelectionError("Configured MFLUX command is not installed.")
            entry = image_registry().get(model)
            if entry is None:
                raise SelectionError("Unsupported model for the installed FLUX.2 Klein command.")
            alias, model = entry
        path = _snapshot(model)
        if service.startswith("image_"):
            config = _json(path / "transformer" / "config.json")
            if config.get("_class_name") != "Flux2Transformer2DModel":
                raise SelectionError("Image checkpoint is not a FLUX.2 transformer.")
            from mflux.models.common.config.model_config import AVAILABLE_MODELS

            registry_config = AVAILABLE_MODELS[alias]
            if any(config.get(key) != value
                   for key, value in registry_config.transformer_overrides.items()):
                raise SelectionError("Image transformer geometry does not match the CLI model.")
            encoder = _json(path / "text_encoder" / "config.json")
            if (encoder.get("architectures") != ["Qwen3ForCausalLM"]
                    or _json(path / "vae" / "config.json").get("_class_name") != "AutoencoderKLFlux2"):
                raise SelectionError("Image VAE/text encoder architecture is unsupported.")
            if any(encoder.get(key) != value
                   for key, value in registry_config.text_encoder_overrides.items()):
                raise SelectionError("Image text encoder geometry does not match the CLI model.")
            _image_weights(path, config, encoder)
            _tokenizer(path / "tokenizer")
        else:
            from model_management_server import model_metadata

            kind, _ = model_metadata(path)
            if kind != service:
                raise SelectionError(f"Model metadata is not compatible with {service}.")
            config = _json(path / "config.json")
            if config.get("auto_map"):
                raise SelectionError("Models requiring remote code are not supported.")
            _weights(path)
            _tokenizer(path)
            if service == "chat":
                from mlx_lm.utils import _get_classes

                _get_classes(config)
            elif (path / "modules.json").is_file():
                modules = json.loads((path / "modules.json").read_text(encoding="utf-8"))
                if not isinstance(modules, list) or not modules:
                    raise SelectionError("Invalid sentence-transformers module metadata.")
                for module in modules:
                    relative = module.get("path")
                    if not isinstance(relative, str) or ".." in Path(relative).parts:
                        raise SelectionError("Invalid embedding module path.")
                    module_path = path / relative
                    if not module_path.is_relative_to(path) or not module_path.is_dir():
                        raise SelectionError("Missing embedding module.")
        return {"model": model, "path": str(path), "alias": alias,
                "validation": "local_files_checked"}
    except SelectionError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        raise SelectionError(f"Candidate validation failed: {exc}") from exc


def switch_model(service, model):
    """Validate a candidate and atomically publish it only on success.

    Args:
        service (str): Service to switch.
        model (str): Downloaded candidate identifier.

    Returns:
        dict: Public active selections after successful commit.

    Raises:
        SelectionError: If local validation or the chat backend load probe fails.
        OSError: If shared state cannot be committed; the previous state is retained.
    """
    with SWITCH_LOCK:
        candidate = prepare_selection(service, model)
        if service == "chat":
            try:
                with chat_request({"messages": [{"role": "user", "content": "Hi"}],
                                   "max_tokens": 1, "stream": False}, candidate) as response:
                    result = json.loads(response.read())
                if not isinstance(result, dict) or not result.get("choices") or result.get("error"):
                    raise SelectionError("Chat backend did not return a completion.")
            except (OSError, ValueError) as exc:
                if isinstance(exc, HTTPError):
                    exc.close()
                raise SelectionError(f"Chat backend could not load the candidate: {exc}") from exc
            candidate["validation"] = "backend_loaded"
        with _state_lock():
            state = read_selections()
            state[service] = candidate
            _write_state(state)
        return active_models()


if __name__ == "__main__":
    reset_active_models()
