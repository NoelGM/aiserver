from urllib.parse import parse_qsl, urlparse

from configuration import service_url, settings
from active_models import CHAT_ROUTE, SERVICES


def endpoint(template, service):
    """Describe a configured endpoint for OpenAPI.

    Args:
        template (str): API URL with optional hostname and port placeholders.
        service: Service settings containing the port used by the template.

    Returns:
        tuple: URL path, OpenAPI server definition and query parameter list.

    Raises:
        ValueError: If the expanded URL is invalid.
    """
    parsed = urlparse(service_url(template, service, "{hostname}"))
    server = {"url": f"{parsed.scheme}://{parsed.netloc}"}
    if "{hostname}" in server["url"]:
        server["variables"] = {"hostname": {"default": "localhost"}}
    parameters = [
        {
            "name": name,
            "in": "query",
            "schema": {"type": "string", "default": value},
        }
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    return parsed.path, server, parameters


def json_response(description, schema):
    """Build an OpenAPI response definition for JSON content.

    Args:
        description (str): Human-readable description of the response.
        schema (dict): OpenAPI schema describing the JSON response body.

    Returns:
        dict: Response definition with its description and JSON content schema.
    """
    return {
        "description": description,
        "content": {"application/json": {"schema": schema}},
    }


def openapi_spec(catalog, configured):
    """Build the OpenAPI specification for the model, chat and image APIs.

    Args:
        catalog (dict): Local model catalog used as the list response example.
        configured (dict): Configured models used as the configuration example.

    Returns:
        dict: OpenAPI 3.0.3 document using current service URLs and defaults.
    """
    error = json_response("Request or backend error.", {
        "type": "object",
        "properties": {"error": {"type": "string"}},
    })
    paths = {
        "/models/list": {"get": {
            "tags": ["Models"],
            "operationId": "listModels",
            "summary": "List locally downloaded models",
            "description": (
                "Scans the Hugging Face cache (HF_HUB_CACHE/HF_HOME) and configured local "
                "model directories. Cache entries without weight files are excluded. "
                "Types and capabilities are inferred from metadata, not backend compatibility. "
                "Downloads may be partial; this does not verify completeness or service readiness. "
                "Unrecognized models use type unknown; cache issues are returned in warnings."
            ),
            "servers": [{"url": "/"}],
            "responses": {"200": {
                "description": "Local models.",
                "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/ModelCatalog"},
                    "example": catalog,
                }},
            }, "500": error},
        }},
        "/models/configured": {"get": {
            "tags": ["Models"],
            "operationId": "configuredModels",
            "summary": "Show configured models by service",
            "description": "Startup chat, embedding, image generation and image editing defaults; "
                           "does not check downloads or backend readiness.",
            "servers": [{"url": "/"}],
            "responses": {"200": {
                "description": "Models configured for each service.",
                "content": {"application/json": {
                    "schema": {
                        "type": "object",
                        "required": ["chat", "embedding", "images"],
                        "properties": {
                            "chat": {"type": "string"},
                            "embedding": {"type": "string"},
                            "images": {
                                "type": "object",
                                "required": ["generate", "edit"],
                                "properties": {
                                    "generate": {"type": "string"},
                                    "edit": {"type": "string"},
                                },
                            },
                        },
                    },
                    "example": configured,
                }},
            }},
        }},
    }
    active_schema = {
        "type": "object",
        "required": [*SERVICES, "validation", "embedding_inference_available"],
        "properties": {
            **{service: {"type": "string"} for service in SERVICES},
            "embedding_inference_available": {"type": "boolean", "enum": [False]},
            "validation": {
                "type": "object",
                "required": list(SERVICES),
                "properties": {service: {"type": "string", "enum": [
                    "startup_default", "local_files_checked", "backend_loaded",
                ]} for service in SERVICES},
            },
        },
    }
    active_response = json_response("Current runtime selections.", {
        "$ref": "#/components/schemas/ActiveModels",
    })
    paths["/models/active"] = {
        "get": {
            "tags": ["Models"],
            "operationId": "activeModels",
            "summary": "Show active runtime selections",
            "description": (
                "Shared by the web/chat proxy and image process. Startup defaults are not "
                "validated. Embedding is selection metadata only; no embedding inference API exists. "
                "backend_loaded means a chat load probe succeeded, not that the model stays resident."
            ),
            "responses": {"200": active_response, "500": error},
        },
        "put": {
            "tags": ["Models"],
            "operationId": "switchActiveModel",
            "summary": "Switch a service to a downloaded model",
            "description": (
                "Accepts cached HF repository IDs, configured local chat/embedding directories, "
                "or installed FLUX.2 Klein aliases/canonical repositories for images. "
                "Other image commands, custom image checkpoints and remote-code models are rejected. "
                "Pins the local snapshot; never downloads or edits YAML. Complete safetensors "
                "headers, shard indexes, tokenizers and architecture metadata are checked; "
                "images also require the installed MFLUX weight mappings for every transformer/encoder layer. "
                "Chat also performs a one-token backend load probe (may take time and GPU memory); "
                "image/embedding checks do not load GPU weights or prove numerical correctness. "
                "A failed validation leaves selections unchanged. Subsequent proxy/image requests "
                "capture the new selection; in-flight requests retain theirs. Changes reset at "
                "application startup. The existing LAN API is unauthenticated; restrict access "
                "to trusted clients. Direct MLX-LM requests bypass runtime defaults."
            ),
            "requestBody": {
                "required": True,
                "content": {"application/json": {
                    "schema": {"type": "object", "additionalProperties": False,
                               "required": ["service", "model"], "properties": {
                                   "service": {"type": "string", "enum": list(SERVICES)},
                                   "model": {"type": "string", "minLength": 1, "maxLength": 1024},
                               }},
                    "example": {"service": "image_generate", "model": "flux2-klein-4b"},
                }},
            },
            "responses": {"200": active_response, "400": error, "422": error, "500": error},
        },
    }
    model_schema = {
        "type": "object",
        "required": ["object", "data"],
        "properties": {
            "object": {"type": "string", "enum": ["list"]},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "data": {"type": "array", "items": {
                "type": "object",
                "required": ["id", "object", "type", "capabilities"],
                "properties": {
                    "id": {"type": "string"},
                    "object": {"type": "string", "enum": ["model"]},
                    "type": {"type": "string", "enum": ["chat", "image", "embedding", "unknown"]},
                    "source": {"type": "string", "enum": ["huggingface_cache", "local"]},
                    "capabilities": {"type": "array", "items": {
                        "type": "string", "enum": ["chat", "generate", "edit", "embedding"],
                    }},
                },
            }},
        },
    }
    chat_path, chat_server, _ = endpoint(settings.chat.api_url, settings.chat)
    paths.setdefault(CHAT_ROUTE, {})["post"] = {
        "tags": ["Chat"],
        "operationId": "chatCompletion",
        "summary": "Generate a chat response",
        "description": (
            "Web proxy captures the active chat selection per request and overrides model, "
            "adapters and draft_model. Disable streaming for JSON. For direct MLX-LM access "
            f"use its configured endpoint {chat_path}; direct requests do not use runtime defaults."
        ),
        "servers": [{"url": "/"}],
        "requestBody": {
            "required": True,
            "content": {"application/json": {
                "schema": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "model": {"type": "string",
                                  "description": "Optional; ignored by the active-model proxy."},
                        "messages": {"type": "array", "minItems": 1, "items": {
                            "type": "object",
                            "required": ["role", "content"],
                            "properties": {
                                "role": {"type": "string", "enum": [
                                    "system", "user", "assistant", "tool",
                                ]},
                                "content": {"type": "string"},
                            },
                        }},
                        "max_tokens": {"type": "integer", "minimum": 1,
                                       "default": settings.chat.max_tokens},
                        "temperature": {"type": "number", "minimum": 0},
                        "stream": {"type": "boolean", "default": False},
                    },
                },
                "example": {
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "max_tokens": settings.chat.max_tokens,
                    "stream": False,
                },
            }},
        },
        "responses": {
            "200": {
                "description": "OpenAI-compatible completion or server-sent events.",
                "content": {
                    "application/json": {"schema": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "object": {"type": "string"},
                            "model": {"type": "string"},
                            "choices": {"type": "array", "items": {"type": "object"}},
                            "usage": {"type": "object"},
                        },
                    }},
                    "text/event-stream": {"schema": {"type": "string"}},
                },
            },
            "400": error,
            "500": error,
            "502": error,
        },
    }
    for template, operation_id, summary, editing in (
        (settings.images.generate_url, "generateImage", "Generate an image", False),
        (settings.images.edit_url, "editImage", "Edit an image", True),
    ):
        path, server, parameters = endpoint(template, settings.images)
        properties = {
            "prompt": {"type": "string", "minLength": 1},
            "steps": {"type": "integer", "minimum": 1, "maximum": 50,
                      "default": settings.images.steps},
            "seed": {"type": "integer"},
        }
        example = {"prompt": "A landscape at sunset", "steps": settings.images.steps}
        required = ["prompt"]
        if editing:
            properties["image"] = {
                "type": "string",
                "description": "Raw base64 image or data URL. Replace the example with your image.",
            }
            example["image"] = "data:image/png;base64,REPLACE_WITH_BASE64_IMAGE"
            required.append("image")
        else:
            for dimension in ("width", "height"):
                properties[dimension] = {
                    "type": "integer",
                    "minimum": settings.images.min_dimension,
                    "maximum": settings.images.max_dimension,
                    "multipleOf": 16,
                    "default": settings.images[dimension],
                }
                example[dimension] = settings.images[dimension]
        paths.setdefault(path, {})["post"] = {
            "tags": ["Images"],
            "operationId": operation_id,
            "summary": summary,
            "description": (
                "Captures the shared active image selection when each request starts. "
                f"Returns PNG bytes. Backend timeout: {settings.images.timeout} seconds."
            ),
            "servers": [server],
            "parameters": parameters,
            "requestBody": {
                "required": True,
                "content": {"application/json": {
                    "schema": {"type": "object", "required": required, "properties": properties},
                    "example": example,
                }},
            },
            "responses": {
                "200": {"description": "Generated or edited image.",
                        "content": {"image/png": {"schema": {
                            "type": "string", "format": "binary",
                        }}}},
                "500": error,
                "504": error,
            },
        }
    paths.setdefault("/health", {})["get"] = {
        "tags": ["Health"],
        "operationId": "serviceHealth",
        "summary": "Check a service",
        "description": "Choose the interface, chat backend or image backend server.",
        "servers": [
            {"url": "/", "description": "Chat interface and model management API"},
            {**chat_server, "description": "Chat backend"},
            endpoint("http://{hostname}:{port}/health", settings.images)[1],
        ],
        "responses": {"200": json_response("Service is responding.", {"type": "object"})},
    }
    return {
        "openapi": "3.0.3",
        "info": {"title": "AI Server API", "version": "1.0.0",
                 "description": "Chat, image generation/editing and the shared model catalog."},
        "servers": [{"url": "/"}],
        "tags": [{"name": name} for name in ("Models", "Chat", "Images", "Health")],
        "paths": paths,
        "components": {"schemas": {"ModelCatalog": model_schema, "ActiveModels": active_schema}},
    }
