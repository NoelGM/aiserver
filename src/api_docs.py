from urllib.parse import parse_qsl, urlparse

from configuration import service_url, settings


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
            "description": "Current chat, embedding, image generation and image editing configuration; "
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
    chat_path, chat_server, chat_parameters = endpoint(settings.chat.api_url, settings.chat)
    paths.setdefault(chat_path, {})["post"] = {
        "tags": ["Chat"],
        "operationId": "chatCompletion",
        "summary": "Generate a chat response",
        "description": "Uses the MLX-LM backend. Disable streaming for a JSON response.",
        "servers": [chat_server],
        "parameters": chat_parameters,
        "requestBody": {
            "required": True,
            "content": {"application/json": {
                "schema": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "model": {"type": "string", "default": settings.chat.model},
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
                    "model": settings.chat.model,
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
            "description": f"Returns PNG bytes. Backend timeout: {settings.images.timeout} seconds.",
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
        "components": {"schemas": {"ModelCatalog": model_schema}},
    }
