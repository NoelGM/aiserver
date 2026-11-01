import json
from urllib.parse import urlparse

from api_docs import openapi_spec
from configuration import ROOT, settings


def model_catalog():
    models = [{
        "id": settings.chat.model,
        "object": "model",
        "type": "chat",
        "capabilities": ["chat"],
    }]
    image_models = {}
    for model, capability in (
        (settings.images.generate_model, "generate"),
        (settings.images.edit_model, "edit"),
    ):
        if model not in image_models:
            image_models[model] = {
                "id": model,
                "object": "model",
                "type": "image",
                "capabilities": [],
            }
        image_models[model]["capabilities"].append(capability)
    models.extend(image_models.values())
    return {"object": "list", "data": models}


class CommonAPIMixin:
    def send_json(self, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_document(body, "application/json; charset=utf-8")

    def send_document(self, body, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/v1/models":
            self.send_json(model_catalog())
        elif route == "/openapi.json":
            self.send_json(openapi_spec(model_catalog()))
        elif route in ("/docs", "/docs/"):
            self.send_document(
                (ROOT / "web" / "swagger.html").read_bytes(),
                "text/html; charset=utf-8",
            )
        else:
            super().do_GET()
