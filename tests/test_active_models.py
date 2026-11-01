"""Stdlib regressions: no GPU model loads, downloads or real inference processes."""

import ast
import base64
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import unittest
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import active_models as active
import chat_server
import image_server
from api_docs import openapi_spec
from chat_server import ChatHandler
from configuration import browser_settings
from model_management_server import configured_models

ORIGINAL_SNAPSHOT = active._snapshot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_weights(path, tensor="weight"):
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [tensor] if isinstance(tensor, str) else sorted(tensor)
    header = json.dumps({name: {"dtype": "F32", "shape": [1],
                               "data_offsets": [index * 4, (index + 1) * 4]}
                         for index, name in enumerate(names)}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * (4 * len(names)))


def write_tokenizer(directory):
    write_json(directory / "tokenizer_config.json", {})
    write_json(directory / "tokenizer.json", {
        "model": {"type": "WordLevel", "vocab": {"[UNK]": 0}, "unk_token": "[UNK]"},
    })


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.workspace = ROOT / (".test-runtime-" + uuid4().hex)
        self.workspace.mkdir()
        self.addCleanup(shutil.rmtree, self.workspace)
        state = patch.object(active, "STATE_FILE", self.workspace / "state" / "active.json")
        state.start()
        self.addCleanup(state.stop)
        active.reset_active_models()
        self.paths = {}
        for name in ("old/chat", "new/chat"):
            directory = self.workspace / name
            write_json(directory / "config.json",
                       {"architectures": ["LlamaForCausalLM"], "model_type": "llama"})
            write_tokenizer(directory)
            write_weights(directory / "model.safetensors")
            self.paths[name] = directory
        embedding = self.workspace / "embedding"
        write_json(embedding / "config.json",
                   {"architectures": ["BertModel"], "model_type": "bert"})
        write_tokenizer(embedding)
        write_json(embedding / "modules.json",
                   [{"path": "", "type": "sentence_transformers.models.Transformer"},
                    {"path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}])
        write_json(embedding / "1_Pooling" / "config.json", {})
        write_weights(embedding / "model.safetensors")
        self.paths["test/embedding"] = embedding
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

        for alias in ("flux2-klein-4b", "flux2-klein-9b"):
            config = AVAILABLE_MODELS[alias]
            directory = self.workspace / alias
            write_json(directory / "transformer" / "config.json",
                       {"_class_name": "Flux2Transformer2DModel",
                        **config.transformer_overrides})
            write_json(directory / "text_encoder" / "config.json",
                       {"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 36,
                        **config.text_encoder_overrides})
            write_json(directory / "vae" / "config.json", {"_class_name": "AutoencoderKLFlux2"})
            for component in Flux2KleinWeightDefinition.get_components():
                names = set()
                for target in component.mapping_getter():
                    if not target.required:
                        continue
                    for pattern in target.from_pattern:
                        patterns = [pattern]
                        if pattern.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                            count = config.transformer_overrides[
                                "num_single_layers" if pattern.startswith("single_") else "num_layers"]
                            patterns = [pattern.replace("{block}", str(block)) for block in range(count)]
                        elif pattern.startswith("model.layers."):
                            patterns = [pattern.replace("{layer}", str(layer)) for layer in range(36)]
                        names.update(re.sub(r"\{[^}]+\}", "0", pattern) for pattern in patterns)
                write_weights(directory / component.hf_subdir / "model.safetensors", names)
            write_tokenizer(directory / "tokenizer")
            self.paths[config.model_name] = directory
        snapshot = patch.object(active, "_snapshot", side_effect=self.snapshot)
        snapshot.start()
        self.addCleanup(snapshot.stop)
        self.backend_requests = []
        self.block_model = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail_model = None
        test = self

        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.backend_requests.append(data)
                if data["model"] == test.block_model and data.get("max_tokens") != 1:
                    test.entered.set()
                    test.release.wait(5)
                if data["model"] == test.fail_model:
                    self.send_response(500)
                    body = b'{"error":"Cannot load weights"}'
                elif data.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    body = b'data: {"text":"hello"}\n\ndata: [DONE]\n\n'
                else:
                    self.send_response(200)
                    body = json.dumps({"choices": [{"message": {"content": "ok"}}],
                                       "model": data["model"]}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.backend = self.server(Backend)
        backend_url = patch.object(
            active, "backend_url",
            return_value=f"http://127.0.0.1:{self.backend.server_port}/v1/chat/completions")
        backend_url.start()
        self.addCleanup(backend_url.stop)
        self.web = self.server(partial(ChatHandler, directory=str(ROOT / "web")))

    def snapshot(self, model):
        if model not in self.paths:
            raise active.SelectionError("Model is not downloaded.")
        return self.paths[model]

    def server(self, handler, server_class=ThreadingHTTPServer):
        server = server_class(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def request(self, path, data=None, method=None, server=None):
        server = server or self.web
        body = None if data is None else json.dumps(data).encode()
        request = Request(f"http://127.0.0.1:{server.server_port}{path}", body,
                          {"Content-Type": "application/json"}, method=method)
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "image/" in content_type or "text/event-stream" in content_type:
                return response.status, content
            return response.status, json.loads(content)

    def test_http_chat_switch_proxy_and_failure_preservation(self):
        defaults = configured_models()
        self.assertEqual(self.request("/models/configured"), (200, defaults))
        status, state = self.request("/models/active", {"service": "chat", "model": "new/chat"}, "PUT")
        self.assertEqual(status, 200)
        self.assertEqual(state["chat"], "new/chat")
        self.assertEqual(state["validation"]["chat"], "backend_loaded")
        probe = self.backend_requests[-1]
        self.assertEqual(probe["max_tokens"], 1)
        self.assertEqual(probe["model"], str(self.paths["new/chat"]))
        status, result = self.request(active.CHAT_ROUTE, {
            "model": "/arbitrary/path", "adapters": "/arbitrary/adapter",
            "draft_model": "remote/model", "messages": [], "max_tokens": 5,
        })
        self.assertEqual(status, 200)
        self.assertEqual(result["model"], str(self.paths["new/chat"]))
        self.assertIsNone(self.backend_requests[-1]["adapters"])
        self.assertIsNone(self.backend_requests[-1]["draft_model"])
        self.fail_model = str(self.paths["old/chat"])
        self.assertEqual(self.request("/models/active",
                                     {"service": "chat", "model": "old/chat"}, "PUT")[0], 422)
        self.assertEqual(active.active_models(), state)
        self.assertEqual(self.request("/models/configured"), (200, defaults))
        self.assertEqual(browser_settings()["chat"]["api_url"], active.CHAT_ROUTE)
        self.assertNotIn("model", browser_settings()["chat"])

    def test_inflight_chat_capture_and_streaming(self):
        active.switch_model("chat", "old/chat")
        self.block_model = str(self.paths["old/chat"])
        result = []
        thread = threading.Thread(target=lambda: result.append(
            self.request(active.CHAT_ROUTE, {"messages": [], "max_tokens": 5})))
        thread.start()
        try:
            self.assertTrue(self.entered.wait(5))
            self.assertEqual(self.request("/models/active",
                                         {"service": "chat", "model": "new/chat"}, "PUT")[0], 200)
        finally:
            self.release.set()
            thread.join(5)
        self.assertEqual(result[0][1]["model"], str(self.paths["old/chat"]))
        status, stream = self.request(active.CHAT_ROUTE, {"messages": [], "stream": True})
        self.assertEqual(status, 200)
        self.assertIn(b"data: [DONE]", stream)
        self.assertEqual(self.backend_requests[-1]["model"], str(self.paths["new/chat"]))

    def test_embedding_is_selection_only_and_state_is_cross_process(self):
        status, state = self.request("/models/active",
                                     {"service": "embedding", "model": "test/embedding"}, "PUT")
        self.assertEqual(status, 200)
        self.assertFalse(state["embedding_inference_available"])
        self.assertEqual(state["validation"]["embedding"], "local_files_checked")
        self.assertEqual(self.backend_requests, [])
        code = ("import sys,json; from pathlib import Path; sys.path.insert(0,'src'); "
                "import active_models as a; a.STATE_FILE=Path(sys.argv[1]); "
                "print(json.dumps(a.active_models()))")
        result = subprocess.run([sys.executable, "-c", code, str(active.STATE_FILE)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), state)
        reset = code.replace("print(json.dumps(a.active_models()))", "a.reset_active_models()")
        subprocess.run([sys.executable, "-c", reset, str(active.STATE_FILE)],
                       cwd=ROOT, capture_output=True, check=True)
        self.assertEqual(active.read_selections(), active.default_selections())

    def test_bad_inputs_unknown_and_incompatible_candidates(self):
        original = active.active_models()
        for data, expected in (
            ([], 400), ({}, 400),
            ({"service": "unknown", "model": "new/chat"}, 422),
            ({"service": "chat", "model": 42}, 422),
            ({"service": "chat", "model": "missing/model"}, 422),
            ({"service": "embedding", "model": "new/chat"}, 422),
            ({"service": "image_generate", "model": "new/chat"}, 422),
            ({"service": "chat", "model": "test/embedding"}, 422),
        ):
            self.assertEqual(self.request("/models/active", data, "PUT")[0], expected)
            self.assertEqual(active.active_models(), original)
        self.assertEqual(self.request("/wrong", {}, "PUT")[0], 404)
        self.assertEqual(self.request(active.CHAT_ROUTE, [], "POST")[0], 400)
        self.assertEqual(self.request("/models/active", {"service": "chat", "model": "x" * 5000},
                                     "PUT")[0], 400)

    def test_shards_truncation_remote_code_and_image_geometry(self):
        directory = self.paths["new/chat"]
        write_json(directory / "model.safetensors.index.json",
                   {"weight_map": {"weight": "missing.safetensors"}})
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("chat", "new/chat")
        (directory / "model.safetensors.index.json").unlink()
        (directory / "model.safetensors").write_bytes(b"incomplete")
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("chat", "new/chat")
        write_weights(directory / "model.safetensors")
        write_json(directory / "config.json",
                   {"architectures": ["UnknownForCausalLM"], "model_type": "not_supported"})
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("chat", "new/chat")
        write_json(directory / "config.json",
                   {"architectures": ["LlamaForCausalLM"], "model_type": "llama",
                    "auto_map": {"AutoModel": "remote.code"}})
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("chat", "new/chat")
        image = self.paths["black-forest-labs/FLUX.2-klein-4B"]
        (image / "text_encoder" / "model.safetensors").unlink()
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("image_generate", "flux2-klein-4b")
        write_weights(image / "text_encoder" / "model.safetensors")
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("image_generate", "flux2-klein-4b")
        write_json(image / "transformer" / "config.json",
                   {"_class_name": "Flux2Transformer2DModel", "num_layers": 99})
        with self.assertRaises(active.SelectionError):
            active.prepare_selection("image_generate", "flux2-klein-4b")

    def test_undownloaded_catalog_and_invalid_tokenizer(self):
        with patch.object(active, "HF_HUB_CACHE", str(self.workspace / "no-cache")):
            with self.assertRaises(active.SelectionError):
                ORIGINAL_SNAPSHOT("not-downloaded/model")
        write_json(self.paths["new/chat"] / "tokenizer.json", {})
        before = active.active_models()
        self.assertEqual(self.request("/models/active", {
            "service": "chat", "model": "new/chat",
        }, "PUT")[0], 422)
        self.assertEqual(active.active_models(), before)
        self.assertEqual(self.backend_requests, [])

    def test_corrupt_state_returns_errors_without_reset(self):
        active.STATE_FILE.write_text("[]")
        with self.assertLogs("model_management_server", level="ERROR"):
            self.assertEqual(self.request("/models/active")[0], 500)
            self.assertEqual(self.request("/models/active", {
                "service": "embedding", "model": "test/embedding",
            }, "PUT")[0], 500)
        self.assertEqual(active.STATE_FILE.read_text(), "[]")
        active.reset_active_models()
        self.assertEqual(self.request("/models/active")[0], 200)

    def test_web_startup_resets_defaults_before_listening(self):
        active.switch_model("embedding", "test/embedding")
        with patch.object(chat_server, "ThreadingHTTPServer") as listener:
            chat_server.main()
        listener.return_value.serve_forever.assert_called_once()
        self.assertEqual(active.read_selections(), active.default_selections())

    def test_image_http_reads_shared_state_and_captures_inflight(self):
        from mflux.models.common.config.model_config import AVAILABLE_MODELS
        from mflux.models.common.resolution.config_resolution import ConfigResolution
        from mflux.models.flux2.cli import flux2_edit_generate, flux2_generate

        data_dir = self.workspace / "images"
        for name in ("inputs", "outputs"):
            (data_dir / name).mkdir(parents=True)
        patches = [
            patch.object(image_server, "DATA_DIR", data_dir),
            patch.object(image_server, "INPUT_DIR", data_dir / "inputs"),
            patch.object(image_server, "OUTPUT_DIR", data_dir / "outputs"),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.assertEqual(self.request("/models/active",
                                     {"service": "image_generate", "model": "flux2-klein-4b"},
                                     "PUT")[0], 200)
        commands = []
        parsed_commands = []
        image = self.server(image_server.ImageHandler, HTTPServer)

        def run(command, **kwargs):
            commands.append((command, kwargs))
            cli = flux2_generate if command[0] == str(image_server.GENERATE_CMD) else flux2_edit_generate
            # Exercise installed CLI normalization, but never construct a GPU model.
            with patch.object(sys, "argv", command):
                args = cli.build_parser().parse_args()
            config = ConfigResolution.resolve_restricted(
                args.model, cli.DEFAULT_MODEL, model_path=args.model_path,
                extra_keys=cli.FAMILY_MODELS, base_model=args.base_model,
            )
            parsed_commands.append((args, config))
            if len(commands) == 1:
                # Publish a switch while the first subprocess is already running.
                self.assertEqual(self.request("/models/active", {
                    "service": "image_generate", "model": "black-forest-labs/FLUX.2-klein-9B",
                }, "PUT")[0], 200)
            Path(command[command.index("--output") + 1]).write_bytes(b"fake PNG")
            return SimpleNamespace(returncode=0, stderr="", stdout="")

        with patch.object(image_server.subprocess, "run", side_effect=run):
            for _ in range(2):
                self.assertEqual(self.request(image_server.GENERATE_ROUTE, {"prompt": "test"},
                                              server=image), (200, b"fake PNG"))
            self.assertEqual(self.request("/models/active", {
                "service": "image_edit", "model": "flux2-klein-9b",
            }, "PUT")[0], 200)
            self.assertEqual(self.request(image_server.EDIT_ROUTE, {
                "prompt": "edit", "image": base64.b64encode(b"image").decode(),
            }, server=image), (200, b"fake PNG"))
            active.reset_active_models()
            self.assertEqual(self.request(image_server.GENERATE_ROUTE, {"prompt": "default"},
                                          server=image), (200, b"fake PNG"))
            self.assertEqual(self.request(image_server.EDIT_ROUTE, {
                "prompt": "default edit", "image": base64.b64encode(b"image").decode(),
            }, server=image), (200, b"fake PNG"))
        for index, alias in enumerate(("flux2-klein-4b", "flux2-klein-9b", "flux2-klein-9b")):
            command = commands[index][0]
            args, config = parsed_commands[index]
            path = str(self.paths[AVAILABLE_MODELS[alias].model_name])
            self.assertEqual(command[command.index("--model") + 1], path)
            self.assertEqual(command[command.index("--base-model") + 1], alias)
            self.assertEqual(args.model_path, path)
            self.assertEqual(args.base_model, alias)
            self.assertIs(config, AVAILABLE_MODELS[alias])
        for index, service in ((3, "image_generate"), (4, "image_edit")):
            selection = active.default_selections()[service]
            args, config = parsed_commands[index]
            self.assertEqual(args.model, selection["alias"] or selection["model"])
            self.assertIsNone(args.model_path)
            self.assertIsNone(args.base_model)
            self.assertNotIn("--base-model", commands[index][0])
            self.assertIs(config, AVAILABLE_MODELS[selection["alias"] or selection["model"]])
        for command, kwargs in commands:
            self.assertNotIn("--model-path", command)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(self.request("/health", server=image)[0], 200)
        self.assertEqual(list((data_dir / "requests").iterdir()), [])

    def test_atomic_updates_and_write_failure(self):
        before = active.active_models()
        with patch.object(active.os, "replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                active.switch_model("embedding", "test/embedding")
        self.assertEqual(active.active_models(), before)
        self.assertFalse(active.STATE_FILE.with_suffix(".pending").exists())
        errors = []

        def update(service, model):
            try:
                active.switch_model(service, model)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=update, args=args) for args in (
            ("embedding", "test/embedding"), ("image_generate", "flux2-klein-4b"),
            ("image_edit", "flux2-klein-9b"),
        )]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(active.active_models()["embedding"], "test/embedding")
        self.assertEqual(active.active_models()["image_edit"], "black-forest-labs/FLUX.2-klein-9B")

    def test_docs_and_public_docstrings(self):
        status, spec = self.request("/openapi.json")
        self.assertEqual(status, 200)
        self.assertIn("put", spec["paths"]["/models/active"])
        request_schema = spec["paths"]["/models/active"]["put"]["requestBody"][
            "content"]["application/json"]["schema"]
        self.assertEqual(request_schema["properties"]["service"]["enum"], list(active.SERVICES))
        self.assertEqual(spec["paths"][active.CHAT_ROUTE]["post"]["servers"], [{"url": "/"}])
        self.assertIn("ActiveModels", spec["components"]["schemas"])
        self.assertIn("image/png", spec["paths"][image_server.GENERATE_ROUTE][
            "post"]["responses"]["200"]["content"])
        for path in (ROOT / "src").glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
                    doc = ast.get_docstring(node) or ""
                    self.assertIn("Args:", doc, f"{path.name}:{node.name}")
                    self.assertIn("Returns:", doc, f"{path.name}:{node.name}")


if __name__ == "__main__":
    unittest.main()
