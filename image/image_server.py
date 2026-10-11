import base64
import json
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

HOST = "0.0.0.0"
PORT = 8090

BIN_DIR = Path(sys.executable).parent
GENERATE_CMD = BIN_DIR / "mflux-generate-flux2"
EDIT_CMD = BIN_DIR / "mflux-generate-flux2-edit"

class ImageHandler(BaseHTTPRequestHandler):

    def send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_cors_headers()
        self.end_headers()

    def send_json(self, status, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        route = urlparse(self.path).path
        if route not in ("/generate", "/edit"):
            self.send_json(404, {"error": "Unknown endpoint"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 30 * 1024 * 1024:
                raise ValueError("Invalid or excessively large request.")

            data = json.loads(self.rfile.read(length))
            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Please enter a prompt.")

            steps = max(1, min(int(data.get("steps", 4)), 50))

            with tempfile.TemporaryDirectory(prefix="mflux_") as temp:
                workdir = Path(temp)
                output_path = workdir / "output.png"

                if route == "/generate":
                    width = max(256, min(int(data.get("width", 768)), 1536))
                    height = max(256, min(int(data.get("height", 768)), 1536))

                    # FLUX image dimensions should be multiples of 16.
                    width = max(256, round(width / 16) * 16)
                    height = max(256, round(height / 16) * 16)

                    command = [
                        str(GENERATE_CMD),
                        "--model", "flux2-klein-4b",
                        "--quantize", "4",
                        "--prompt", prompt,
                        "--width", str(width),
                        "--height", str(height),
                        "--steps", str(steps),
                        "--output", str(output_path),
                        "--low-ram",
                    ]
                else:
                    image_data = data.get("image", "")
                    if not image_data:
                        raise ValueError("Please select an image to edit.")

                    # Accept a base64 data URL or raw base64 string.
                    if "," in image_data and image_data.startswith("data:"):
                        image_data = image_data.split(",", 1)[1]

                    input_path = workdir / "input.png"
                    input_path.write_bytes(base64.b64decode(image_data))

                    command = [
                        str(EDIT_CMD),
                        "--model", "flux2-klein-4b",
                        "--quantize", "4",
                        "--image-paths", str(input_path),
                        "--prompt", prompt,
                        "--steps", str(steps),
                        "--output", str(output_path),
                        "--low-ram",
                    ]

                seed = data.get("seed")
                if seed not in (None, ""):
                    command.extend(["--seed", str(int(seed))])

                print(f"Running MFLUX {'generation' if route == '/generate' else 'editing'}...")
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )

                if result.returncode != 0:
                    details = (result.stderr or result.stdout or "Unknown MFLUX error")
                    print(details[-5000:])
                    self.send_json(500, {"error": details[-2500:]})
                    return

                # MFLUX may add a suffix to the output filename.
                if not output_path.exists():
                    candidates = list(workdir.glob("output*.png"))
                    if candidates:
                        output_path = candidates[0]

                if not output_path.exists():
                    self.send_json(500, {
                        "error": "MFLUX finished but no output image was found."
                    })
                    return

                image_bytes = output_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(image_bytes)))
                self.send_cors_headers()
                self.end_headers()
                self.wfile.write(image_bytes)
                print("Image returned successfully.")

        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "Image generation timed out after 10 minutes."})
        except Exception as exc:
            print(f"Request error: {exc}")
            self.send_json(500, {"error": str(exc)})

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))

if not GENERATE_CMD.exists() or not EDIT_CMD.exists():
    raise SystemExit("MFLUX commands not found. Activate the image .venv first.")

print(f"Starting image server on {HOST}:{PORT}")
print(f"Generate command: {GENERATE_CMD}")
print(f"Edit command: {EDIT_CMD}")

HTTPServer((HOST, PORT), ImageHandler).serve_forever()
