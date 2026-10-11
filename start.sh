#!/bin/zsh

# Local AI environment startup script

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

ROOT="$HOME/LLM"
WEB_DIR="$ROOT/web"
IMAGE_DIR="$ROOT/image"
IMAGE_PYTHON="$IMAGE_DIR/.venv/bin/python"
MODEL="mlx-community/gpt-oss-20b-OptiQ-4bit"

MLX_LOG="$ROOT/mlx.log"
WEB_LOG="$ROOT/web.log"
IMAGE_LOG="$ROOT/image.log"

mkdir -p "$ROOT"

echo "======================================"
echo " Starting local AI environment"
echo "======================================"

# Detect the Mac's current LAN IP address
IP=""
for INTERFACE in en0 en1 en2; do
    IP=$(ipconfig getifaddr "$INTERFACE" 2>/dev/null)
    if [[ -n "$IP" ]]; then
        break
    fi
done

if [[ -z "$IP" ]]; then
    IP="localhost"
fi

# Check required files and commands
if ! command -v mlx_lm.server >/dev/null 2>&1; then
    echo "ERROR: mlx_lm.server was not found in PATH."
    echo "Open your usual terminal environment and check MLX installation."
    read -k 1 "?Press any key to exit..."
    echo
    exit 1
fi

if [[ ! -x "$IMAGE_PYTHON" ]]; then
    echo "ERROR: Image virtual environment not found:"
    echo "$IMAGE_PYTHON"
    read -k 1 "?Press any key to exit..."
    echo
    exit 1
fi

if [[ ! -f "$IMAGE_DIR/image_server.py" ]]; then
    echo "ERROR: $IMAGE_DIR/image_server.py was not found."
    read -k 1 "?Press any key to exit..."
    echo
    exit 1
fi

# Stop previous instances of these services
echo "Stopping previous server instances..."
pkill -f "mlx_lm.server" 2>/dev/null
pkill -f "http.server 8000" 2>/dev/null
pkill -f "image_server.py" 2>/dev/null
sleep 2

# Start the chat model server
echo "Starting GPT-OSS 20B on port 8080..."
nohup mlx_lm.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    > "$MLX_LOG" 2>&1 < /dev/null &

# Wait for the chat API to become available
echo "Waiting for the chat model to load..."
MLX_READY=0

for i in {1..90}; do
    if curl -fsS http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then
        MLX_READY=1
        break
    fi
    sleep 2
done

if [[ "$MLX_READY" -eq 1 ]]; then
    echo "OK: Chat model is ready."
else
    echo "WARNING: Chat model is not responding yet."
    echo "Check log: $MLX_LOG"
fi

# Start the web interface
echo "Starting web interface on port 8000..."
(
    cd "$WEB_DIR" || exit 1
    nohup python3 -m http.server 8000 --bind 0.0.0.0 \
        > "$WEB_LOG" 2>&1 < /dev/null &
)

# Start the MFLUX image generation/editing server
echo "Starting MFLUX image server on port 8090..."
(
    cd "$IMAGE_DIR" || exit 1
    nohup "$IMAGE_PYTHON" image_server.py \
        > "$IMAGE_LOG" 2>&1 < /dev/null &
)

# Wait briefly for the web and image servers
sleep 2

echo
echo "======================================"
echo " Startup summary"
echo "======================================"

if curl -fsS "http://127.0.0.1:8000/" >/dev/null 2>&1; then
    echo "Web interface:  OK"
else
    echo "Web interface:  CHECK $WEB_LOG"
fi

if lsof -nP -iTCP:8090 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Image server:   OK"
else
    echo "Image server:   CHECK $IMAGE_LOG"
fi

if [[ "$MLX_READY" -eq 1 ]]; then
    echo "Chat model:     OK"
else
    echo "Chat model:     CHECK $MLX_LOG"
fi

echo
echo "Mac URL:  http://localhost:8000"
echo "iPad URL: http://$IP:8000"
echo
echo "Logs:"
echo "  Chat:   $MLX_LOG"
echo "  Web:    $WEB_LOG"
echo "  Images: $IMAGE_LOG"
echo
echo "Startup complete. Servers run in the background."
