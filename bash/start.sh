#!/bin/zsh

# Local AI environment startup script

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WEB_DIR="$ROOT/web"
IMAGE_DIR="$ROOT/src"
IMAGE_PYTHON="$ROOT/.venv/bin/python"
MODEL="mlx-community/gpt-oss-20b-OptiQ-4bit"

MLX_LOG="$ROOT/logs/mlx.log"
WEB_LOG="$ROOT/logs/web.log"
IMAGE_LOG="$ROOT/logs/image.log"

mkdir -p "$ROOT/logs" "$ROOT/data/images/inputs" "$ROOT/data/images/outputs"

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
if [[ ! -x "$IMAGE_PYTHON" ]]; then
    echo "ERROR: Image virtual environment not found:"
    echo "$IMAGE_PYTHON"
    read -k 1 "?Press any key to exit..."
    echo
    exit 1
fi

if ! "$IMAGE_PYTHON" -c "import mlx_lm.server" >/dev/null 2>&1; then
    echo "ERROR: MLX-LM is not installed in $ROOT/.venv."
    echo "  \"$IMAGE_PYTHON\" -m pip install -r \"$ROOT/requirements.txt\""
    exit 1
fi

if [[ ! -f "$IMAGE_DIR/image_server.py" ]]; then
    echo "ERROR: $IMAGE_DIR/image_server.py was not found."
    read -k 1 "?Press any key to exit..."
    echo
    exit 1
fi

if [[ ! -x "$ROOT/.venv/bin/mflux-generate-flux2" ||
      ! -x "$ROOT/.venv/bin/mflux-generate-flux2-edit" ]]; then
    echo "ERROR: MFLUX is not installed in $ROOT/.venv."
    echo "Install image dependencies with:"
    echo "  \"$IMAGE_PYTHON\" -m pip install -r \"$ROOT/requirements.txt\""
    exit 1
fi

# Stop previous instances of these services
echo "Stopping previous server instances..."
for PORT in 8080 8000 8090; do
    for PID in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN); do
        COMMAND=$(ps -p "$PID" -o args=)
        case "$PORT:$COMMAND" in
            8080:*mlx_lm.server*|8000:*http.server\ 8000*|8090:*image_server.py*)
                kill "$PID"
                ;;
            *)
                echo "ERROR: Port $PORT is occupied by another application (PID $PID)."
                exit 1
                ;;
        esac
    done
done
sleep 2

# Start the chat model server
echo "Starting GPT-OSS 20B on port 8080..."
nohup "$IMAGE_PYTHON" -m mlx_lm.server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
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
    nohup "$IMAGE_PYTHON" -m http.server 8000 --bind 0.0.0.0 \
        > "$WEB_LOG" 2>&1 < /dev/null &
)

# Start the MFLUX image generation/editing server
echo "Starting MFLUX image server on port 8090..."
(
    cd "$IMAGE_DIR" || exit 1
    nohup "$IMAGE_PYTHON" -u image_server.py \
        > "$IMAGE_LOG" 2>&1 < /dev/null &
)

# Wait for the image API, not just an open TCP port.
IMAGE_READY=0
for i in {1..30}; do
    if curl -fsS --max-time 2 "http://127.0.0.1:8090/health" >/dev/null 2>&1; then
        IMAGE_READY=1
        break
    fi
    sleep 1
done

echo
echo "======================================"
echo " Startup summary"
echo "======================================"

if curl -fsS "http://127.0.0.1:8000/" >/dev/null 2>&1; then
    echo "Web interface:  OK"
else
    echo "Web interface:  CHECK $WEB_LOG"
fi

if [[ "$IMAGE_READY" -eq 1 ]]; then
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

if [[ "$IMAGE_READY" -ne 1 ]]; then
    echo "ERROR: Image API failed to start. See $IMAGE_LOG."
    exit 1
fi

echo "Startup complete. Servers run in the background."
