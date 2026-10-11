#!/bin/zsh

# Local AI environment startup script

export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.local/bin:$PATH"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_DIR="$ROOT/src"
IMAGE_PYTHON="$ROOT/.venv/bin/python"

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

get_setting() {
    "$IMAGE_PYTHON" "$ROOT/src/configuration.py" "$1" "$2"
}

CHAT_HOST=$(get_setting chat host) || exit 1
CHAT_PORT=$(get_setting chat port) || exit 1
MODEL=$(get_setting chat model) || exit 1
WEB_PORT=$(get_setting web port) || exit 1
IMAGE_PORT=$(get_setting images port) || exit 1
IMAGE_GENERATE_COMMAND=$(get_setting images generate_command) || exit 1
IMAGE_EDIT_COMMAND=$(get_setting images edit_command) || exit 1
CHAT_HEALTH=$(get_setting chat health_url) || exit 1
WEB_HEALTH=$(get_setting web health_url) || exit 1
IMAGE_HEALTH=$(get_setting images health_url) || exit 1

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

if [[ ! -x "$ROOT/.venv/bin/$IMAGE_GENERATE_COMMAND" ||
      ! -x "$ROOT/.venv/bin/$IMAGE_EDIT_COMMAND" ]]; then
    echo "ERROR: MFLUX is not installed in $ROOT/.venv."
    echo "Install image dependencies with:"
    echo "  \"$IMAGE_PYTHON\" -m pip install -r \"$ROOT/requirements.txt\""
    exit 1
fi

# Stop previous instances of these services
echo "Stopping previous server instances..."
for PORT in "$CHAT_PORT" "$WEB_PORT" "$IMAGE_PORT"; do
    for PID in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN); do
        COMMAND=$(ps -p "$PID" -o args=)
        case "$COMMAND" in
            *mlx_lm.server*|*http.server*|*web_server.py*|*image_server.py*)
                kill "$PID" || exit 1
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
echo "Starting $MODEL on port $CHAT_PORT..."
nohup "$IMAGE_PYTHON" -m mlx_lm.server \
    --model "$MODEL" \
    --host "$CHAT_HOST" \
    --port "$CHAT_PORT" \
    > "$MLX_LOG" 2>&1 < /dev/null &

# Wait for the chat API to become available
echo "Waiting for the chat model to load..."
MLX_READY=0

for i in {1..90}; do
    if curl -fsS --max-time 2 "$CHAT_HEALTH" >/dev/null 2>&1; then
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
echo "Starting web interface on port $WEB_PORT..."
nohup "$IMAGE_PYTHON" -u "$ROOT/src/web_server.py" \
    > "$WEB_LOG" 2>&1 < /dev/null &

# Start the MFLUX image generation/editing server
echo "Starting MFLUX image server on port $IMAGE_PORT..."
(
    cd "$IMAGE_DIR" || exit 1
    nohup "$IMAGE_PYTHON" -u image_server.py \
        > "$IMAGE_LOG" 2>&1 < /dev/null &
)

# Wait for the image API, not just an open TCP port.
IMAGE_READY=0
WEB_READY=0
for i in {1..30}; do
    if curl -fsS --max-time 2 "$IMAGE_HEALTH" >/dev/null 2>&1; then
        IMAGE_READY=1
    fi
    if curl -fsS --max-time 2 "$WEB_HEALTH" >/dev/null 2>&1; then
        WEB_READY=1
    fi
    if [[ "$IMAGE_READY" -eq 1 && "$WEB_READY" -eq 1 ]]; then
        break
    fi
    sleep 1
done

echo
echo "======================================"
echo " Startup summary"
echo "======================================"

if [[ "$WEB_READY" -eq 1 ]]; then
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
echo "Mac URL:  http://localhost:$WEB_PORT"
echo "iPad URL: http://$IP:$WEB_PORT"
echo
echo "Logs:"
echo "  Chat:   $MLX_LOG"
echo "  Web:    $WEB_LOG"
echo "  Images: $IMAGE_LOG"
echo

if [[ "$IMAGE_READY" -ne 1 || "$WEB_READY" -ne 1 || "$MLX_READY" -ne 1 ]]; then
    echo "ERROR: One or more services failed to start. See the logs above."
    exit 1
fi

echo "Startup complete. Servers run in the background."
