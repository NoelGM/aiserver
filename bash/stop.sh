#!/bin/bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
CHAT_PORT=$("$PYTHON" "$ROOT/src/configuration.py" chat port) || exit 1
WEB_PORT=$("$PYTHON" "$ROOT/src/configuration.py" web port) || exit 1
IMAGE_PORT=$("$PYTHON" "$ROOT/src/configuration.py" images port) || exit 1

echo ""
echo "=========================================="
echo "             STOPPING MY AI"
echo "=========================================="
echo ""

for PORT in "$CHAT_PORT" "$WEB_PORT" "$IMAGE_PORT"; do
    for PID in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN); do
        COMMAND=$(ps -p "$PID" -o args=)
        case "$COMMAND" in
            *mlx_lm.server*|*http.server*|*web_server.py*|*image_server.py*)
                echo "Stopping service on port $PORT (PID $PID)..."
                kill "$PID" || exit 1
                ;;
            *)
                echo "ERROR: Port $PORT belongs to another application (PID $PID)."
                exit 1
                ;;
        esac
    done
done

echo ""
echo "=========================================="
echo "              MY AI STOPPED"
echo "=========================================="
echo ""

echo "This window will remain open."
echo "You can close it when ready."

if [[ -t 0 ]]; then
    read -n 1
fi
