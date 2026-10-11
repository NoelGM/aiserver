#!/bin/bash

echo ""
echo "=========================================="
echo "             STOPPING MY AI"
echo "=========================================="
echo ""

for PORT in 8080 8000 8090; do
    for PID in $(lsof -tiTCP:"$PORT" -sTCP:LISTEN); do
        COMMAND=$(ps -p "$PID" -o args=)
        case "$PORT:$COMMAND" in
            8080:*mlx_lm.server*|8000:*http.server\ 8000*|8090:*image_server.py*)
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
