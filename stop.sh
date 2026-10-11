#!/bin/bash

echo ""
echo "=========================================="
echo "             STOPPING MY AI"
echo "=========================================="
echo ""

echo "→ Stopping MLX-LM..."

pkill -f "mlx_lm.server"

echo "✓ MLX-LM stopped."

echo ""
echo "→ Stopping web server..."

pkill -f "http.server 8000"

echo "✓ Web server stopped."

echo ""
echo "=========================================="
echo "              MY AI STOPPED"
echo "=========================================="
echo ""

echo "This window will remain open."
echo "You can close it when ready."

read -n 1
