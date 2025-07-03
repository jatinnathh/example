#!/bin/bash
set -xe  # print each command before running it
echo "🚀 Starting backend (FastAPI)..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

echo "🌐 Starting Expo (Tunnel)..."
expo start --tunnel --clear
