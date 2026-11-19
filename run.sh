#!/usr/bin/env bash
# SignBridge AI Startup Script
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================================="
echo "    Starting SignBridge AI Assistant Server"
echo "=========================================================="

if [ -d "$DIR/venv" ]; then
    PYTHON="$DIR/venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$DIR"
echo "Using Python: $($PYTHON --version)"

# Check if model exists
if [ ! -f "$DIR/models/asl_classifier.joblib" ]; then
    echo "Trained model not found. Training model now..."
    $PYTHON scripts/train_model.py
fi

echo "Launching SignBridge AI on http://localhost:8000"
exec $PYTHON -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
