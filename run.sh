#!/bin/bash
# LocalTrainer v3.0 - Startup Script

set -e

echo "========================================="
echo "  LocalTrainer v3.0 - Linux Sysadmin AI"
echo "========================================="

# فحص Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    exit 1
fi

echo "Python: $(python3 --version)"

# فحص Ollama
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama not installed"
    echo "  Install: curl -fsSL https://ollama.com/install.sh | sh"
else
    echo "Ollama: $(ollama --version 2>&1 | head -1)"
fi

# فحص GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
else
    echo "WARNING: No NVIDIA GPU detected"
fi

# إنشاء المجلدات
mkdir -p data models logs checkpoints

echo ""
echo "Server: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo "Frontend: http://localhost:8000/static/index.html"
echo ""

python -m backend.main
