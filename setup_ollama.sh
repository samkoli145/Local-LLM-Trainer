#!/bin/bash
# Setup Ollama for LocalTrainer

set -e

echo "Setting up Ollama..."

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed: $(ollama --version 2>&1 | head -1)"
fi

# Start Ollama in background
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama server..."
    ollama serve &
    sleep 3
fi

# Pull models
echo "Pulling teacher model (qwen2.5:7b)..."
ollama pull qwen2.5:7b

echo "Pulling student model (qwen2.5:3b)..."
ollama pull qwen2.5:3b

echo "Ollama setup complete!"
ollama list
