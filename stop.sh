#!/bin/bash
# Stop the vLLM container started by ./start.sh. Leaves the Hugging Face model
# cache (HF_MODEL_CACHE_ROOT, default /home/aibox/models/huggingface) untouched.
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm}"
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true
echo "$CONTAINER_NAME stopped"
