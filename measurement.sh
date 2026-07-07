#!/bin/bash
# Measure vLLM generation throughput (completion tokens per second).
set -euo pipefail

VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
MODEL="${MODEL:-Qwen3.5-35B-A3B}"
MAX_TOKENS="${MAX_TOKENS:-256}"
RUNS="${RUNS:-1}"
WARMUP="${WARMUP:-1}"
PROMPT="${PROMPT:-Write a short paragraph about edge AI and local LLM inference.}"

export no_proxy="${no_proxy:-127.0.0.1,localhost,${VLLM_HOST}}"

URL="http://${VLLM_HOST}:${VLLM_PORT}/v1/chat/completions"

usage() {
cat <<EOF
Usage: $(basename "$0")

Environment:
  VLLM_HOST   default: 127.0.0.1
  VLLM_PORT   default: 8000
  MODEL       default: Qwen3.5-35B-A3B
  MAX_TOKENS  default: 256
  RUNS        default: 1
  WARMUP      default: 1
  PROMPT      prompt text

Example:
  RUNS=3 MAX_TOKENS=512 ./measurement.sh
EOF
}

[[ "${1:-}" =~ ^(-h|--help)$ ]] && { usage; exit 0; }

command -v curl >/dev/null || { echo "curl is required"; exit 1; }
command -v python3 >/dev/null || { echo "python3 is required"; exit 1; }

payload=$(cat <<EOF
{
  "model":"${MODEL}",
  "messages":[
    {
      "role":"user",
      "content":"${PROMPT}"
    }
  ],
  "max_tokens":${MAX_TOKENS},
  "temperature":0
}
EOF
)

run_once() {

    tmp=$(mktemp)

    start=$(date +%s%N)

    code=$(curl -sS \
        -o "$tmp" \
        -w "%{http_code}" \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$URL")

    end=$(date +%s%N)

    if [[ "$code" != "200" ]]; then
        echo "HTTP $code"
        cat "$tmp"
        rm -f "$tmp"
        exit 1
    fi

    elapsed=$(python3 - <<EOF
print(($end-$start)/1e9)
EOF
)

    python3 - "$elapsed" "$tmp" <<'EOF'
import json
import sys

elapsed=float(sys.argv[1])

with open(sys.argv[2]) as f:
    body=json.load(f)

usage=body.get("usage",{})

prompt=usage.get("prompt_tokens",0)
completion=usage.get("completion_tokens",0)
total=usage.get("total_tokens",prompt+completion)

tps=completion/elapsed if elapsed else 0

print(f"prompt_tokens={prompt}")
print(f"completion_tokens={completion}")
print(f"total_tokens={total}")
print(f"elapsed_s={elapsed:.3f}")
print(f"completion_tokens_per_s={tps:.2f}")
EOF

    rm -f "$tmp"
}

echo "vLLM measurement"
echo "URL:   $URL"
echo "Model: $MODEL"

if [[ "$WARMUP" == "1" ]]; then
    echo "==> Warmup"
    run_once >/dev/null
fi

sum=0

for ((i=1;i<=RUNS;i++)); do

    echo "==> Run $i/$RUNS"

    out=$(run_once)

    echo "$out"

    tps=$(echo "$out" | awk -F= '/completion_tokens_per_s/ {print $2}')

    sum=$(python3 - <<EOF
print($sum+$tps)
EOF
)

done

if [[ "$RUNS" -gt 1 ]]; then

avg=$(python3 - <<EOF
print($sum/$RUNS)
EOF
)

printf "\nAverage completion_tokens_per_s=%.2f (%d runs)\n" "$avg" "$RUNS"

fi
