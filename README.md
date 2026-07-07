# vLLM on Intel XPU (Arc)

This directory runs [Intel llm-scaler-vllm](https://github.com/intel/llm-scaler) in Docker on **Intel Arc (XPU)** with an OpenAI-compatible API on port **8000**.

The main entrypoint is **`start.sh`**: it loads the container image (if needed), downloads the Hugging Face model **once** into a persistent cache, stops any old `vllm` container, and starts a new one.

## Prerequisites

- Docker (or Podman with compatible `docker` CLI)
- Intel Arc GPU with `/dev/dri` available on the host
- Image `intel/llm-scaler-vllm:latest` locally, or `llm-scaler-vllm.tar` in this directory (loaded automatically)
- Enough **disk** under `HF_MODEL_CACHE_ROOT` for the chosen model
- For gated models: `HF_TOKEN` set on first download

Recommended image for Qwen3.5 MoE models: `intel/llm-scaler-vllm:0.14.0-b8.3.1` or newer (override with `VLLM_IMAGE`).

## Quick start

**Default profile:** `Qwen/Qwen3.5-35B-A3B` on **one Arc (~32 GiB)** with **online `sym_int4`** (Intel’s recommended way to fit 35B MoE on a single GPU). Plain `./start.sh` uses that—no extra env vars required.

```bash
cd /data/vllm
chmod +x start.sh

export HF_TOKEN=hf_...   # first download only, if the repo requires it
./start.sh

docker logs -f vllm
no_proxy=127.0.0.1 curl -s http://127.0.0.1:8000/v1/models
```

Equivalent explicit settings (already the `start.sh` defaults for 35B MoE):

```bash
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=sym_int4 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_PREFIX_CACHING=0 \
./start.sh
```

`start.sh` **stops and replaces** the existing container named `vllm` (or `VLLM_CONTAINER_NAME`). You do not need a separate stop script before re-running.

First startup for **35B + online INT4** can take **20–40+ minutes** (loading 14 safetensor shards, quantizing on XPU). Later restarts are faster if weights are already on disk.

---

## Recipes by model size

Use **one command block** that matches your model. The API model name is the last segment of `HF_MODEL_ID` (e.g. `Qwen3-8B`, `Qwen3.5-35B-A3B`). Set the same name in `measurement.sh` via `MODEL=`.

### Qwen3-8B (dense, single Arc ~24 GiB)

Fits in BF16/FP16 on one GPU without quantization.

```bash
HF_MODEL_ID=Qwen/Qwen3-8B \
VLLM_QUANTIZATION= \
VLLM_DTYPE=bfloat16 \
VLLM_MAX_MODEL_LEN=8192 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_ENFORCE_EAGER=0 \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `HF_MODEL_ID` | `Qwen/Qwen3-8B` | Model to download/serve |
| `VLLM_QUANTIZATION` | *(empty)* | No online quant needed |
| `VLLM_DTYPE` | `bfloat16` or `float16` | Full-precision weights |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | One GPU |
| `VLLM_ENFORCE_EAGER` | `0` | Allow compile/graph opts if stable |

Optional FP8 on 8B (smaller VRAM, slightly different numerics):

```bash
HF_MODEL_ID=Qwen/Qwen3-8B \
VLLM_QUANTIZATION=fp8 \
VLLM_DTYPE=float16 \
./start.sh
```

---

### Do **not** use `Qwen/Qwen3.5-35B-A3B-FP8` on Intel XPU

Hugging Face **pre-quantized FP8** checkpoints (`*-FP8`) are **not supported** for Qwen3.5 **MoE** on this stack. vLLM fails during layer init with:

```text
AssertionError: assert not quant_config.is_checkpoint_fp8_serialized
```

Intel XPU MoE only implements **online FP8** from full BF16/FP16 weights (`--quantization fp8`), not serialized FP8 safetensors. `start.sh` blocks `*-FP8` model ids unless you set `VLLM_ALLOW_PREQUANT_FP8=1`.

---

### Qwen3.5-35B-A3B (MoE, single Arc ~32 GiB) — **default** (`sym_int4`)

This is what **`./start.sh` runs by default** when `HF_MODEL_ID` is `Qwen/Qwen3.5-35B-A3B` and `VLLM_TENSOR_PARALLEL_SIZE=1`. Online FP8 from BF16 on one card often OOMs; Intel’s fit-for-one-GPU path is **online `sym_int4`**.

```bash
# Same as plain ./start.sh for the default model:
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=sym_int4 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_PREFIX_CACHING=0 \
./start.sh
```

| Variable | Default (35B MoE, 1 GPU) | Why |
|----------|---------------------------|-----|
| `VLLM_QUANTIZATION` | `sym_int4` | Fits ~32 GiB VRAM |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | Host RAM during quant load |
| `VLLM_MAX_MODEL_LEN` | `4096` | Leaves headroom for KV cache |
| `VLLM_PREFIX_CACHING` | `0` | Saves VRAM |

`start.sh` sets `VLLM_QUANTIZE_Q40_LIB` inside the container for `sym_int4`.

---

### Qwen3.5-35B-A3B (MoE, full BF16 + online FP8, single Arc) — optional

**Not the default.** Often **OOMs on ~32 GiB XPU** after all shards load. Prefer the default **`sym_int4`** recipe or **two GPUs** below.

```bash
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=fp8 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.82 \
VLLM_TENSOR_PARALLEL_SIZE=1 \
VLLM_MAX_MODEL_LEN=4096 \
VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
VLLM_BLOCK_SIZE=64 \
VLLM_ENFORCE_EAGER=1 \
VLLM_CPU_OFFLOAD_GB=0 \
VLLM_PREFIX_CACHING=0 \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `VLLM_QUANTIZATION` | `fp8` | Shrink weights for one GPU |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | CPU RAM during online quant |
| `VLLM_MAX_MODEL_LEN` | `4096` | Less KV reservation (may still OOM on 32 GiB) |
| `VLLM_CPU_OFFLOAD_GB` | `0` | **Do not** use vLLM CPU offload on this stack |

If you still see **XPU OOM** at end of load, try **`VLLM_TENSOR_PARALLEL_SIZE=2`** or lower `VLLM_MAX_MODEL_LEN`.

---

### Qwen3.5-35B-A3B (MoE, two Arc GPUs)

Intel’s reference uses **tensor parallel = 2** across two cards.

```bash
HF_MODEL_ID=Qwen/Qwen3.5-35B-A3B \
VLLM_QUANTIZATION=fp8 \
VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1 \
VLLM_DTYPE=float16 \
VLLM_GPU_MEMORY_UTILIZATION=0.90 \
VLLM_TENSOR_PARALLEL_SIZE=2 \
ZE_AFFINITY_MASK=0,1 \
VLLM_MAX_MODEL_LEN=16384 \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 \
VLLM_BLOCK_SIZE=64 \
VLLM_ENFORCE_EAGER=1 \
./start.sh
```

| Variable | Value | Why |
|----------|--------|-----|
| `VLLM_TENSOR_PARALLEL_SIZE` | `2` | Split model across 2 XPUs |
| `ZE_AFFINITY_MASK` | `0,1` | Bind to first two Arc devices |
| `VLLM_MAX_MODEL_LEN` | higher (e.g. `16384`–`40000`) | More KV cache possible with 2 GPUs |

Adjust `ZE_AFFINITY_MASK` to match your `renderD*` / GPU indices.

---

### Other sizes (14B / 32B dense, 30B-A3B MoE)

Use the **closest recipe** and tune VRAM:

| Approx. size | Starting point |
|--------------|----------------|
| ≤ 8B dense | 8B recipe, no quant |
| 14B–32B dense | 8B recipe; add `VLLM_QUANTIZATION=fp8` if OOM |
| 30B-A3B / 35B-A3B MoE | 35B recipe; `tp=1` or `tp=2` |
| 122B-A10B MoE | Intel docs: FP8, often `tp≥2`, newer image tag |

Always set `HF_MODEL_ID` to the full Hugging Face id (e.g. `Qwen/Qwen3-14B`).

---

## Environment reference

All variables are optional unless noted. Export them **before** `./start.sh`.

### Model and cache

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_ID` | `Qwen/Qwen3.5-35B-A3B` | Hugging Face model id |
| `HF_GGUF_FILE` | *(unset)* | If set, download/serve a single `.gguf` file instead of full safetensors |
| `HF_MODEL_CACHE_ROOT` | `/home/aibox/models/huggingface` | Host directory mounted at `/models` in the container |
| `HF_TOKEN` | *(unset)* | Token for gated downloads (first run) |

Cache path: `$HF_MODEL_CACHE_ROOT/$(basename "$HF_MODEL_ID")`.

### Container and image

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE` | `intel/llm-scaler-vllm:latest` | Docker image |
| `VLLM_CONTAINER_NAME` | `vllm` | Container name |
| `VLLM_PORT` | `8000` | Host port → API `8000` |
| `VLLM_IPC` | `host` | `host` → `--ipc=host`; else use `VLLM_SHM_SIZE` (default `16g`) |

### Memory and quantization (XPU)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_QUANTIZATION` | `sym_int4` for 35B MoE (1 GPU) | Online quant: `sym_int4`, `fp8`, etc. Set **empty** for no quant |
| `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT` | `1` | Use host RAM during FP8/INT4 load (only if quant enabled) |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of XPU memory for weights + KV cache |
| `VLLM_CPU_OFFLOAD_GB` | `0` | vLLM `--cpu-offload-gb`; **not recommended** for Qwen3.5 MoE on XPU |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of XPUs (`--tensor-parallel-size`) |
| `ZE_AFFINITY_MASK` | *(unset)* | e.g. `0,1` for two GPUs |

### vLLM serve tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_DTYPE` | `float16` | `float16`, `bfloat16`, etc. |
| `VLLM_MAX_MODEL_LEN` | `8192` | Max context length |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | `8192` | Chunked prefill batch cap |
| `VLLM_BLOCK_SIZE` | `64` | KV block size |
| `VLLM_ENFORCE_EAGER` | `1` | `1` → `--enforce-eager`; `0` to disable |
| `VLLM_PREFIX_CACHING` | `1` or profile | `0` → `--no-enable-prefix-caching` (large MoE / tight VRAM) |
| `VLLM_EXTRA_ARGS` | *(unset)* | Extra flags passed to `vllm serve` (shell word-split) |

### Intel worker env (set inside container by script)

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow long `max_model_len` when supported |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Multiprocessing start method |

`VLLM_TARGET_DEVICE=xpu` is always set by the script.

`start.sh` picks defaults from the model id (8B vs `*-FP8` vs full 35B MoE). Export a variable before `./start.sh` to override.

---

## Why not `--cpu-offload-gb` for 35B on XPU?

Generic vLLM **CPU weight offload** (`VLLM_CPU_OFFLOAD_GB` / `--cpu-offload-gb`) targets a different path than Intel’s Arc stack. For **Qwen3.5 MoE** on XPU it often fails with **engine core initialization failed** before loading finishes.

The supported way to use **both host RAM and GPU** here is:

1. **`--quantization fp8`** (or INT4) so weights fit on XPU at inference time  
2. **`VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=1`** so **CPU RAM** is used while shards are loaded and quantized  

That is what the default `start.sh` settings implement for the 35B model.

---

## Operations

**Logs**

```bash
docker logs -f vllm
```

**Health / model list**

```bash
no_proxy=127.0.0.1 curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool
```

**Stop without restart**

```bash
docker rm -f vllm
```

**Benchmark** (after the server is ready; `MODEL` must match `--served-model-name`):

```bash
MODEL=Qwen3.5-35B-A3B RUNS=3 ./measurement.sh
```

---

## Troubleshooting

| Symptom | Things to try |
|---------|----------------|
| `Engine core initialization failed` | Remove `VLLM_CPU_OFFLOAD_GB`; use FP8 recipe for 35B; check `docker logs` lines *above* the RuntimeError |
| Stuck at `Loading safetensors checkpoint shards` | Normal for first FP8 load; wait until 14/14 completes |
| OOM on GPU at end of load (`~30 GiB allocated`, +1 GiB fails) | Default is already `sym_int4`; try **`VLLM_TENSOR_PARALLEL_SIZE=2`** or lower `VLLM_MAX_MODEL_LEN` |
| `AssertionError` / `checkpoint_fp8_serialized` | Do not use HF `*-FP8` MoE checkpoints on XPU; use full `Qwen3.5-35B-A3B` + online quant |
| OOM on GPU during inference | Lower `VLLM_MAX_MODEL_LEN`, `VLLM_GPU_MEMORY_UTILIZATION`, or `VLLM_PREFIX_CACHING=0` |
| OOM on host during load | `VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT=0` (Intel doc tradeoff) or more free RAM |
| Download fails | Set `HF_TOKEN`; ensure network on first run (download container only) |
| Image not found | Place `llm-scaler-vllm.tar` here or `docker load -i ...` |

---

## Files

| File | Purpose |
|------|---------|
| `start.sh` | Download (once) + run vLLM server |
| `measurement.sh` | Simple throughput test against `/v1/chat/completions` |
| `llm-scaler-vllm.tar` | Optional offline image load |

Further model matrix and flags: [Intel llm-scaler vLLM README](https://github.com/intel/llm-scaler/blob/main/vllm/README.md).
