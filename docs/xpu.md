# Intel GPUs

prime-rl runs RL and SFT on Intel GPUs through PyTorch's XPU backend. The trainer picks CUDA or XPU from the installed torch build, so the same entrypoints, configs and CLI flags apply. This page covers the XPU image, what differs from a CUDA run, and how to validate an install.

## Table of Contents

- [Image](#image)
- [Running](#running)
- [Selecting GPUs](#selecting-gpus)
- [Differences from CUDA](#differences-from-cuda)
- [Fitting smaller cards](#fitting-smaller-cards)
- [Validating an install](#validating-an-install)

## Image

`uv sync` resolves CUDA wheels only, so Intel GPUs use [`Dockerfile.xpu`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/Dockerfile.xpu). It installs the XPU builds of torch and vLLM and keeps every other package at the version pinned in `uv.lock`. The host needs an Intel GPU with its kernel driver loaded; the user-mode driver ships in the image.

Build from the repo root with submodules checked out:

```bash
eval "$(bash scripts/docker-editable-pretend-versions.sh --shell .)"
docker build -f Dockerfile.xpu -t prime-rl:xpu \
  --build-arg VERIFIERS_PRETEND_VERSION --build-arg RENDERERS_PRETEND_VERSION \
  --build-arg PYDANTIC_CONFIG_PRETEND_VERSION .
```

Start a container with the host GPUs:

```bash
docker run --rm -it --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface prime-rl:xpu
```

The `/dev/dri/by-path` mount lets oneCCL exchange GPU memory handles inside the container. Multi-GPU training needs it.

The image sets `UV_NO_SYNC=1`, so `uv run` keeps the XPU stack instead of syncing the CUDA lock. Commands from the rest of the docs work unchanged inside the container.

## Running

Use the filesystem weight broadcast. The default NCCL broadcast is CUDA-only:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml --weight-broadcast.type filesystem
```

SFT needs the same flag when it runs online evals against an inference server:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --weight-broadcast.type filesystem
```

The reverse-text configs carry the setting as a commented `[weight_broadcast]` block if you prefer TOML.

## Selecting GPUs

`ZE_AFFINITY_MASK` plays the role of `CUDA_VISIBLE_DEVICES`. The launcher reads it, then hands each inference server and trainer its own mask. An unset, empty or `default` mask means every device.

```bash
# Inference on GPU 0, trainer on GPU 1 (the default two-GPU layout)
ZE_AFFINITY_MASK=0,1 uv run rl @ examples/basic/reverse-text/rl.toml --weight-broadcast.type filesystem
```

Tile selectors such as `0.0` reach the trainer unchanged. vLLM accepts device indices only, so inference GPUs need plain indices.

## Differences from CUDA

| Area | CUDA | XPU |
|---|---|---|
| Trainer collectives | NCCL | XCCL (oneCCL) |
| Weight broadcast | NCCL, NIXL or filesystem | filesystem |
| Attention | `attn = "auto"` picks FA4, FA3 or FA2 | `attn = "auto"` picks `flash_attention_2` |
| FlashAttention-2 | `flash_attn` package | `kernels-community/flash-attn2` from the Kernel Hub, downloaded into the Hugging Face cache on first use |
| Full optimizer offload | binds each rank to its GPU's NUMA node | skips NUMA binding with a warning |
| Trainer allocator | expandable segments on | expandable segments off (oneCCL collectives need it off) |

CUDA-only kernel packages are not installed in the image: FlashAttention 3 and 4, DeepEP, DeepGEMM, FlashMLA, mamba-ssm, TileLang, quack, sonic-moe, NIXL and Mooncake. Features built on them are unavailable on XPU.

Covered on XPU: vLLM inference on one GPU or with tensor parallelism, the orchestrator and environment servers, the FSDP2 trainer on one or more GPUs, context parallelism, `torch.compile`, activation checkpointing and offloading, LoRA, both CPU optimizer offload modes, checkpoint save and resume, and SFT with online evals. Tested models are dense Qwen3 (0.6B and 4B).

## Fitting smaller cards

Example configs are sized for 80 GB cards. On 24 to 32 GB cards:

- Cap the context vLLM reserves KV cache for. `Qwen3-4B-Instruct-2507` advertises a 262k context; `--inference.vllm.max-model-len 8192` is enough for the alphabet-sort example.
- Lower `seq_len` for SFT. The reverse-text SFT example fits 24 GB at `--seq-len 2048`.
- vLLM claims 90% of its GPU by default and refuses to start when another process already holds memory there. Check that the inference GPU is idle before launching.

## Validating an install

Run these inside the container, cheapest first.

1. Device selection and the launcher's GPU masks. These fake the torch build, so they also run in CPU CI:

    ```bash
    uv run pytest tests/unit/utils/test_device.py -v
    ```

2. Optimizer offload on the GPU:

    ```bash
    uv run pytest tests/unit/train/test_state_offload.py tests/unit/train/test_cpu_adam.py -v
    ```

3. The reverse-text RL example end to end (inference, orchestrator, trainer). Eval reward should climb from about 0.1 to above 0.8 over the 20 steps:

    ```bash
    uv run rl @ examples/basic/reverse-text/rl.toml --weight-broadcast.type filesystem --no-monitors.prime
    ```
