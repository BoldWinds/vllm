# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Project Overview

vLLM is a high-throughput LLM inference and serving engine. The core performance themes from the README/docs are PagedAttention/KV-cache memory management, continuous batching, chunked prefill, prefix caching, CUDA/HIP graph execution, optimized attention/GEMM/MoE kernels, speculative decoding, structured outputs, LoRA, multimodal models, and distributed inference across tensor/pipeline/data/expert/context parallelism.

This repository is Python-first but includes substantial C++/CUDA/HIP kernel code under `csrc/` and Python bindings/wrappers under `vllm/_custom_ops.py`, `vllm/kernels/`, and attention/model-executor layers.

## Common Development Commands

Follow `AGENTS.md`: never use system `python3` or bare `pip`; use `uv` and `.venv/bin/python`.

### Environment and editable install

```bash
uv venv --python 3.12
uv pip install -r requirements/lint.txt
pre-commit install

# Python-only development, using precompiled kernels
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# Python + C/CUDA development
uv pip install -e . --torch-backend=auto
```

For CUDA/C++ kernel iteration, prefer the documented incremental CMake workflow after an editable install:

```bash
.venv/bin/python tools/generate_cmake_presets.py
cmake --preset release
cmake --build --preset release --target install
```

### Tests

```bash
# Install common test dependencies
uv pip install -r requirements/test/cuda.in
# On x86_64 CUDA CI-compatible environments, requirements/test/cuda.txt may be used instead.

# Run a single test file
.venv/bin/python -m pytest tests/path/to/test_file.py -v

# Run one test by node id
.venv/bin/python -m pytest tests/path/to/test_file.py::test_name -v

# Run all tests in a directory
.venv/bin/python -m pytest tests/v1/core -v
```

Many tests require GPU/hardware-specific dependencies; CPU-only local runs are not expected to pass the full suite.

### Linting and type checks

```bash
# Staged files
pre-commit run

# All files
pre-commit run --all-files

# Specific hooks
pre-commit run ruff-check --all-files
pre-commit run ruff-format --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
```

### Documentation preview

```bash
uv pip install -r requirements/docs.txt
API_AUTONAV_EXCLUDE=vllm mkdocs serve
```

Use full `mkdocs serve` only when the generated API reference is needed; it is much slower.

## High-Level Architecture

### User/API entrypoints

- `vllm/entrypoints/cli/main.py` exposes the `vllm` console script configured in `pyproject.toml`.
- `vllm/entrypoints/llm.py` defines the offline `LLM` API.
- `vllm/entrypoints/openai/api_server.py` builds the OpenAI-compatible server; route-specific serving logic lives under `vllm/entrypoints/openai/*/serving.py`.
- `vllm/engine/arg_utils.py` converts CLI/API arguments into `EngineArgs`, which creates the central `VllmConfig`.

### Configuration model

- `vllm/config/vllm.py` defines `VllmConfig`, the aggregate runtime config passed through engine, executor, scheduler, and workers.
- Specialized configs live in `vllm/config/`: model, cache, scheduler, parallelism, compilation, device, LoRA, quantization, speculative decoding, multimodal, structured outputs, and observability.
- When tracing behavior, start from the relevant config object rather than hardcoding flags; most subsystems read from `VllmConfig`.

### V1 engine request flow

The active engine path is centered on `vllm/v1/`:

1. Frontends create requests and call `vllm/v1/engine/llm_engine.py::LLMEngine` or `vllm/v1/engine/async_llm.py::AsyncLLM`.
2. `InputProcessor` in `vllm/v1/engine/input_processor.py` converts frontend inputs into `EngineCoreRequest` objects.
3. `EngineCoreClient` in `vllm/v1/engine/core_client.py` selects in-process or multiprocessing clients.
4. `EngineCore` in `vllm/v1/engine/core.py` owns the inner loop: model executor setup, KV-cache initialization, scheduler construction, request stepping, and output assembly.
5. `OutputProcessor` in `vllm/v1/engine/output_processor.py` converts core outputs into public `RequestOutput`/streaming outputs.

### Scheduler and KV cache

- `vllm/v1/core/sched/scheduler.py::Scheduler` implements continuous batching decisions. Important state includes `waiting`, `running`, `requests`, `finished_req_ids`, scheduling token budgets, encoder cache state, speculative decoding state, and optional KV/EC connectors.
- `vllm/v1/core/sched/output.py::SchedulerOutput` is the boundary object passed from scheduling to execution.
- `vllm/v1/core/kv_cache_utils.py` derives KV-cache specs/configs from model/executor profiling and cache settings.
- `vllm/v1/core/kv_cache_manager.py::KVCacheManager` coordinates prefix-cache lookup, block allocation, admission checks, and block lifecycle.
- `vllm/v1/core/block_pool.py::BlockPool` manages physical KV-cache blocks and cached block hashes.

### Executors, workers, and model runners

- `vllm/v1/executor/abstract.py::Executor` abstracts model execution over one or many devices; concrete backends include uniprocess, multiprocessing, Ray, and external launcher executors.
- Executors communicate with worker objects via `collective_rpc` and return `ModelRunnerOutput` objects to `EngineCore`.
- `vllm/v1/worker/gpu_worker.py::Worker` owns device initialization, distributed initialization, memory profiling, KV-cache allocation, sleep/wake memory behavior, and model-runner lifecycle.
- `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner` is the main GPU execution path: it prepares batches from scheduler output, manages input tensors/block tables/attention metadata, runs model forward passes, handles CUDA graph/torch.compile integration, and invokes sampling/pooling/spec-decode logic.
- CPU/TPU/XPU variants live next to the GPU worker/model runner under `vllm/v1/worker/`.

### Model execution and kernels

- Model definitions live in `vllm/model_executor/models/`; e.g. Llama layers are in `vllm/model_executor/models/llama.py`.
- Model loading is under `vllm/model_executor/model_loader/`; `get_model_loader` and `get_model` choose load formats and initialize model classes.
- Attention layers are under `vllm/model_executor/layers/attention/`; the generic `Attention` module delegates backend-specific execution through V1 attention metadata/backends.
- V1 attention backend interfaces are in `vllm/v1/attention/backend.py`; backend selection/registration is in `vllm/v1/attention/backends/registry.py`; concrete backends and Triton/FlashAttention/FlashInfer/etc. implementations live under `vllm/v1/attention/backends/` and `vllm/v1/attention/ops/`.
- `vllm/v1/attention/ops/paged_attn.py::PagedAttention` wraps paged KV-cache layout operations and calls platform-specific custom ops.
- `vllm/_custom_ops.py` exposes Python wrappers around registered `torch.ops._C` kernels implemented under `csrc/`.
- Kernel source areas include `csrc/attention`, `csrc/moe`, `csrc/quantization`, `csrc/mamba`, `csrc/cutlass_extensions`, `csrc/cpu`, and `csrc/rocm`.

### Sampling, structured outputs, and extensions

- Sampling logic lives in `vllm/v1/sample/`, with `sampler.py::Sampler`, logits processors, penalties, top-k/top-p ops, and logprobs helpers.
- Speculative decoding is under `vllm/v1/spec_decode/`.
- Structured output backends are under `vllm/v1/structured_output/`.
- LoRA support spans `vllm/lora/` plus worker/model-runner mixins.
- Multimodal processing and registries live under `vllm/multimodal/` and interact with scheduler encoder budgets/caches.
- Distributed groups and collectives are managed in `vllm/distributed/parallel_state.py`; KV transfer, disaggregated prefill/decode, elastic EP, and weight transfer are under `vllm/distributed/`.

## Testing Map

- Engine/scheduler/KV-cache tests: `tests/v1/engine`, `tests/v1/core`, `tests/v1/worker`, `tests/v1/attention`.
- API server and frontend tests: `tests/entrypoints`.
- Kernel tests: `tests/kernels`, plus hardware-specific `tests/cuda` and `tests/rocm`.
- Model behavior/weight loading tests: `tests/models`, `tests/model_executor`, `tests/weight_loading`.
- Distributed behavior: `tests/distributed` and `tests/v1/distributed`.
- Compilation/CUDA graph behavior: `tests/compile` and `tests/v1/cudagraph`.

## Repository-Specific Notes

- Before modifying `AGENTS.md` or guides it links to, read `docs/contributing/editing-agent-instructions.md`.
- `.gemini/config.yaml` configures Gemini review behavior only; there are no Cursor or Copilot instruction files in this checkout.
- PR work must follow the duplicate-work checks and AI-assisted contribution policy in `AGENTS.md`.
