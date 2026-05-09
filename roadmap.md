# vLLM 源码阅读 Roadmap：100 小时内建立 AI Infra 面试主线

## 0. 目标与边界

这个路线面向秋招 AI Infra 面试，不追求读完整个 vLLM 仓库，而是围绕 **LLM serving 主链路** 建立能讲清楚、能画图、能回答追问的源码理解。

总时间控制在 **80–100 小时**。如果时间不足，优先完成主链路、Scheduler、KV cache / PagedAttention、GPU ModelRunner；辅助模块只做定位和面试级理解。

每个模块阅读后，都沉淀到 `docs/code-reading/` 下的一篇 Markdown 笔记。笔记固定回答：

- 这个模块负责什么？
- 核心类 / 函数 / 状态有哪些？
- 输入是什么？输出是什么？
- 谁调用它？它调用谁？
- 它解决了 LLM serving 的什么问题？
- 面试时应该怎么讲？

## 1. 推荐阅读总顺序

先按一次请求的 serving 链路读，再补关键性能模块：

```text
API / LLM entrypoint
  -> EngineArgs / VllmConfig
  -> LLMEngine / AsyncLLM
  -> EngineCoreClient
  -> EngineCore
  -> Scheduler
  -> KVCacheManager / BlockPool
  -> Worker / GPUModelRunner
  -> Attention / PagedAttention
  -> Sampler / OutputProcessor
```

阅读原则：

1. 先读 Python 调度主链路，再读 CUDA/C++ kernel。
2. 先搞清楚对象关系和状态流转，再看优化细节。
3. 每个模块只抓主干，避免一开始陷入 LoRA、多模态、分布式、spec decode 的分支。
4. 遇到 Python 复杂语法时，优先翻译成 C++ 风格的对象、方法、状态机、队列来理解。
5. 每次阅读都要产出面试表达，而不是只摘抄源码注释。

## 2. Phase A：Serving 主链路总览，10–12 小时

### 目标

建立 vLLM V1 的全局地图，能讲清楚一次请求从用户 API 进入，到 engine 内部执行，再到输出返回的大致路径。

### 必读源码

- `vllm/entrypoints/llm.py::LLM`
  - `generate`
  - `_add_request`
  - `_run_engine`
- `vllm/entrypoints/openai/api_server.py`
- `vllm/entrypoints/openai/*/serving.py`
- `vllm/engine/arg_utils.py::EngineArgs`
- `vllm/config/vllm.py::VllmConfig`
- `vllm/v1/engine/llm_engine.py::LLMEngine`
  - `from_engine_args`
  - `add_request`
  - `step`
- `vllm/v1/engine/async_llm.py::AsyncLLM`
  - `generate`
  - `add_request`
  - `_run_output_handler`

### 要回答的问题

1. offline `LLM.generate()` 和 online OpenAI server 最后都会进入哪条 engine 链路？
2. `EngineArgs` 和 `VllmConfig` 的关系是什么？为什么 vLLM 不在各模块里直接读 CLI 参数？
3. `LLMEngine` 和 `AsyncLLM` 分别适合什么场景？
4. 用户输入在进入 engine 前被包装成了哪些对象？
5. 同步 engine 的 `step()` 和异步 engine 的输出 handler 在职责上有什么差别？

### 面试表达

> vLLM 的入口层主要负责把用户 API 请求、OpenAI 请求或者 offline generate 请求统一转换成 engine 内部请求。真正的推理主循环不在 API server 里，而是在 V1 engine 里。配置通过 `EngineArgs` 聚合成 `VllmConfig`，再传给 engine、scheduler、executor 和 worker，避免各个模块直接依赖 CLI 参数。

### 建议笔记

- `docs/code-reading/01_serving_entrypoints.md`
- 图：offline / online 请求如何汇入 V1 engine。

## 3. Phase B：EngineCore 与请求生命周期，12–15 小时

### 目标

理解 vLLM 的 engine 内核如何接收请求、维护请求状态、驱动 scheduler 和 executor，并把模型执行结果更新回请求。

### 必读源码

- `vllm/v1/engine/input_processor.py::InputProcessor`
- `vllm/v1/engine/output_processor.py::OutputProcessor`
- `vllm/v1/engine/core_client.py::EngineCoreClient`
  - `InprocClient`
  - `MPClient`
  - `SyncMPClient`
  - `AsyncMPClient`
- `vllm/v1/engine/core.py::EngineCore`
  - `__init__`
  - `_initialize_kv_caches`
  - `add_request`
  - `preprocess_add_request`
  - `step`
  - `post_step`
- `vllm/v1/request.py::Request`
- `vllm/v1/request.py::RequestStatus`
- `vllm/v1/outputs.py::ModelRunnerOutput`

### 要回答的问题

1. `EngineCore` 初始化时做了哪些关键事情？
2. `EngineCoreClient` 为什么要区分 in-process 和 multiprocessing？
3. `EngineCore.step()` 一次迭代大致分成哪几步？
4. `Request` 里哪些状态用于描述 prompt、生成进度、KV cache、输出？
5. `ModelRunnerOutput` 如何反馈给 scheduler 和 output processor？

### 面试表达

> `EngineCore` 是 vLLM V1 的内核循环，它连接了 scheduler 和 executor。它不直接做模型计算，而是先让 scheduler 决定本轮执行哪些请求和多少 token，再把 `SchedulerOutput` 交给 executor/worker 执行，最后用 `ModelRunnerOutput` 更新请求状态并生成对外输出。

### 建议笔记

- `docs/code-reading/02_engine_core_request_lifecycle.md`
- 图：`LLMEngine.add_request()` 到 `EngineCore.step()` 的调用链。

## 4. Phase C：Scheduler 与 Continuous Batching，18–22 小时

### 目标

理解 vLLM 的 continuous batching 是如何由 scheduler 实现的：每一步动态混合新请求、prefill、decode、chunked prefill，并受 token budget 和 KV cache 约束。

### 必读源码

- `vllm/v1/core/sched/scheduler.py::Scheduler`
  - `__init__`
  - `add_request`
  - `schedule`
  - `_update_after_schedule`
  - `update_from_output`
  - `finish_requests`
  - `_preempt_request`
- `vllm/v1/core/sched/output.py::SchedulerOutput`
- `vllm/v1/core/sched/output.py::NewRequestData`
- `vllm/v1/core/sched/output.py::CachedRequestData`
- `vllm/v1/core/sched/request_queue.py`
- `tests/v1/core/test_scheduler.py`
- `tests/v1/core/test_scheduler_e2e.py`
- `tests/v1/core/test_priority_scheduler_random.py`

### 核心状态

重点观察 `Scheduler.__init__` 中的状态：

- `waiting`
- `skipped_waiting`
- `running`
- `requests`
- `finished_req_ids`
- `max_num_running_reqs`
- `max_num_scheduled_tokens`
- `encoder_cache_manager`
- speculative decoding 相关状态
- KV connector / EC connector 相关状态

### 要回答的问题

1. vLLM 的 continuous batching 和传统 static batching 的差异是什么？
2. `waiting` 和 `running` 分别存什么？请求何时从 waiting 进入 running？
3. Scheduler 每轮 `schedule()` 的输入是什么？输出是什么？
4. token budget 如何限制本轮能调度的 token 数？
5. prefill、decode、chunked prefill 如何共存？
6. 请求什么时候会被 preempt、finish、free？
7. `SchedulerOutput` 为什么是 scheduler 和 worker 之间的关键边界？

### 面试表达

> vLLM 的 continuous batching 本质上是每个 engine step 都重新做一次调度。Scheduler 维护 waiting/running 请求集合，根据 token budget、最大并发请求数、KV cache 可用块数和请求状态，决定本轮为每个请求推进多少 token。这样新请求不需要等整个 batch 完成，可以在 decode 过程中持续加入，从而提高 GPU 利用率和吞吐。

### 建议笔记

- `docs/code-reading/03_scheduler_continuous_batching.md`
- 图：waiting/running/request status 状态流转。
- 图：一次 `schedule()` 如何生成 `SchedulerOutput`。

## 5. Phase D：KV Cache、BlockPool 与 Prefix Caching，18–22 小时

### 目标

理解 vLLM 如何把 KV cache 管理成 block/page，如何分配、复用、释放和缓存 block，以及这些状态如何支撑 PagedAttention。

### 必读源码

- `vllm/v1/core/kv_cache_utils.py`
  - `KVCacheBlock`
  - `FreeKVCacheBlockQueue`
  - `get_kv_cache_configs`
  - `generate_scheduler_kv_cache_config`
  - `get_request_block_hasher`
  - `hash_block_tokens`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_manager.py::KVCacheManager`
  - `get_computed_blocks`
  - `can_fit_full_sequence`
  - `allocate_slots`
  - `cache_blocks`
  - `free`
  - `reset_prefix_cache`
- `vllm/v1/core/block_pool.py::BlockPool`
  - `get_cached_block`
  - `cache_full_blocks`
  - `get_new_blocks`
  - `touch`
  - `free_blocks`
  - `evict_blocks`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/worker/gpu/input_batch.py`
- `tests/v1/core/test_kv_cache_utils.py`
- `tests/v1/core/test_prefix_caching.py`
- `tests/v1/core/test_single_type_kv_cache_manager.py`

### 要回答的问题

1. KV cache 为什么不能简单按请求连续分配？
2. block/page 管理解决了什么显存碎片问题？
3. `KVCacheBlock` 的核心字段是什么？
4. `BlockPool` 如何管理 free blocks 和 cached blocks？
5. prefix caching 的 hash key 如何生成？命中后如何减少 prefill 计算？
6. `allocate_slots()` 为一个请求分配的到底是什么？
7. block table 和 slot mapping 分别服务于什么？

### 面试表达

> PagedAttention 的前提是把 KV cache 拆成固定大小的 block。Scheduler 不再为每个请求维护一段连续 KV，而是通过 block table 把逻辑 token 位置映射到物理 KV block。这样不同长度的请求可以共享统一的 block pool，完成的请求释放 block，prefix 相同的请求还可以复用 cached block，从而降低显存浪费并提升 prefill 效率。

### 建议笔记

- `docs/code-reading/04_kv_cache_block_pool_prefix_cache.md`
- 图：request tokens -> block hashes -> block table -> physical KV blocks。
- 图：prefix cache hit 与 recompute 的边界。

## 6. Phase E：GPU Worker 与 ModelRunner 执行链路，15–20 小时

### 目标

理解 scheduler 输出如何变成 GPU 上一次 forward 的输入，包括 worker 初始化、显存 profiling、KV cache 分配、batch tensor 准备、attention metadata 构造和模型执行。

### 必读源码

优先读新版 GPU runner：

- `vllm/v1/worker/gpu_worker.py::Worker`
  - `init_device`
  - `load_model`
  - `determine_available_memory`
  - `get_kv_cache_spec`
  - `initialize_from_config`
  - `compile_or_warm_up_model`
  - `execute_model`
  - `sample_tokens`
- `vllm/v1/worker/gpu/model_runner.py::GPUModelRunner`
  - `load_model`
  - `get_kv_cache_spec`
  - `initialize_kv_cache`
  - `profile_run`
  - `add_requests`
  - `update_requests`
  - `prepare_inputs`
  - `prepare_attn`
  - `execute_model`
  - `sample_tokens`
  - `postprocess`
- `vllm/v1/worker/gpu/input_batch.py::InputBatch`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/worker/gpu/states.py`
- `tests/v1/worker/test_gpu_model_runner.py`
- `tests/v1/worker/test_gpu_input_batch.py`

兼容旧路径时再对照：

- `vllm/v1/worker/gpu_model_runner.py::GPUModelRunner`

### 要回答的问题

1. `Worker` 和 `GPUModelRunner` 的职责边界是什么？
2. vLLM 如何通过 profiling 决定 KV cache 可用显存？
3. `SchedulerOutput` 到 `InputBatch` 之间做了哪些转换？
4. block table、slot mapping、attention metadata 在 forward 前如何准备？
5. `execute_model()` 的输入输出是什么？
6. CUDA graph / torch.compile 在这条链路上大致接在哪里？

### 面试表达

> Worker 更像设备侧执行环境，负责初始化 GPU、分布式环境、模型加载、显存 profiling 和 KV cache 分配。GPUModelRunner 则负责把 scheduler 的抽象调度结果转成具体 GPU tensor，包括 input ids、positions、block table、slot mapping 和 attention metadata，然后调用模型 forward，并在需要时执行 sampling。

### 建议笔记

- `docs/code-reading/05_gpu_worker_model_runner.md`
- 图：`SchedulerOutput` -> `InputBatch` -> attention metadata -> model forward -> `ModelRunnerOutput`。

## 7. Phase F：Attention、PagedAttention 与 CUDA Kernel 入口，12–16 小时

### 目标

把 Python attention layer、V1 attention backend、PagedAttention wrapper 和 CUDA kernel 入口串起来。这里不要求完整手推 kernel，但要能讲清楚 PagedAttention 为什么需要 block table、slot mapping，以及 kernel 大致消费哪些张量。

### 必读源码

- `vllm/model_executor/layers/attention/attention.py::Attention`
  - `forward`
  - `get_attn_backend`
  - `get_kv_cache_spec`
- `vllm/v1/attention/backend.py`
  - `AttentionBackend`
  - `AttentionMetadata`
  - `CommonAttentionMetadata`
  - `AttentionMetadataBuilder`
  - `AttentionImpl`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`
- `vllm/v1/attention/ops/paged_attn.py::PagedAttention`
  - `split_kv_cache`
  - `write_to_paged_cache`
- `vllm/_custom_ops.py`
  - `paged_attention_v1`
  - `paged_attention_v2`
  - `reshape_and_cache`
- `csrc/attention/paged_attention_v1.cu`
- `csrc/attention/paged_attention_v2.cu`
- `csrc/attention/attention_kernels.cuh`
- `tests/v1/attention/test_attention_backends.py`
- `tests/kernels/attention/`

### 要回答的问题

1. model layer 里的 `Attention` 为什么不直接写死某个 attention kernel？
2. attention backend 选择发生在哪里？
3. `AttentionMetadata` 包含哪些执行时信息？
4. `PagedAttention.write_to_paged_cache()` 写入的 key/value 和 `slot_mapping` 有什么关系？
5. `paged_attention_v1/v2` 的参数里，哪些来自 query，哪些来自 KV cache，哪些来自 block table？
6. CUDA kernel 层面为什么需要按 block 访问 KV？

### 面试表达

> vLLM 的 attention layer 是一个抽象入口，它根据平台、模型结构和配置选择具体 backend。PagedAttention 的核心不是把 KV cache 当成连续数组，而是通过 block table 间接访问分散的物理 KV blocks。写入 KV 时使用 slot mapping，把新 token 的 key/value 放到对应物理槽位；读取时 attention kernel 根据 block table 找到历史 token 的 KV block，从而支持动态 batch 和非连续 KV 管理。

### 建议笔记

- `docs/code-reading/06_attention_paged_attention_kernel.md`
- 图：`Attention.forward()` -> backend impl -> paged attention op -> `torch.ops._C` -> CUDA kernel。

## 8. Phase G：Sampling 与输出链路，6–8 小时

### 目标

理解模型 forward 后，vLLM 如何处理 logits、采样 token、计算 logprobs，并把结果返回给请求输出层。

### 必读源码

- `vllm/v1/sample/sampler.py::Sampler`
  - `forward`
  - `apply_temperature`
  - `greedy_sample`
  - `sample`
  - `compute_logprobs`
  - `apply_logits_processors`
  - `apply_penalties`
- `vllm/v1/sample/metadata.py::SamplingMetadata`
- `vllm/v1/sample/logits_processor/`
- `vllm/v1/sample/ops/topk_topp_sampler.py`
- `vllm/v1/sample/ops/penalties.py`
- `vllm/v1/engine/output_processor.py::OutputProcessor`
- `vllm/outputs.py::RequestOutput`
- `tests/v1/sample/test_sampler.py`
- `tests/v1/sample/test_topk_topp_sampler.py`
- `tests/v1/sample/test_logprobs.py`

### 要回答的问题

1. logits processor、penalty、temperature、top-k/top-p 的顺序大致是什么？
2. greedy sampling 和随机 sampling 的分支在哪里？
3. logprobs 是在哪一步计算和收集的？
4. sampling 输出如何进入 `ModelRunnerOutput`？
5. `OutputProcessor` 如何把 engine 内部输出转成用户可见的 `RequestOutput`？

### 面试表达

> vLLM 的 sampling 阶段发生在模型 forward 得到 logits 之后。Sampler 根据每个请求的 sampling params，对 logits 做 processors、penalty、temperature、top-k/top-p 等处理，然后采样下一个 token，并按需计算 logprobs。采样结果回到 engine 后，OutputProcessor 负责把内部 token / finish reason / logprobs 组装成对外的 RequestOutput 或流式输出。

### 建议笔记

- `docs/code-reading/07_sampling_output_processor.md`

## 9. Phase H：辅助模块选读，8–12 小时

这些模块不作为主线深挖。目标是知道它们解决什么问题、接在 serving 链路哪里、面试被问到时能给出系统定位。

### 9.1 Speculative Decoding，3–4 小时

阅读：

- `vllm/v1/spec_decode/`
- `vllm/v1/spec_decode/eagle.py`
- `vllm/v1/spec_decode/ngram_proposer.py`
- `vllm/v1/spec_decode/rejection_sampler.py`
- `tests/v1/spec_decode/`

重点问题：

- draft tokens 从哪里来？
- scheduler / model runner 如何处理 speculative tokens？
- rejection sampling 的作用是什么？

面试表达：

> Speculative decoding 用一个更便宜的 proposer 先猜多个 token，再由主模型验证，从而减少主模型 decode 步数。它不是替代 scheduler，而是给每个请求增加 draft token 状态，scheduler 和 model runner 需要能调度和验证这些 token。

### 9.2 Distributed / Parallel State，2–3 小时

阅读：

- `vllm/distributed/parallel_state.py`
- `vllm/distributed/communication_op.py`
- `vllm/config/parallel.py::ParallelConfig`
- `vllm/v1/executor/abstract.py::Executor`
- `vllm/v1/executor/multiproc_executor.py`
- `vllm/v1/executor/ray_executor.py`

重点问题：

- TP / PP / DP / EP 分别解决什么问题？
- executor 和 worker 如何对应多进程 / 多卡？
- `GroupCoordinator` 大致管理什么？

### 9.3 LoRA，1–2 小时

阅读：

- `vllm/lora/`
- `vllm/v1/worker/lora_model_runner_mixin.py`
- `vllm/v1/worker/gpu/lora_utils.py`

重点问题：

- LoRA request 如何进入 engine？
- 多 LoRA serving 的核心挑战是什么？

### 9.4 Structured Output / Multimodal / Metrics，2–3 小时

阅读：

- `vllm/v1/structured_output/`
- `vllm/multimodal/`
- `vllm/v1/metrics/`

重点问题：

- structured output 如何约束 decode？
- multimodal 如何影响 input processor、encoder cache 和 scheduler？
- metrics 如何观察吞吐、延迟、KV cache 使用？

## 10. 100 小时预算建议

| 模块 | 估时 |
| --- | ---: |
| Phase A：Serving 主链路总览 | 10–12h |
| Phase B：EngineCore 与请求生命周期 | 12–15h |
| Phase C：Scheduler 与 continuous batching | 18–22h |
| Phase D：KV cache / BlockPool / prefix caching | 18–22h |
| Phase E：GPU Worker / ModelRunner | 15–20h |
| Phase F：Attention / PagedAttention / kernel 入口 | 12–16h |
| Phase G：Sampling 与输出 | 6–8h |
| Phase H：辅助模块选读 | 8–12h |
| 合计 | 99h 上下 |

如果压缩到 80 小时：

- Scheduler 保留 18h。
- KV cache 保留 18h。
- GPU ModelRunner 保留 15h。
- Attention / kernel 只看 Python 到 CUDA 入口，压到 10h。
- 辅助模块压到 4h，只做定位。

## 11. 最终必须产出的面试材料

### 11.1 三张图

1. **Serving 主链路图**

```text
LLM / OpenAI API
  -> InputProcessor
  -> EngineCoreClient
  -> EngineCore
  -> Scheduler
  -> Executor / Worker
  -> GPUModelRunner
  -> Model forward + Sampler
  -> OutputProcessor
```

2. **Scheduler / KV cache / Worker 交互图**

```text
Scheduler.schedule()
  -> query KVCacheManager
  -> allocate / reuse blocks
  -> produce SchedulerOutput
  -> Worker.execute_model()
  -> ModelRunnerOutput
  -> Scheduler.update_from_output()
```

3. **PagedAttention block table 图**

```text
request logical tokens
  -> logical block ids
  -> block table
  -> physical KV cache blocks
  -> paged attention kernel reads K/V by block table
```

### 11.2 十五个面试问题

1. vLLM 的一次请求从 API 到 GPU forward 经历哪些模块？
2. `EngineCore` 的职责是什么？它为什么不直接做模型 forward？
3. continuous batching 相比 static batching 解决了什么问题？
4. Scheduler 每一步主要决策什么？
5. prefill 和 decode 在调度上的差异是什么？
6. chunked prefill 为什么能改善长 prompt 对 decode 的阻塞？
7. KV cache 为什么要分页 / 分块管理？
8. `BlockPool` 和 `KVCacheManager` 的职责边界是什么？
9. prefix caching 如何判断 cache hit？
10. block table 和 slot mapping 分别是什么？
11. PagedAttention kernel 为什么需要 block table？
12. `Worker` 和 `GPUModelRunner` 的职责边界是什么？
13. Scheduler 的输出如何变成 GPU tensor？
14. vLLM sampling 阶段如何从 logits 得到 token？
15. 如果吞吐低或显存不足，你会从哪些模块排查？

### 11.3 最终一句话总述

> vLLM 的核心是围绕 LLM serving 的动态请求调度和 KV cache 管理做系统优化：前端把请求统一转成 engine 内部请求，Scheduler 每一步基于 token budget 和 KV cache 状态做 continuous batching，KV cache 通过 block/page 管理和 prefix caching 降低显存浪费，GPUModelRunner 把调度结果转成 GPU tensor 并执行模型 forward，PagedAttention kernel 通过 block table 访问非连续 KV blocks，最后 Sampler 和 OutputProcessor 生成用户可见输出。

## 12. 阅读时的停止条件

每个模块不要无限深挖，满足下面条件就可以进入下一个模块：

1. 能画出这个模块在 serving 链路中的位置。
2. 能说出 3–5 个核心类 / 函数。
3. 能解释输入、输出和调用者。
4. 能回答本 roadmap 中列出的核心问题。
5. 能写出一段 1 分钟面试表达。

如果一个细节暂时讲不清楚，先记录在笔记的“待追问”部分，不要阻塞主链路推进。
