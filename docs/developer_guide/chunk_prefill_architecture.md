# SGLang 分层架构与 Chunk Prefill 设计报告

本文从 SGLang 的运行时分层架构出发，分析 chunk prefill 的设计目标、控制路径、数据路径、缓存语义与演进历史。内容基于静态代码分析，并结合 `git log` 中与 chunk prefill、mixed chunk、pipeline parallelism、dynamic chunking 相关的提交记录整理而成。

## 0. 重审范围与方法

这次重审比第一版更严格，分成两部分执行：

1. 通读与 runtime、scheduler、pipeline parallelism、server args 相关的现有文档和核心入口代码。
2. 按时间顺序通读 chunk prefill 相关提交序列，并对关键拐点提交展开 `git show --stat` 复核。

需要说明的是，当前仓库总提交数为 12439。逐 patch 通读整个仓库全部提交超出单轮可验证范围，也会引入大量与本主题无关的噪声。因此，本次“通读所有 commit”的严格范围收敛为：

- 路径范围：`python/sglang/srt` 与 `docs/advanced_features/pipeline_parallelism.md`
- 主题范围：`chunk`、`prefill`、`mixed chunk`、`dynamic chunk`、`pipeline parallel`

在这个范围内，实际命中的相关提交共有 344 条，本文已完整通读这 344 条提交序列，并额外复核了以下关键拐点提交的 diff/stat：

- `7cd4f244a42178d0cdfb6a81156f38e87a7d92cd` `Chunked prefill (#800)`
- `c020f9cedafe1b3eb0c4575c9a6d394e05fc8277` `Support chunked prefill when radix cache is disabled (#811)`
- `f724f1f1e99406a120874de2579e671f304ca58c` `PrefillAdder abstraction (#968)`
- `3694f8f996e25c862cd67057e2bfa5844900fc98` `Mixed style of chunked prefill (#1013)`
- `11383cec3c08e7912c4398838e33eafe529e1732` `[PP] Add pipeline parallelism (#5724)`
- `c01b2ee0940a6395f929523d0eadbaee60800e92` `[PP] Refactor PP to async mode (#11852)`
- `ee1ca51de891bd6d9c913e0613a0f1aa8c54b1c3` `[PP] Fix dynamic chunking strategy for PP (#15372)`
- `c67569491c747bd17a31157325c3880d84aacea7` `Ensure chunked request extension length respects both rem_chunk_tokens and rem_total_tokens limits (#10003)`
- `722e25a621300b9f6d8f16d51ad19eadc67626a1` `Fix SWA eviction boundary and page-align chunked prefill (#22470)`
- `50fc2c9e2308114b40d507ca8446e2c3bdd3842e` `Fix hybrid swa chunked prefill oom (#23174)`
- `5f07ff92717ce7eba230ca7fc0ffab44ed255fe8` `Added the prefill delayer policy (#17456)`
- `560829a171c033b6bc1c89fb6ab31535bf638b03` `feat(scheduler): add adaptive queue-based prefill delayer trigger (#23189)`

## 1. 结论先行

SGLang 的 chunk prefill 不是一个孤立优化，而是贯穿配置层、调度层、执行层和缓存层的联合设计。

它解决的核心问题有三个：

1. 长上下文 prefill 一次性做完时，激活内存和 KV 分配峰值过高，容易压缩并发度。
2. 连续批处理场景下，纯 prefill 批次会阻塞 decode 批次，吞吐和时延都不稳定。
3. 在 PP 和 PD 场景下，固定大 prefill 会形成 pipeline bubble，甚至放大跨阶段不均衡。

因此，SGLang 当前的设计可以概括为：

- 用 `chunked_prefill_size` 把单次 prefill 计算上限显式化。
- 用 `PrefillAdder` 把 chunk 约束落到统一的 token budget 模型上。
- 用 `Req.prefix_indices` 和 `Req.is_chunked` 把跨迭代 chunk 状态显式保存在请求对象里。
- 用 `Scheduler.chunked_req` 把“还有后续 chunk 的请求”从一次 forward 延续到下一轮调度。
- 用 `enable_mixed_chunk` 把 prefill 和 decode 混入同一轮次，避免 decode 完全饿死。
- 用 `enable_dynamic_chunking` 在 PP 模式下动态缩小后续 chunk，减轻 stage bubble。
- 用 `ChunkCache` / `SWAChunkCache` 处理 radix cache 关闭时的跨 chunk KV 续接问题。

## 2. SGLang 的分层架构

如果只看 serving runtime，SGLang 可以粗分成六层。

### 2.1 接入与配置层

职责：接收 CLI 参数、构造运行时配置、选择 HTTP 或 gRPC 等入口。

关键文件：

- `python/sglang/launch_server.py`
- `python/sglang/srt/server_args.py`
- `python/sglang/srt/entrypoints/http_server.py`

这一层定义了 chunk prefill 的外部开关：

- `--chunked-prefill-size`
- `--max-prefill-tokens`
- `--enable-mixed-chunk`
- `--enable-dynamic-chunking`
- `--prefill-max-requests`

其中，`python/sglang/srt/server_args.py` 负责：

- 声明参数。
- 选择默认值。
- 校验约束，例如 `chunked_prefill_size` 与 `page_size` 的整除关系。
- 根据设备内存、DP/PP 场景、模型能力对参数做二次调整。

这意味着 chunk prefill 的第一层语义不是“某个 scheduler 小技巧”，而是运行时资源模型的一部分。

### 2.2 Engine 编排层

职责：搭建运行时主进程与子进程，建立进程间通信边界。

关键文件：

- `python/sglang/srt/entrypoints/engine.py`

`Engine` 的代码把主架构拆成三个核心组件：

1. `TokenizerManager`
2. `Scheduler`
3. `DetokenizerManager`

其中：

- `TokenizerManager` 负责请求接入、分词、会话态与 API 交互。
- `Scheduler` 负责调度、batch 组装、模型执行、KV 生命周期管理。
- `DetokenizerManager` 负责输出 token 的反解码和回传。

chunk prefill 的核心逻辑几乎全部落在 `Scheduler` 所在层，但它的输入来自 `TokenizerManager`，输出则通过 detokenizer 和 API 返回给用户。

### 2.3 请求与调度控制层

职责：从 waiting queue 中选择请求，决定本轮 prefill/decode 如何组 batch，以及如何切 chunk。

关键文件：

- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/schedule_policy.py`
- `python/sglang/srt/managers/prefill_delayer.py`
- `python/sglang/srt/managers/scheduler_pp_mixin.py`
- `python/sglang/srt/managers/overlap_utils.py`

这一层是 chunk prefill 的控制中枢。

### 2.4 请求状态与 batch 表示层

职责：把请求对象变成一轮 forward 可消费的 batch 表示，并在轮次之间携带状态。

关键文件：

- `python/sglang/srt/managers/schedule_batch.py`

`Req` 是 chunk prefill 设计里的关键状态载体，尤其是下面几个字段：

- `prefix_indices`: 之前 chunk 或 prefix cache 已经命中的 KV 索引。
- `extend_input_len`: 本轮需要新做 prefill 的 token 数。
- `is_chunked`: 一个计数器，不是简单的布尔值。
- `fill_ids`: 当前轮次参与计算的输入 token 序列。
- `start_send_idx` / `tmp_end_idx`: PD disaggregation 下按 chunk 发送 KV 的边界。

### 2.5 模型执行与内核层

职责：把 batch 映射成 forward，调用 attention backend、MoE backend、CUDA graph、PP 通信等底层实现。

关键文件：

- `python/sglang/srt/model_executor/model_runner.py`
- `python/sglang/srt/model_executor/forward_batch_info.py`
- `python/sglang/srt/layers/attention/*`
- `python/sglang/srt/layers/moe/*`

chunk prefill 在这一层的关键作用不是“重新定义 attention 算法”，而是改变每轮 forward 的输入长度和 forward mode，从而影响：

- 激活峰值
- CUDA graph 适配
- 混合 decode/prefill 批次形态
- PP 中每个 micro-batch 的计算时长

### 2.6 缓存与内存层

职责：维护 prefix cache、token 到 KV 的映射、页式分配器和 SWA 场景下的附加预算。

关键文件：

- `python/sglang/srt/mem_cache/chunk_cache.py`
- `python/sglang/srt/mem_cache/cache_init_params.py`
- `python/sglang/srt/mem_cache/memory_pool.py`
- `python/sglang/srt/mem_cache/swa_memory_pool.py`

chunk prefill 的一个关键事实是：它既是调度问题，也是缓存续接问题。没有缓存层对跨 chunk KV 的正确保存，chunk prefill 只是“重复切片重算”，而不是连续 prefill。

## 3. Chunk Prefill 的设计目标

### 3.1 控制一次 prefill 的计算和内存峰值

`--chunked-prefill-size` 把一次 extend 的最大 token 数压到固定上限内。对 GPU 来说，这意味着：

- 单次激活显存更可控。
- 单次 `alloc_extend` 的页式 KV 分配更可控。
- `max_prefill_tokens` 和 `max_running_requests` 更容易保持平衡。

### 3.2 让 decode 不被长 prefill 完全阻塞

启用 `--enable-mixed-chunk` 后，scheduler 会把运行中的 decode batch 和新的 prefill chunk 混在同一轮次执行。调度预算会先扣掉 decode token，再决定本轮还能塞多少 prefill token。

### 3.3 为 PP 和 PD 场景提供更细粒度的调度单位

在 pipeline parallelism 中，长序列 prefill 若用固定 chunk，后续 chunk 的执行时间会随着 prefix 变长而变慢，导致 stage bubble。`--enable-dynamic-chunking` 就是在这个前提下引入的。

在 prefill/decode disaggregation 中，chunk 切分还直接决定 KV 传输是一次性交付还是逐 chunk 交付。

## 4. Chunk Prefill 的核心控制链路

下面按代码路径说明一次 chunked prefill 如何发生。

### 4.1 参数进入运行时

入口在 `python/sglang/srt/server_args.py`。

CLI 参数定义如下：

- `chunked_prefill_size`: 单个 chunk 的最大 token 数，`-1` 表示关闭。
- `max_prefill_tokens`: 单轮 prefill batch 的总 token 上限。
- `enable_mixed_chunk`: 是否把 decode 和 prefill chunk 混合。
- `enable_dynamic_chunking`: 是否只在 PP 模式下动态调整 chunk 大小。

这一层还会做几个重要修正：

- 某些不支持的模型或后端会主动禁用 chunk prefill。
- 启用 DP attention 时可能会缩小 `chunked_prefill_size`。
- 在 decode-only disaggregation 模式下，部分校验会被跳过。

### 4.2 Scheduler 初始化 chunk prefill 语义

`python/sglang/srt/managers/scheduler.py` 中的 `init_chunked_prefill()` 负责把 server args 变成 runtime 状态。

它做了几件事：

1. 读取并规范化 `self.chunked_prefill_size`。
2. 对不兼容场景直接关闭 chunk prefill，例如部分 multimodal + transformers backend 组合。
3. 初始化 `self.chunked_req = None`，表示当前没有“还没跑完的 chunked 请求”。
4. 初始化 `self.is_mixed_chunk`。
5. 如果是 PP 且启用了 dynamic chunking，则初始化 predictor。

这里有一个很重要的设计选择：

`Scheduler` 不只是“每轮临时判断是否要切块”，它显式维护了一个跨轮次存在的 `chunked_req`。这让 chunk prefill 成为 scheduler 状态机的一部分。

### 4.3 请求对象在每一轮重建输入态

在 `python/sglang/srt/managers/schedule_batch.py` 中，`Req.init_next_round_input()` 会：

1. 用 `origin_input_ids + output_ids` 重建 `fill_ids`。
2. 调用 `tree_cache.match_prefix()` 做 prefix 命中。
3. 更新 `prefix_indices`、`last_node`、`host_hit_length` 等字段。
4. 重新计算 `extend_input_len = len(fill_ids) - len(prefix_indices)`。

对 chunk prefill 而言，这意味着下一轮不是“接着上次的切片继续跑”，而是重新用最新 prefix 状态计算“这轮还剩多少未完成输入”。

### 4.4 PrefillAdder 是真正的 chunk 决策器

真正决定“当前请求是整段 prefill 还是截断成 chunk”的逻辑在 `python/sglang/srt/managers/schedule_policy.py` 的 `PrefillAdder`。

`PrefillAdder` 维护一组统一预算：

- `rem_input_tokens`: 本轮 prefill token 总预算。
- `rem_chunk_tokens`: 当前 chunk 的预算上限。
- `rem_total_tokens`: 从 allocator 和 cache 角度看，全局还能承受多少 token。
- `rem_swa_tokens`: SWA 模式下的附加预算。
- `rem_dllm_tokens`: diffusion LLM 的特殊预算。

这套设计的关键点在于，chunk prefill 不是单独的分支判断，而是复用统一 budget 机制。

#### 新请求进入时的逻辑

`add_one_req()` 的核心语义是：

- 如果 `rem_chunk_tokens is None`，说明 chunk prefill 没开，按普通 prefill 处理。
- 如果 `input_tokens <= rem_chunk_tokens`，说明这是最后一个 chunk 或本来就不需要切块，按普通 prefill 处理。
- 否则，把当前请求截断到 `trunc_len`，写回 `req.set_extend_input_len(trunc_len)`，同时把 `self.new_chunked_req = req`。

也就是说：

- 本轮先执行“当前能容纳的前一段”。
- 同一个 `Req` 对象会被标记为 `new_chunked_req`，在后续轮次继续完成剩余前缀。

#### 已经在切块中的请求

`add_chunked_req()` 处理 `Scheduler.chunked_req`。

它会用如下预算计算本轮剩余可运行长度：

- `min(rem_chunk_tokens, rem_total_tokens)`
- SWA 模式下还要额外保留一页给 `alloc_extend`

如果剩余长度不足以完整跑完，则截断并返回原请求对象，表示“后面还有 chunk”。
如果本轮刚好跑完，则返回 `None`，表示 chunk 生命周期结束。

### 4.5 Scheduler 在每轮 prefill 前后维护 chunk 生命周期

`python/sglang/srt/managers/scheduler.py` 中的 `_get_new_batch_prefill_raw()` 是关键主循环。

它的 chunk 逻辑可以简化为：

1. 如果已有 `self.chunked_req`，先调用 `init_next_round_input()`。
2. 再调用 `adder.add_chunked_req(self.chunked_req)` 计算当前轮次能继续跑多少。
3. 如果 waiting queue 中新来的请求也被截断，则 `adder.new_chunked_req` 会被设置。
4. 构造 `ScheduleBatch` 后，如果 `self.chunked_req is not None`，就执行 `self.chunked_req.is_chunked += 1`。

这里有两个设计细节很重要：

#### 设计细节一：当前实现默认只有一个“正在继续切块的请求”

从代码与注释看，`self.chunked_req` 是单值，而不是列表。`process_batch_result_prefill()` 里也明确写了“当前最多只有一个正在 chunked 的请求”。

这表明现在的设计不是“多个长请求同时做多段切块”，而是“一轮中允许多个 prefill 请求进入，但跨轮延续的 chunked 请求只有一个主跟踪对象”。这是实现复杂度和调度稳定性之间的折中。

#### 设计细节二：mixed chunk 会先扣 decode 预算

`PrefillAdder` 初始化时会收到 `num_mixed_decode_tokens`。这会同时影响：

- `rem_input_tokens`
- `rem_chunk_tokens`
- `rem_total_token_offset`

因此 mixed chunk 不是简单把两个 batch 拼起来，而是在 admission control 阶段就把 decode 成本纳入 prefill 预算里。

### 4.6 组装新 batch 并进入模型执行

调度完成后，`ScheduleBatch.init_new(...).prepare_for_extend()` 负责把 `Req` 列表变成 forward batch。

如果启用了 mixed chunk 且 running decode batch 非空，scheduler 会执行：

- `self.running_batch.prepare_for_decode()`
- `new_batch.mix_with_running(self.running_batch)`

这一步是 chunk prefill 从“调度策略”进入“实际 forward 形态”的分界点。

### 4.7 forward 完成后，未完成 chunk 不转入正常流式输出

`python/sglang/srt/managers/scheduler_output_processor_mixin.py` 中的 `process_batch_result_prefill()` 会区分普通 prefill 和未完成 chunk：

- 如果 `req.is_chunked <= 0`，说明这是最后一个 chunk 或非 chunked prefill，可以 append 首个输出 token、检查 finish、缓存 unfinished req 等。
- 如果 `req.is_chunked > 0`，则只做 `req.is_chunked -= 1`，并且把这个请求作为 `skip_stream_req`，避免过早流式返回。

这说明 chunk prefill 的语义是“prefill 还没真正完成”，所以不应让上层把它当成已经进入正常 decode 阶段。

## 5. Chunk Prefill 的状态载体与不变量

### 5.1 `Req.is_chunked` 是计数器，不是布尔值

这点非常重要。

它的用法是：

- scheduler 在某轮发现该请求后面还有 chunk 时，执行 `is_chunked += 1`
- output processor 在该 chunk forward 完成后，执行 `is_chunked -= 1`

因此它表达的是“还有多少个未被完整结算的 chunk 轮次”，而不是“这个请求是否曾经被 chunk 过”。

### 5.2 `Req.prefix_indices` 是跨 chunk 续接的关键桥梁

在 radix cache 开启时，`prefix_indices` 来自 prefix match。

在 radix cache 关闭时，`ChunkCache.cache_unfinished_req()` 会直接把当前请求已生成的 KV 索引复制回 `req.prefix_indices`。下一轮 `add_chunked_req()` 就依赖它来继续后续 chunk。

因此，`prefix_indices` 统一了承担两种来源：

- prefix cache 命中得到的已存在 KV
- 上一个 chunk 刚刚完成而尚未“正式成为完整请求前缀”的 KV

### 5.3 chunk 长度必须服从页对齐和特殊对齐约束

`PrefillAdder.add_one_req()` 在截断时不是直接用 `rem_chunk_tokens`，而是会：

- 先按 `page_size` 对齐。
- 如果启用了 deterministic inference 等特殊路径，还要按 `truncation_align_size` 继续向下取整。

这说明 chunk prefill 的真正单位不是“任意 token 数”，而是“与 allocator 和 backend 对齐约束兼容的 token 数”。

### 5.4 动态 chunking 只是在“每轮 chunk 上限”上覆盖默认值

`_get_new_batch_prefill_raw()` 在存在 `self.chunked_req` 且启用 dynamic chunking 时，会用 predictor 给出 `dynamic_size`，覆盖本轮 `chunked_prefill_size`。

这个设计很克制：

- 不改变 `PrefillAdder` 的基本预算模型。
- 只改变“本轮 rem_chunk_tokens 的初始值”。
- 因此不会污染普通 chunk prefill 的逻辑分支。

## 6. Chunk Prefill 与缓存层的关系

### 6.1 Radix cache 开启时

chunk prefill 会尽量复用 `match_prefix()` 的结果，让下一轮只处理还没有进入缓存的尾部 token。

### 6.2 Radix cache 关闭时

`python/sglang/srt/mem_cache/chunk_cache.py` 中的 `ChunkCache` 是专门的兜底实现。

它有两个显著特点：

1. `match_prefix()` 永远返回空，说明没有 prefix 匹配能力。
2. `cache_unfinished_req()` 会把当前请求已写入 KV pool 的索引复制到 `req.prefix_indices`，供下一轮 chunk 使用。

也就是说，chunk prefill 即便不依赖 radix tree，也仍然能成立；只是“跨 chunk 续接”从基于树的共享前缀，退化成基于请求本地状态的直接续接。

### 6.3 SWA 模式下的特殊预算

`PrefillAdder` 为 hybrid SWA 单独维护了 `_swa_budget_for_req()` 和 `rem_swa_tokens`。

原因很直接：

- chunk N 还在运行
- sliding window 的已锁定前缀也占空间
- chunk N+1 还要额外分配空间

因此 SWA 模式下，chunk prefill 的 admission control 比普通 KV allocator 更保守。这也是最近多次 bugfix 的集中区域。

## 7. Chunk Prefill 与混合调度、PP、PD 的耦合点

### 7.1 Mixed chunked prefill

mixed chunk 的本质不是“把两个 batch 简单拼接”，而是：

- decode 先占用一部分 token budget
- prefill 再用剩余预算切 chunk
- 最终在 `ScheduleBatch.mix_with_running()` 处合成一个 forward batch

这解释了为什么相关单测关注的是 budget 扣减，而不只是输出正确性。

### 7.2 Pipeline parallelism 下的 dynamic chunking

`docs/advanced_features/pipeline_parallelism.md` 已经把目标写得很清楚：

- 固定 chunk 大小时，随着 prefix 变长，后续 chunk 时间更长。
- 这会让不同 PP stage 的执行时长越来越不均衡。
- dynamic chunking 用二次函数拟合 prefill 累积耗时，然后根据当前 prefix 长度预测下一个 chunk 应该缩到多大，以维持近似相同的 chunk 执行时间。

从架构角度看，这不是“另一个调度器”，而是给现有 chunk prefill 注入了一个自适应 chunk size 预测器。

### 7.3 Prefill/Decode disaggregation

PD 模式下，chunk prefill 的另一个职责是控制 KV 发送边界。

`Req.start_send_idx` 和 `Req.tmp_end_idx` 的存在表明：

- prefill 端并不一定等所有 prefix 都做完才一次性发送 KV。
- 它可以按照 chunk 的完成边界逐步发送。
- 在 overlap schedule 下，KV 发送还要等到结果真正 ready 才能做。

所以在 PD 模式下，chunk prefill 同时是“计算切分单位”和“KV 传输切分单位”。

## 8. 代表性测试与它们验证的语义

从测试布局可以看出，SGLang 团队把 chunk prefill 当作一个调度和缓存语义特性，而不是单纯的数值正确性特性。

代表性测试包括：

- `test/registered/unit/managers/test_prefill_adder.py`
  - 验证 mixed chunk 下 decode token 如何扣减 prefill budget。
  - 验证 hybrid SWA 模式下必须额外保留一页给 `alloc_extend`。
  - 验证 SWA token 不足时应延后 chunk，而不是冒险 over-commit。

- `test/registered/scheduler/test_mixed_chunked_prefill.py`
  - 验证开启 `--enable-mixed-chunk --chunked-prefill-size 32` 后的整机推理路径。
  - 同时覆盖带 radix cache 和关闭 radix cache 的两种模式。

- `test/manual/scheduler/test_no_chunked_prefill.py`
  - 验证 `--chunked-prefill-size -1` 的退化路径仍然成立。

这组测试结构本身也说明了设计边界：

- 单元测试主要盯 budget 和 allocator。
- 集成测试盯运行时行为。
- 手工测试盯配置开关退化路径。

## 9. 从 Git 历史看设计演进

下面按时间主线整理与 chunk prefill 直接相关的关键提交和 PR 线索。这里的结论不是只看少量里程碑标题得出的，而是建立在 344 条相关提交序列完整通读的基础上，再用若干关键 patch 做锚点校正。

### 9.1 第一阶段：引入基本 chunked prefill

代表提交：

- `2ec39ab71` `Chunked prefill support (#797)`
- `98111fbe3` `Revert "Chunked prefill support" (#799)`
- `7cd4f244a` `Chunked prefill (#800)`
- `c020f9ced` `Support chunked prefill when radix cache is disabled (#811)`

这说明 chunk prefill 早期经历过快速试错：

- 先引入
- 再回滚
- 再重新落地
- 紧接着补齐“关闭 radix cache 时也能继续工作”的必要能力

换句话说，团队很早就意识到 chunk prefill 不能只依赖 radix tree 才成立。

### 9.2 第二阶段：把逻辑从散点条件抽成统一调度抽象

代表提交：

- `f724f1f1e` `PrefillAdder abstraction (#968)`
- `3694f8f99` `Mixed style of chunked prefill (#1013)`

这是一个重要拐点。

从 `f724f1f1e` 的 diff 规模看，核心变化不是小修补，而是把原来散落在 `tp_worker` 里的 prefill admission 逻辑集中抽到单独抽象里。也正因为这个抽象先被建立，后续的 mixed chunk、SWA、PP、prefill delayer 才有一个统一的预算入口可以挂接。

### 9.3 第三阶段：从单机 chunk prefill 扩展到 PP

代表提交：

- `11383cec3` `[PP] Add pipeline parallelism (#5724)`
- `c01b2ee09` `[PP] Refactor PP to async mode (#11852)`

这阶段 chunk prefill 的角色发生了变化：

- 早期主要是内存与连续批处理优化。
- 引入 PP 后，它同时变成 pipeline 微批粒度控制器。

从 `11383cec3` 和 `c01b2ee09` 的 diff 规模看，PP 不是在原有 scheduler 外面套一层壳，而是实质性重写了 scheduler、pp mixin、model runner 与 server args 的协作方式。`docs/advanced_features/pipeline_parallelism.md` 里对 dynamic chunking 的说明，基本可以视为这条演进线的架构注脚。

### 9.4 第四阶段：约束、兼容性与默认值调优

代表提交：

- `b01df48cf` `Adjust default chunked prefill size and cuda graph max bs according to GPU memory capacity (#2044)`
- `3bffe1127` `Fix chunked prefill size validation for disabled state (#8973)`
- `3795b6a43` `Skip chunked_prefill_size validation when disaggregation mode is decode (#10358)`

这些提交说明 chunk prefill 已经不是实验特性，而是正式运行时参数的一部分，因此默认值、禁用语义、与 disaggregation 的交互都被系统化处理。与此同时，`c67569491` 这类单文件小修补也很有代表性，它说明后期的主要问题已经从“是否支持 chunk prefill”转向“预算模型是否在所有角落都自洽”。

### 9.5 第五阶段：与 mixed chunk、spec、PD、SWA 深度耦合

代表提交：

- `731146f6c` `Fix mixed chunked prefill in overlap mode (#2158)`
- `e1ce44cdb` `Disabling mixed chunked prefill when eagle is enabled (#6874)`
- `58c6b871b` `Remove compatibility restriction between Pipeline Parallelism and Mixed Chunked Prefill (#22920)`
- `722e25a62` `Fix SWA eviction boundary and page-align chunked prefill (#22470)`
- `50fc2c9e2` `Fix hybrid swa chunked prefill oom (#23174)`
- `aea527afd` `Fix swa chunk req deferred (#24318)`

这说明最近两年的主要工作不再是“有没有 chunk prefill”，而是“它在各种高级调度和缓存场景下是否仍然正确”。从 `722e25a62`、`50fc2c9e2` 的 diff 可以看到，SWA 和大 page size 场景下的问题已经主要体现为边界条件与预算保守性，而不是主干逻辑缺失。

### 9.6 第六阶段：进一步把 chunk prefill 纳入更高层调度增强器

代表提交：

- `5f07ff927` `Added the prefill delayer policy ... (#17456)`
- `560829a17` `feat(scheduler): add adaptive queue-based prefill delayer trigger (#23189)`

prefill delayer 不是 chunk prefill 本身，但它已经进入同一条 admission control 路径。这说明 SGLang 正在把“何时发起 prefill、每轮给多少预算、是否让 decode 优先”统一收敛到一个更完整的调度层能力集合。`5f07ff927` 到 `560829a17` 之间的一串提交也显示，这条线在 2026 年已经从单纯的 delay 开关，演进到了带队列阈值和 wall-clock cap 的自适应触发器。

### 9.7 第七阶段：最近半年真正的重心是稳定化，而不是再造新主干

通读 344 条相关提交后，一个很明显的趋势是：2025 年下半年到 2026 年上半年，和 chunk prefill 直接相关的大部分提交都不再是“重写主流程”，而是围绕四类问题做密集收敛：

1. `page_size > 1`、SWA、HiCache、radix/chunk cache 之间的边界对齐。
2. PP async event loop 与 dynamic chunking 的阶段错配和 hang。
3. PD disaggregation 下 prefill 端、decode 端、KV 发送游标与 bootstrap 状态的一致性。
4. prefill observability、metrics、prefill delayer 策略和日志精度。

这意味着当前 chunk prefill 主干设计已经比较稳定。最近大量提交更多是在证明这条主干可以覆盖更多硬件、模型和调度组合，而不是要推翻最初的设计抽象。

## 10. 架构视角下的关键设计判断

### 10.1 为什么 `chunked_req` 放在 Scheduler，而不是放在 Req 自己内部队列里

因为 chunk prefill 的本质不是请求内部迭代，而是调度器对“本轮资源是否允许继续这个请求”的判断。

如果把后续 chunk 完全封装在请求对象内部，scheduler 就很难把 mixed decode token、SWA 预算、PP 动态 chunk size 放到统一 admission control 中。

### 10.2 为什么 `PrefillAdder` 是这个设计里最关键的抽象

因为它把多个原本可能分散在 scheduler 分支里的约束统一为 budget 问题：

- chunk budget
- prefill budget
- total token budget
- SWA budget
- DLLM budget
- prefill batch size limit
- prefill delayer 允许条件

这也是为什么 chunk prefill 后续可以自然吸纳 mixed chunk、PP 和 SWA，而不是每次新增一个完全平行的调度路径。

### 10.3 为什么当前只跟踪单个 `chunked_req`

这是当前实现里的一个显式折中。

它的好处是：

- 跨轮状态简单。
- output processor 行为简单。
- 与 overlap、PD、SWA 的组合爆炸被压住。

它的代价是：

- 多个超长请求同时分段推进的灵活性受限。
- chunk 间公平性更多依赖 waiting queue 的外层调度，而不是多个活跃 chunked 请求的细粒度交错。

从现有代码和测试来看，SGLang 明显优先选择了可维护性和正确性。

## 11. 对后续维护者最有价值的排查入口

如果后续要继续分析或修改 chunk prefill，建议优先从下面这些位置入手：

1. `python/sglang/srt/managers/scheduler.py`
   - 看 `init_chunked_prefill()` 和 `_get_new_batch_prefill_raw()`。

2. `python/sglang/srt/managers/schedule_policy.py`
   - 看 `PrefillAdder.add_one_req()`、`add_chunked_req()`、`_update_prefill_budget()`。

3. `python/sglang/srt/managers/schedule_batch.py`
   - 看 `Req.init_next_round_input()`、`set_extend_input_len()`、`is_chunked`、`prefix_indices`。

4. `python/sglang/srt/managers/scheduler_output_processor_mixin.py`
   - 看未完成 chunk 如何避免过早流式输出。

5. `python/sglang/srt/mem_cache/chunk_cache.py`
   - 看 radix cache 关闭时如何跨 chunk 续接 KV。

6. `docs/advanced_features/pipeline_parallelism.md`
   - 看 dynamic chunking 的目标函数和调优方法。

## 12. 总结

从分层架构角度看，SGLang 的 chunk prefill 设计有三个最值得注意的特征。

第一，它不是 attention kernel 优化，而是运行时调度与缓存协同优化。真正的控制点不在 backend 内核，而在 `Scheduler + PrefillAdder + Req + Cache` 这条链路上。

第二，它不是单一特性，而是 serving runtime 的基础能力。mixed chunk、PP dynamic chunking、PD disaggregation、SWA allocator 都是在这个基础能力上继续叠加语义。

第三，它的实现风格相当克制。SGLang 没有把 chunk prefill 做成多个并行复杂状态机，而是依靠：

- 一个 `chunked_req`
- 一个 `PrefillAdder`
- 一组统一预算
- 一个请求对象上的少数字段

来承载跨轮次 chunk 生命周期。这也是当前实现能够持续演进而没有失控的主要原因。

关于 SGLang 基础分布式架构的独立分析，见 `docs/developer_guide/distributed_architecture.md`。

