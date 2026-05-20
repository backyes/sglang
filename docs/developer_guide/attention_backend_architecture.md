# SGLang Attention Backend 设计综述

## 0. 分析范围与方法

本文不是简单罗列 `flashinfer`、`triton`、`fa3`、`trtllm_mla` 这些后端名称，而是从 SGLang 代码实际采用的抽象层次，梳理 attention backend 在运行时中的职责、边界和演化方向。

- 代码范围：重点通读 `python/sglang/srt/layers/attention/attention_registry.py`、`base_attn_backend.py`、`hybrid_attn_backend.py`、`flashinfer_backend.py`、`triton_backend.py`、`nsa_backend.py`，以及 `model_executor/model_runner.py`、`server_args.py` 中与后端选路和约束直接相关的逻辑。
- 文档范围：通读 `docs/advanced_features/attention_backend.md`，并结合 SGLang 现有 DeepSeek / speculative / hybrid attention 文档中的相关约束。
- 提交历史：基于 attention backend、FlashInfer、FA4、TRTLLM MLA、NSA 等关键词收敛出 137 条相关提交，再抽取其中能反映架构转折的节点。
- 行业参考：参考 FlashInfer 与 TensorRT-LLM 公开资料，仅用于解释为什么 serving attention 需要按 prefill / decode / append 拆分，以及为什么 page table、packed/ragged metadata、KV cache 形态会反过来塑造 backend 设计。

本文的核心判断标准只有一个：SGLang 的 attention backend 到底只是“一组可替换 kernel”，还是“一套面向 serving phase、KV 组织和模型结构的运行时协议”。通读代码后，答案显然是后者。

## 1. 总体结论：attention backend 在 SGLang 中是运行时协议，不只是 kernel 名称

如果只看 CLI，attention backend 似乎只是 `--attention-backend` 的几个字符串选项。但代码结构说明，它承担的是更重的职责。

- 它决定每次 forward 前需要准备什么 metadata。
- 它决定 decode、extend、mixed、target_verify、draft_extend 各 phase 如何调度具体 kernel。
- 它决定 KV cache 如何被索引，page table 是否原生支持，prefix cache / sliding window / cross attention / speculative decoding 是否可用。
- 它还要和 cuda graph、deterministic inference、DP attention、TBO、PD、multimodal、线性注意力包装层协同。

因此，最准确的概括不是“后端选择某个算子库”，而是：

> SGLang 的 attention backend 是一层 phase-aware、metadata-aware、KV-layout-aware 的 serving 抽象；kernel 只是这个抽象在某个平台上的一个落点。

## 2. 第一层：注册表把“后端家族”统一进一个运行时命名空间

`attention_registry.py` 是整个系统的第一层抽象。

### 2.1 统一注册，而不是散落分支判断

SGLang 用 `ATTENTION_BACKENDS` 注册表统一维护 attention backend 工厂。每个 backend 名称映射到一个构造函数，例如：

- `flashinfer`
- `triton`
- `fa3` / `fa4`
- `trtllm_mha` / `trtllm_mla`
- `flashmla` / `cutlass_mla`
- `ascend`
- `nsa`
- `dsv4`
- `intel_xpu` / `intel_amx`

这一步的意义不是“方便 import”，而是把 attention backend 明确提升为一个受控扩展点：

- 选路逻辑都走统一入口；
- 非法 backend 能在统一层面报错；
- 新 backend 能通过注册方式进入系统，而不是在各个模型/worker/scheduler 中手工塞分支。

### 2.2 注册的是“后端家族”，不是单个 kernel

从注册表可以直接看出，SGLang 的 backend 概念比底层 kernel 大得多。

- `flashinfer` 后端内部还会按 prefill / decode / wrapper type 再拆。
- `nsa` 后端内部还会选择不同 sub-backend，如 `flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`、`trtllm`。
- `fa3` / `fa4` 共用一部分实现，但因架构、page size、MLA/MHA 能力不同，被视为不同 family。
- `attn_backend_wrapper()` 甚至会在 full attention backend 外面再包一层 hybrid linear attention 适配器。

所以注册表统一的是“运行时后端家族”，不是单个裸 kernel 句柄。

## 3. 第二层：`AttentionBackend` 基类定义了真正的协议边界

`base_attn_backend.py` 是 attention backend 最关键的抽象。它暴露的不是一个 `forward()` 就完事，而是一整套 serving 协议。

### 3.1 metadata 初始化是第一公民

每个 backend 都必须实现：

- `init_forward_metadata()`
- `init_forward_metadata_capture_cuda_graph()`
- `init_forward_metadata_replay_cuda_graph()`
- `init_cuda_graph_state()`

这说明在 SGLang 里，attention backend 的首要职责不是直接算 attention，而是先把本轮 batch 映射成 kernel 能消费的数据布局。这和训练框架里“直接传 dense tensor 调一个 attention op”很不一样。

这也是 serving 语境的核心特征：

- 输入通常是 packed / ragged / paged，而不是规则 dense batch；
- 每轮 batch 都可能混有 decode、extend、verify、idle；
- 同一个 backend 还要在 graph capture / replay 两种状态下工作。

### 3.2 `forward()` 只是统一门面，内部仍按 phase 分流

基类默认 `forward()` 会根据 `forward_batch.forward_mode` 分发到：

- `forward_decode()`
- `forward_extend()`
- `forward_mixed()`

其中 decode / extend 的分离非常关键。行业公开资料也支持这一点：FlashInfer 和 TensorRT-LLM 都明确把 context/prefill 与 generation/decode 视为不同问题，因为前者更偏 compute-bound，后者更偏 IO-bound，最佳 kernel 结构往往不同。

SGLang 的抽象正是沿这个 serving reality 建立的，而不是强行把所有 phase 塞进一个 kernel API。

### 3.3 speculative verify 也被纳入 backend 协议

基类还提供：

- `get_verify_buffers_to_fill_after_draft()`
- `update_verify_buffers_to_fill_after_draft()`

这说明 speculative decoding 在 SGLang 中不是 attention backend 的“外部调用者”，而是反过来要求 backend 暴露 verify 所需 buffer 与 metadata 更新接口。attention backend 的职责已经扩展到了 speculative runtime 集成。

## 4. 第三层：`ModelRunner` 负责把用户配置变成 phase-aware backend 实例

attention backend 的实例化不在 scheduler，而在 `ModelRunner`。

### 4.1 `_get_attention_backend()` 是系统总入口

`model_runner.py` 里的 `_get_attention_backend()` 做了三件核心事情。

1. 如果当前是 draft worker 且显式设置了 `speculative_draft_attention_backend`，优先覆盖默认 backend。
2. 从 `ServerArgs.get_attention_backends()` 拿到 prefill / decode 两个 phase 的目标 backend。
3. 如果二者不同，则自动包成 `HybridAttnBackend`；否则直接实例化单一 backend。

这个入口意味着 backend 选择已经从“一个模型一个 backend”升级成“同一模型按 phase 可选不同 backend”。

### 4.2 `ServerArgs.get_attention_backends()` 很简单，但语义很强

`get_attention_backends()` 的实现本身很简单：

- `prefill_attention_backend` 若为空则继承 `attention_backend`
- `decode_attention_backend` 若为空则继承 `attention_backend`

但这个简单接口的工程意义很大。它把 phase-aware attention 选择正式编码进公共配置层，而不是只留给少数模型或少数 backend 特例使用。

### 4.3 attention backend 会被 TBO / PDMux / draft worker 再包装

`init_attention_backend()` 还会根据更上层运行时形态再套壳。

- `enable_pdmux` 时，会初始化多个 decode backend group。
- `enable_two_batch_overlap` 时，非 draft worker 会走 `TboAttnBackend`。
- hybrid GDN / mambaish 模型会在 `attn_backend_wrapper()` 里把 full attention backend 再包成 hybrid linear attention backend。

这说明 SGLang 的 backend 层不是“底层”，而是一个会被更多运行时调度策略继续复用的中层接口。

## 5. 代表性后端一：FlashInfer backend 展示了 serving-first 的完整设计

`flashinfer_backend.py` 很适合作为理解 SGLang attention backend 的代表。

### 5.1 它不是一个 kernel，而是一组 wrapper + workspace + dispatch 逻辑

`FlashInferAttnBackend` 内部维护：

- decode wrapper
- prefill wrapper
- paged / ragged 两类路径
- sliding window / cross attention 的 wrapper dispatch
- 全局 workspace buffer 与 override indptr buffer

这和行业资料一致。FlashInfer 本身就是围绕 prefill / append / decode 三种 serving phase 和 paged / ragged KV 组织设计的，因此 SGLang 对它的集成也自然偏向“backend runtime 适配器”，而不是“调一个单 kernel”。

### 5.2 workspace 是共享资源，不是后端内部小细节

FlashInfer backend 维护全局 `workspace_buffer`，并支持为某些运行时模式（如 PD-multiplexing）单独初始化 workspace。这反映了一个很现实的 serving 问题：高性能 attention 往往不是纯函数，还依赖外部临时内存、plan buffer 和调度状态。

因此，backend 抽象必须能承载资源生命周期，而不只是计算逻辑。

### 5.3 backend 必须理解模型形态和附加功能

FlashInfer backend 还会关心：

- 是否是 multimodal
- 是否 sliding window
- 是否 encoder-decoder
- 是否 deterministic inference
- 是否使用 MIS
- 是否 speculative decoding
- 是否需要 tensor cores

这说明后端并不是完全“模型无关”。在 serving runtime 里，backend 往往是模型结构、KV 形态、hardware 能力三者的交点。

## 6. 代表性后端二：Triton backend 展示了 SGLang 自研路径的抽象方式

`triton_backend.py` 是另一种重要代表。它没有像 FlashInfer 一样依赖外部完整库，而是把 decode / extend kernel 和 metadata 管理更多地收在 SGLang 自己手里。

### 6.1 Triton backend 直接把 metadata 视为核心资产

它维护的 `ForwardMetadata` 明确包含：

- `kv_indptr`
- `kv_indices`
- `qo_indptr`
- `custom_mask`
- sliding-window 专用索引与 offsets
- `num_kv_splits`
- `attn_logits` / `attn_lse`

这说明 Triton backend 的抽象重点不是包装第三方 API，而是把 SGLang 自己的 batch/KV layout 直接编译成 Triton kernel 需要的索引结构。

### 6.2 decode / extend 是不同 kernel 家族

Triton backend 初始化时就单独拿：

- `decode_attention_fwd`
- `extend_attention_fwd`
- `extend_attention_fwd_unified`

这和前文的总体判断一致。SGLang 不是把 attention phase 差异留给内部 if/else，而是在 backend 层就承认这些 phase 需要不同 kernel 族和不同 metadata 组织。

### 6.3 确定性与 split 策略是 backend 自己要承担的职责

Triton backend 会根据 deterministic inference、split tile size、static/dynamic kv splits、MLA context length 等条件调整 `max_kv_splits` 与调度策略。这说明 backend 不是被动执行者，它需要内置性能/正确性 heuristics。

## 7. 代表性后端三：NSA backend 展示了“backend 内再分 backend”的模型特化路线

`nsa_backend.py` 代表了另一种路线：某些模型的 attention 结构已经特殊到，单靠在通用 backend 上叠 patch 不够，需要一个模型专属 backend family。

### 7.1 `nsa` 不是通用 sparse backend，而是 DeepSeek DSA 运行时

代码里直接断言：`NSA backend only supports DeepSeek NSA`。这表明 `nsa` 的目标不是提供一个抽象的“任意 sparse attention”后端，而是针对 DeepSeek V3.2 DSA 这种特殊 attention 结构，提供 serving-ready 实现。

### 7.2 它内部再按 prefill / decode 选择 sub-backend

NSA backend 自身还有：

- `nsa_prefill_impl`
- `nsa_decode_impl`

可选项包括 `flashmla_sparse`、`flashmla_kv`、`fa3`、`tilelang`、`trtllm` 等。这意味着：

- SGLang 的 backend family 可以是树状结构，而不是平面结构；
- 用户看到的 `--attention-backend nsa` 只是第一层选择；
- 真正执行时仍要按 phase 和硬件再细分。

### 7.3 MTP、FP8、DP 等组合要求 backend 内部自带额外协议

NSA backend 里能直接看到对：

- MTP precompute
- FP8 KV cache
- deterministic inference
- page size 转换
- Triton / TRTLLM workspace

的处理。这说明模型特化 backend 一旦进入 serving runtime，就必须自己吸收这些系统复杂度，不能指望上层统一屏蔽掉所有差异。

## 8. HybridAttnBackend 体现了 SGLang 对 serving phase 差异的显式建模

`HybridAttnBackend` 是 attention backend 架构里最有代表性的设计之一。

### 8.1 prefill 和 decode 被正式视作不同优化目标

`HybridAttnBackend` 根据 `ForwardMode` 选择 backend：

- `decode_or_idle` 一律走 decode backend
- `target_verify` / `draft_extend` 根据 `speculative_attention_mode` 决定走 decode 还是 prefill backend
- 其他 prefill/extend 路径走 prefill backend

这说明 SGLang 已经把一个行业常识显式编码进架构里：

- prefill 通常更适合 compute-heavy kernel；
- decode 通常更适合 IO-optimized / paged / split-KV / generation-specialized kernel；
- speculative verify 与 draft extend 处于二者之间，必须单独指定语义归属。

### 8.2 hybrid 不是多余功能，而是 backend 体系成熟后的自然结果

当系统里同时存在：

- FlashInfer、FA3/FA4、TRTLLM MHA/MLA、Triton 等不同优势区间的后端；
- speculative、paged KV、chunked prefix cache、MLA、MHA、NSA 这些不同 workload；

那么“一个模型只用一个 attention backend”反而会变成限制。Hybrid backend 的出现说明 SGLang 已经从“选一个最好后端”进入“按 phase 组合最优后端”的阶段。

## 9. `ServerArgs` 中的兼容性规则说明 backend 选择本质上是约束求解

`server_args.py` 里大量逻辑并不是默认值填充，而是在做兼容性裁剪。

### 9.1 平台默认值只是第一步

SGLang 会根据设备自动设默认 backend，例如：

- CPU 倾向 `torch_native` 或 `intel_amx`
- NPU 走 `ascend`
- 某些 Blackwell / Hopper / ROCm 路径下按模型和平台设 `trtllm_mha`、`fa3`、`aiter`、`triton`
- DeepSeek MLA 在特定平台上会直接偏向 `trtllm_mla`

但这只是起点。

### 9.2 真正控制 backend 可用性的，是功能组合约束

代码和文档里都能看到大量“某功能要求某类 backend”的约束，例如：

- `enable_mis` 强制 prefill/decode 都是 `flashinfer`
- 某些 deterministic inference 只允许特定 backend 集合
- hybrid GDN 在 Blackwell / NPU 上要求特定 full-attention backend
- speculative topk 与 page size 的组合会限制可用 backend
- chunked prefix cache、FP8/FP4 KV、multimodal、sliding window、cross attention 都各自有支持矩阵

这说明 attention backend 选择本质上像一个约束求解问题，而不是单一优先级排序。

## 10. 行业对照：SGLang 的设计与主流 serving 共识一致，但更强调可组合性

行业资料能帮助解释 SGLang 为什么会这样设计。

### 10.1 为什么 prefill / decode 要分开

FlashInfer 公开材料明确指出：

- prefill/query length 较大时，attention 更偏 compute-bound；
- decode/query length 为 1 时，更偏 IO-bound；
- append/speculative verify 则介于两者之间。

TensorRT-LLM 也明确区分 context phase 和 generation phase，并针对 generation phase 使用专门的 masked MHA/XQA/multi-block 优化。

SGLang 的 `forward_decode()` / `forward_extend()`、`HybridAttnBackend`、以及 speculative attention mode，正是把这一 serving 共识做成了统一协议。

### 10.2 为什么 metadata 和 packed/paged KV 会成为 backend 中心

TensorRT-LLM 强调 packed tensor 和 paged KV cache，FlashInfer 强调 ragged/page-table/prefill-append-decode 一体化支持。这与 SGLang 的实践完全一致：真正复杂的不是 `QK^T` 本身，而是把当前 continuous batching 里的请求状态、page table、prefix cache 命中、speculative tree、sliding window 限制，编码成 kernel 可以高效消费的 metadata。

所以在 SGLang 中，attention backend 的核心工作往往是“metadata 编译”而不是“矩阵乘法调用”。

## 11. 提交历史时间线：从默认 FlashInfer 到 phase-aware / model-aware / hybrid-aware 架构

从 137 条相关提交里，可以看出 attention backend 体系经历了几个明显阶段。

### 11.1 2024 年中：从 `disable-flashinfer` 时代走向正式 backend 抽象

- 2024-07：默认开启 FlashInfer。
- 2024-09：引入 `--attention-backend`，并出现 “Refactor attention backend” 提交。

这一阶段的关键变化是：attention 从隐含实现细节升级成正式用户接口和运行时模块。

### 11.2 2024 年末到 2025 年初：基类协议与 speculative / graph 能力逐步补齐

- 增加 `torch_native` backend。
- 修复 Triton 的 cuda graph padding。
- 支持 target verification in attention backend。
- Triton backend 开始与 overlap scheduler 等运行时功能深度耦合。

这说明 backend 层开始承接 speculative 和 graph 语义，而不再只是普通推理 path。

### 11.3 2025 年上半年：MLA 后端家族成型

- FlashInfer MLA 支持 DeepSeek V3。
- prefix cache、ragged prefill、fast decode plan 等能力逐步进入 FlashInfer MLA。
- `flashinfer_mla` / `flashmla` / `cutlass_mla` / `trtllm_mla` 等 MLA family 逐渐展开。

这代表 attention backend 已从“平台分叉”扩展到“模型架构分叉”。

### 11.4 2025 年下半年：Hybrid attention 与 speculative / page-size 组合规则成熟

- 支持 prefill / decode 使用不同 attention backend。
- hybrid attention 支持 speculative decoding。
- 对 `topk>1` 与 `page_size>1` 的不兼容组合显式报错。
- 引入 speculator attention backend switch。

这一阶段说明 SGLang 已不再满足于单 backend 路径，而是在做 attention backend 的组合编排。

### 11.5 2025-2026：模型专属 backend family 与平台特化深化

- Ascend backend 成型。
- NSA / DSA backend 逐步完善。
- FA4、TRTLLM MLA/MHA、FlashInfer MXFP4、deterministic inference、PD-multiplexing workspace reuse 等能力陆续补齐。

这表明 SGLang 的 attention backend 体系正在从 “几个主流 kernel 后端” 演化为 “面向模型结构、平台能力和 serving phase 的可组合运行时架构”。

## 12. 结论：SGLang attention backend 的真正设计对象是 serving workload，而不是 attention 算法名称

如果只用一句话概括 SGLang 的 attention backend 设计，最准确的说法是：

> 它不是给用户选一个更快的 attention kernel，而是把 serving 中不同 phase、不同 KV 布局、不同模型结构、不同运行时约束，统一映射到一套可扩展的 backend 协议，再由具体 backend 家族去实现这些协议。

基于这个定义，可以把最容易混淆的几个问题说清楚。

- backend 不是单个 kernel：它通常是 metadata 管理、workspace 资源、phase 分发和 kernel 族的组合体。
- backend 不是单一全局选择：SGLang 已经正式支持 prefill/decode 分 phase 选型，并允许 speculative path 再细分。
- backend 也不是纯平台抽象：它还要承接模型结构差异，如 MLA、NSA、hybrid linear attention。
- 选 backend 不是简单 benchmark 排名：更像在 page size、KV dtype、speculative、prefix cache、graph、DP/PD、multimodal 等约束下求一个可运行且性能合理的组合。

从这个角度看，SGLang 的 attention backend 设计已经明显超出了“更换 kernel 实现”的层次，而是一套围绕 LLM serving 现实问题组织起来的运行时架构。

## 13. 参考锚点

- 代码：`python/sglang/srt/layers/attention/attention_registry.py`
- 代码：`python/sglang/srt/layers/attention/base_attn_backend.py`
- 代码：`python/sglang/srt/layers/attention/hybrid_attn_backend.py`
- 代码：`python/sglang/srt/layers/attention/flashinfer_backend.py`
- 代码：`python/sglang/srt/layers/attention/triton_backend.py`
- 代码：`python/sglang/srt/layers/attention/nsa_backend.py`
- 代码：`python/sglang/srt/model_executor/model_runner.py`
- 代码：`python/sglang/srt/server_args.py`
- 文档：`docs/advanced_features/attention_backend.md`
- 行业资料：FlashInfer “Accelerating Self-Attentions for LLM Serving with FlashInfer”
- 行业资料：TensorRT-LLM “Multi-Head, Multi-Query, and Group-Query Attention”