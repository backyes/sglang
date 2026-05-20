# enable_dp_attention 的设计与实现

## 0. 分析范围与方法

本文不是只读一个入口文件后的机制猜测，而是按 `enable_dp_attention` 的完整控制路径做交叉分析。

- 代码范围：通读 `python/sglang/srt` 下 `enable_dp_attention` 的直接命中路径，重点包括 `layers/dp_attention.py`、`managers/scheduler.py`、`managers/scheduler_dp_attn_mixin.py`、`managers/data_parallel_controller.py`、`model_executor/forward_batch_info.py`、`layers/communicator.py`、`model_executor/model_runner.py`、`model_executor/model_runner_kv_cache_mixin.py`、`disaggregation/common/conn.py`。
- 文档范围：通读 `docs/basic_usage/deepseek_v3.md` 中 DP attention 说明，并结合 SGLang v0.4 官方博客对该机制的公开解释。
- 提交历史：基于 `git log --grep='dp attention|enable_dp_attention|data parallel attention' -- python/sglang/srt docs/basic_usage/deepseek_v3.md` 收敛出 22 条直接相关提交，再抽查关键拐点提交的 diff/stat。

这里的核心判断标准只有一个：`enable_dp_attention` 到底是把历史 KV 在多个 DP worker 之间复制后再做并行，还是把请求与 KV 的拥有权切开，仅在需要跨 rank 聚合激活时通信。通读代码后，答案是后者。

## 1. 它解决的不是“更多 DP”，而是 DeepSeek 类模型上的注意力侧冗余

在常规 TP/DP 组合中，DP 更像是请求级复制：每个 DP 副本各自维护完整请求状态；TP 负责切分张量计算。这个模式在 DeepSeek MLA 类模型上有两个问题。

- 第一，KV cache 会随着 DP 副本数线性重复，占掉大量显存。
- 第二，decode 阶段真正重的未必是 dense MLP，而往往是长历史上的注意力读写；如果继续按“完整请求副本”复制，吞吐很快受 KV 容量与带宽约束。

`enable_dp_attention` 的设计不是再叠一层传统 DP，而是把“attention 看到的请求与 KV”按 DP 维切成多个 shard。每个 shard 持有不同请求集合的 KV 与调度状态；而在 attention 之后，如果后续 MLP 或 MoE 需要更完整的 token 视图，再通过显式 gather/scatter 把激活张量拼起来、用完后再切回去。

所以它的抽象边界是：

- 请求拥有权和 KV 拥有权：按 attention-DP 分片。
- 权重副本与执行进程：仍然按 scheduler / model runner 进程部署。
- 跨 rank 通信：主要围绕激活张量与批次元数据，而不是围绕历史 KV 的持续复制。

## 2. 拓扑重写：`dp_size` 被折叠进 attention world

`enable_dp_attention` 打开后，系统不再把 `dp_size` 当作“完全独立的一层请求副本”。相反，`layers/dp_attention.py` 中的 `compute_dp_attention_world_info()` 会把原来的全局 TP rank 重新解释成三层局部坐标：

- `attention_tp_rank` / `attention_tp_size`：attention 内部真正做张量切分的 TP 维。
- `attention_dp_rank` / `attention_dp_size`：请求与 KV 的拥有权分片维。
- `attention_cp_rank` / `attention_cp_size`：如果开启 CP，再叠一层 context parallel 切分。

这一步的工程含义很大。

- 对调度器来说，仍然会存在多个 worker/scheduler 进程，需要做负载均衡、端口分发、控制消息广播。
- 对 attention kernel 来说，当前 rank 只负责自己那一片请求与 KV，不再假设所有 DP rank 上都有同一份历史。
- 对后续 dense/MLP/MoE 来说，需要知道“本轮所有 attention-DP shard 各自产生了多少 token”，否则无法正确组织 gather/scatter 与 padding。

这也是为什么 `scheduler_dp_attn_mixin.py` 与 `forward_batch_info.py` 会额外维护 `global_num_tokens`、`global_num_tokens_for_logprob`、`dp_padding_mode`、`dp_local_start_pos` 这些字段。它们不是附属信息，而是 attention-DP 能正常组织 collective 的批次协议。

## 3. 控制面：先同步批次形状，再同步计算

`enable_dp_attention` 的第一层通信不是 hidden states，而是“这一轮每个 attention-DP rank 到底有没有活、活有多少”。

### 3.1 请求进入 scheduler 时的广播策略

`scheduler.py` 在 `enable_dp_attention` 路径下把输入拆成两类。

- work request：真正携带 prompt / decode 工作的数据，需要在 attention-TP 与 attention-CP 相关 rank 间广播，保证同一 shard 内的计算视图一致。
- control request：如停止、刷新、状态类消息。这里又支持 `enable_dp_attention_local_control_broadcast`，尽量只在本地相关 rank 间广播，避免每个控制消息都做一次全 TP 组同步。

这说明 DP attention 从一开始就不是“只改 kernel 的后端优化”。它连 scheduler 输入面都重写了，因为进程之间的职责边界已经变了。

### 3.2 `MLPSyncBatchInfo`：每轮先 all-gather 批次元信息

`scheduler_dp_attn_mixin.py` 的 `prepare_mlp_sync_batch_raw()` 会为每个本地 batch 抽出一份轻量元数据：

- 本 rank 的 token 数与 logprob token 数。
- 本地 forward mode 是 decode、extend、idle 还是 prebuilt。
- 是否可以跑 CUDA graph。
- 这轮 batch 是否包含 extend。

这些元信息通过 `MLPSyncBatchInfo.all_gather()` 在 DP 维上汇总成 `global_num_tokens` 等全局数组。随后 scheduler 再决定：

- 是否需要给空闲 rank 造一个 `idle batch`，仅用于参与 collective。
- 如果当前是 `prebuilt batch`，是否额外挂一个 `inner_idle_batch` 参与同步。
- 当前轮次是否适合跑 DP CUDA graph。

因此，DP attention 的一个关键工程约束是：哪怕某些 rank 这一轮没有真实请求，它们也不能直接“缺席”，否则 collective 的形状就断了。

## 4. 数据面：通信对象是激活张量，不是整段历史 KV

真正决定架构性质的是通信对象。

`layers/dp_attention.py` 和 `layers/communicator.py` 展示得很清楚：DP attention 的主要 collective 是围绕 hidden states 做 gather/scatter，而不是把已有 KV cache 周期性 all-gather 成一份全局副本。

### 4.1 gather 的两个模式

`dp_attention.py` 提供两类 gather 路径。

- `dp_gather_partial()`：把各 DP shard 的局部 hidden states 按 `global_num_tokens` 收集到全局缓冲区。常见于 attention 后、MLP 或 MoE 前。
- `dp_gather_replicate()`：把 gathered 结果复制到每个 rank 都可读的形式。使用场景更少，成本也更高。

底层实现又分成两类。

- `all_gather_into_tensor` 路径：要求各 rank token 数经过 padding 后可拼成规则张量。
- `all_reduce` 路径：借助预先布局好的分片缓冲区做求和式聚合，适合某些对称内存或已有布局。

### 4.2 scatter / reduce-scatter：聚合后立即切回本地 shard

在 `communicator.py` 中，attention 后如果进入 MLP/MoE，需要先 gather。计算完成后，又会走：

- `dp_scatter()`：按 `dp_local_start_pos` 和 `dp_local_num_tokens` 把全局结果切回本地 shard。
- `dp_reduce_scatter_tensor()` 或 `reduce_scatterv()`：在满足 padding/布局条件时，直接用 reduce-scatter 降低回传成本。

这也是 2025 年中后期多条提交不断优化的焦点。SGLang 后续不是在“让更多模型支持一个开关”而已，而是在持续把 DP attention 的回传通信从一般 scatter 改造成更省的 reduce-scatter 族实现。

### 4.3 `MAX_LEN` 与 `SUM_LEN`：通信模式由批次形状决定

`ForwardBatch.prepare_mlp_sync_batch()` 会为每轮选择 `DpPaddingMode`。

- `MAX_LEN`：把所有 DP rank 的 token 数 pad 到相同长度，便于 `all_gather_into_tensor` 与 `reduce-scatter`。代价是多做 padding。
- `SUM_LEN`：保留各 rank 实际 token 数，缓冲区按总长度组织，节省 padding，但 collective 组织更复杂。

这不是纯实现细节，而是 decode / extend / mixed batch 性能差异的来源之一。高批量下，多一点 padding 通常换来更规则的 collective；低批量下，collective 固定开销反而可能把收益吃掉。

## 5. Prefill：历史 KV 不跨 DP 合并，新增 KV 只写入本地拥有的 shard

用户特别关心的点是 prefill 时“历史 KV”和“追加 KV”到底怎么处理。这个问题必须把 extend 语义说开。

在 SGLang 里，prefill 不只是“第一次进来的整段 prompt”，还包括带前缀缓存的 extend。对一个 extend batch，系统会区分：

- `extend_prefix_lens`：已经存在的历史前缀长度。
- `extend_seq_lens`：本轮真正要追加计算的 token 长度。

在 DP attention 模式下，二者的处理不同。

### 5.1 历史 KV：留在本地 shard，不做跨 DP 常驻复制

历史 KV 来自两部分：

- 之前轮次已经写入本 rank KV pool 的老 token。
- prefix cache / radix / chunk cache 命中的前缀映射。

`model_runner_kv_cache_mixin.py` 在 `enable_dp_attention` 下会按 `dp_size` 缩减 `max_running_requests`、`max_mamba_cache_size` 等容量预算，说明设计假设是“每个 rank 只需要承接自己那片请求的 KV”，而不是每个 rank 都再存一份全量历史。

这和传统请求级 DP 的直觉相反。DP attention 不是把历史 KV 复制更多份，而是靠 attention-DP 分片把历史 KV 常驻显存需求降下来。

### 5.2 追加 KV：只为本 rank 拥有的请求生成并写入 `out_cache_loc`

本轮 extend/prefill 产生的新 token 会在本地 attention 计算后写入当前 rank 的 KV 位置，也就是 worker batch 中的 `out_cache_loc`。`forward_batch_info.py` 里对 extend batch 的处理仍然是本地构造 positions、prefix lens、seq lens；`prepare_mlp_sync_batch()` 只会额外组织 `global_num_tokens` 供后续 MLP/MoE 同步使用。

换句话说，新增 KV 的生命周期是：

1. scheduler 把请求路由到某个 attention-DP shard；
2. 该 shard 用自己的历史 KV 做 attention；
3. 该 shard 为新增 token 分配并写入本地 KV 槽位；
4. attention 后若后续层需要更完整 token 视图，再同步 hidden states，而不是反向回填历史 KV 到所有 shard。

这个边界很关键：DP attention 跨 rank 同步的是“当前轮的激活结果”，不是“把旧前缀和新追加 KV 拼成全局共享缓存”。

### 5.3 为什么 prefill 仍然需要空闲 rank 参与

即使某个 DP rank 本轮没有 extend 请求，它也可能被迫生成 idle batch 参与 collective。这不是因为它需要接收别人的历史 KV，而是因为 attention 之后的 MLP/MoE gather/scatter 仍然要求参与 rank 集合固定。

因此 prefill 阶段最容易误解的点是：

- attention 读写 KV：本地化。
- attention 之后的隐藏态同步：全局化。

两者是不同层次的通信，不能混为一谈。

## 6. Decode：每个 DP shard 独立增长自己的 KV，跨 rank 只同步当步激活

decode 阶段更能体现 DP attention 的真实价值。

SGLang 官方博客对外的描述是：每个 DP worker 可以独立处理 prefill、decode 或 idle batch；attention 处理后的数据在 MoE 前 all-gather，之后再重新分发。代码实现与这一直觉一致。

### 6.1 本地 decode 的 token 计数非常简单

在 `scheduler_dp_attn_mixin.py` 里，decode batch 的 `num_tokens = local_batch.batch_size()`。原因很直接：decode 每个请求每轮通常只追加一个 token，所以本 rank 这一轮要处理多少 token，本质上就是它拥有多少活跃序列。

接下来，各 rank 把自己的 `num_tokens` all-gather 成 `global_num_tokens`，供通信层决定 gather/scatter 的布局。

### 6.2 历史 KV 沿本地序列继续增长

decode 没有“把所有历史拿出来重算”的逻辑。每个 rank 上的每条活跃序列继续沿自己的 KV 历史追加 1 token：

- 本地序列的历史 KV 仍在本地池中。
- 本轮 decode 生成的新 KV 仍写回本地池中。
- 若后续 MLP/MoE 需要全局 token 视图，再做 hidden-state gather/scatter。

这正是它能减少 KV duplication 的根因。传统 DP 会让多个副本都背着同一批长历史；DP attention 则让每个 rank 只背自己那一片长历史。

### 6.3 Decode 吞吐提升为什么主要出现在大 batch

官方文档已经给出定性结论：DP attention 更适合高吞吐、大 batch decode，而不适合低延迟、小 batch。代码层原因也很明确。

- 优势项：KV 容量与带宽压力按 attention-DP shard 分散，更多请求可以常驻；长历史 decode 的注意力侧负担下降。
- 代价项：每轮都要额外同步 `global_num_tokens`，并在 MLP/MoE 边界做 gather/scatter 或 reduce-scatter；如果 batch 太小，通信与 padding 的固定成本无法摊薄。

所以它不是通用低延迟优化，而是高 batch、长上下文、MLA/MoE 模型上的吞吐优化机制。

## 7. MLP/MoE 边界才是通信热点

`communicator.py` 里的逻辑说明，DP attention 的通信热点并不在 attention 本身，而是在 attention 输出如何喂给后面的层。

### 7.1 Dense/LayerNorm 路径

`_gather_hidden_states_and_residual()` 在 `attn_dp_size != 1` 时会：

- 必要时先在更小的本地张量上做 layernorm；
- 把本地 hidden states gather 到全局 DP buffer；
- 之后再按需要 scatter 回本地。

这条路径直接证明通信对象是 hidden states/residual，而不是 KV cache。

### 7.2 MoE 路径

MoE 更典型。attention 后，系统先 gather token 对应的 hidden states，让后续专家路由有足够全局视图；MoE 完成后，再通过 `_scatter_hidden_states_moe()` 把输出切回各自 shard。若还叠加了 MoE CP，则会先在 CP 维切出本 rank 实际 token，再继续做 DP scatter。

因此，DP attention 与 DeepEP/EP 的真正耦合点在“token 激活怎么跨 rank 送到正确专家”，不是“KV 怎么在 rank 间共享一整份”。

## 8. 与 PD 解耦的关系：KV 传输语义也要降到 system-DP

DP attention 与 Prefill-Decode Disaggregation 不是天然兼容的，这一点从历史提交也能看出来。直到 2025-04-23 的 `e0673969`，SGLang 才明确补上“PD + DP attention + Mooncake”支持。

`disaggregation/common/conn.py` 中 `system_dp_size = 1 if enable_dp_attention else dp_size` 这一点很关键。它说明在 KV 传输协议里，系统不能再把 DP attention 当成普通多副本 DP 来处理；否则会错误地假定一个请求的 KV 需要按原 `dp_size` 扩散或索引。

更准确地说：

- 对调度层，`dp_size` 仍然存在，因为确实有多个 attention-DP shard 在跑。
- 对 KV 传输协议，`enable_dp_attention` 代表“请求的真实拥有者只有一个 shard”，所以 system-DP 语义必须收缩。

这和本文前面的主结论一致：DP attention 的主语义是分片拥有权，而不是多副本复制。

## 9. 资源与运行时副作用

### 9.1 显存预算按 shard 缩减

`model_runner_kv_cache_mixin.py` 会在 `enable_dp_attention` 时按 `dp_size` 缩减部分请求数与缓存容量预算。这不是 incidental optimization，而是架构声明：本 rank 不需要为别的 DP shard 的请求预留同等 KV 容量。

### 9.2 CUDA graph 约束更多

`server_args.py` 中 piecewise CUDA graph 会在 `enable_dp_attention` 下自动禁用。原因不难理解：DP attention 的 batch 形状、空闲 rank 补位、padding 模式切换都比单纯 TP 路径更动态，很难保持稳定图形状。

### 9.3 调度要关心跨 shard 均衡，而不只是总 token 数

2025-08-03 的 `f7b2853f` 引入 minimum token load balance，说明早期实现更偏“能跑通”，后续才逐步转向“减少 attention-DP shard 之间的 token 偏斜”。对这个架构来说，负载不均衡的代价不仅是慢 rank 拖住快 rank，还会直接放大 gather/scatter 的 padding 浪费。

## 10. 提交历史时间线：从 DeepSeek 专用能力走向通用通信后端

以下时间线只列出能代表设计转向的节点。

### 10.1 2024-12 到 2025-01：从可运行走向稳定

- `0ba2c589`：移除 dp attention 下 CUDA graph batch size 特判，说明当时已经有一套可跑实现，但 batch 形状与 graph 交互还不稳。
- `8b84e69f`：修复 TP token sync，说明早期问题集中在跨 rank token 计数一致性，而不是单个 kernel 正确性。

### 10.2 2025-03：从 `dp == tp` 走向 `1 <= dp < tp`

- `7f19e083`：支持 `1 <= dp < tp` 的 DeepEP 场景，这是拓扑层面的重要泛化。DP attention 不再被绑死在最特殊的 rank 切分关系上，而是正式进入“attention world 可重写”的阶段。

### 10.3 2025-04 到 2025-05：接入 PD 与更多模型

- `e0673969`：补上 PD + Mooncake 支持，说明 KV 传输协议需要为 attention-DP 的拥有权语义让路。
- `4bd2952a`：扩展到 Qwen2/3 MoE，说明这项能力从 DeepSeek MLA 特化路径开始向更普遍的 MoE 族迁移。

### 10.4 2025-08：进入调度与通信优化期

- `f7b2853f`：引入 minimum token load balance，优化 attention-DP 之间的工作分布。
- `4c22897a`：补上 qwen / llama4 的 dp attention padding reduce-scatter 支持，说明热点开始转向“如何更便宜地把 gathered 激活再切回去”。

### 10.5 2025-08 之后：持续补模型、补边角、补系统集成

后续提交集中在几类问题：

- 更多模型族的支持，如 Qwen dense、Llama4、MiniMax-M2.5。
- 与 HiRadixCache、spec decoding、VLM、Mooncake store 的兼容问题。
- 避免双重 reduce、修正端口分配、修复 launch crash 之类的系统集成问题。

这条历史说明 DP attention 已经从一个“DeepSeek 解码优化开关”，演化成 SGLang 分布式运行时中的一个长期维护子系统。

## 11. 结论：它本质上是“KV 拥有权分片 + 激活同步协议”

如果只用一句话概括 `enable_dp_attention`，最准确的说法不是“开启一种新的 DP”，而是：

> 它把请求和 KV cache 的拥有权沿 attention-DP 维切开，让每个 shard 只维护自己那部分长历史；然后在 attention 后、MLP/MoE 前后，通过 `global_num_tokens` 驱动的 gather/scatter 协议临时重建所需的激活视图。

基于这个定义，用户最容易混淆的三个问题都能回答清楚。

- Prefill 历史 KV：不在 DP rank 间常驻复制，留在本地 shard。
- Prefill 新增 KV：只写回本地 shard 的 `out_cache_loc`，不会为了同步 MLP/MoE 而变成全局共享 KV。
- Decode 过程：每个 shard 独立沿本地序列增长 KV，跨 rank 同步的是当步激活与批次元信息，因此收益主要体现为高 batch decode 吞吐，而不是小 batch latency。

从实现上看，`enable_dp_attention` 不是一个局部 kernel trick，而是同时改写了：

- scheduler 输入广播与 idle batch 语义；
- forward batch 的全局 token 协议；
- hidden-state gather/scatter 通信层；
- KV 容量预算与 PD 传输语义；
- 负载均衡与模型适配边界。

这也是为什么它值得被单独视作一套分布式执行架构，而不是“DeepSeek 的一个性能参数”。

## 12. 参考

- 代码：`python/sglang/srt/layers/dp_attention.py`
- 代码：`python/sglang/srt/managers/scheduler_dp_attn_mixin.py`
- 代码：`python/sglang/srt/model_executor/forward_batch_info.py`
- 代码：`python/sglang/srt/layers/communicator.py`
- 文档：`docs/basic_usage/deepseek_v3.md`
- 博客：https://lmsys.org/blog/2024-12-04-sglang-v0-4/