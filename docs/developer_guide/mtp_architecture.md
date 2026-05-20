# SGLang 中 MTP 的设计与实现综述

## 0. 分析范围与方法

本文沿着 MTP 在 SGLang 中的完整系统路径做综述，而不是只看某个模型文件或某个 speculative worker。

- 代码范围：通读 `python/sglang/srt` 中与 `MTP`、`NEXTN`、`FROZEN_KV_MTP`、`speculative_algorithm` 直接相关的控制路径，重点包括 `server_args.py`、`speculative/spec_info.py`、`model_executor/model_runner.py`、`managers/scheduler.py`、`managers/schedule_batch.py`、`speculative/multi_layer_eagle_worker*.py`、`speculative/frozen_kv_mtp_worker*.py`、若干 `*_mtp.py` 模型实现，以及与注意力/缓存/压缩耦合的运行时代码。
- 文档范围：通读 `docs/advanced_features/speculative_decoding.md`、`docs/basic_usage/deepseek_v3.md`、`docs/basic_usage/deepseek_v32.md` 中与 MTP 直接相关的说明。
- 提交历史：基于 `git log --grep='MTP|multi-token prediction|frozen-kv|nextn' -- python/sglang/srt docs test` 收敛出 79 条相关提交，再抽查关键里程碑提交。

本文的核心判断标准只有一个：MTP 在 SGLang 中究竟是“独立于 speculative decoding 的另一套框架”，还是“speculative decoding 内部的一种模型特化执行路径”。通读系统后，答案很明确，是后者。

## 1. 总体定位：MTP 不是平行子系统，而是 speculative decoding 的模型特化分支

SGLang 对外文档已经给出一个很重要的定性：MTP 是通过 speculative decoding 使用的。代码实现也完全沿着这个边界组织。

- 配置入口仍然是 `--speculative-algorithm`，而不是单独一组 `--mtp-*` 主开关。
- scheduler、KV 预分配、acceptance 统计、draft/verify 循环，都复用 speculative decoding 框架。
- 真正因模型而变的是 draft worker 的实现方式、draft 模型的加载方式，以及部分 attention backend / cuda graph / KV 视图的处理细节。

所以，如果要从分层架构理解 MTP，最合适的说法不是“它是一套新解码器”，而是：

> MTP 是 speculative decoding 框架中的一类 draft 源。不同模型把“draft 能力”嵌在目标模型内部，所以 SGLang 用不同的 worker 和模型包装器把这些内生 draft 能力接进统一的 speculative 调度/验证框架。

## 2. 第一层：配置入口先把 `EAGLE/NEXTN` 解析成具体 MTP 语义

`server_args.py` 的 `_resolve_speculative_algorithm_alias()` 是理解 MTP 入口的关键。

### 2.1 为什么用户经常只写 `--speculative-algorithm EAGLE`

在用户视角里，DeepSeek MTP 经常直接通过：

- `--speculative-algorithm EAGLE`
- 不显式提供 draft model，或由系统自动补全 draft path

来开启。这容易让人误以为 “MTP 就是 EAGLE”。实际上代码里不是这个关系。

- `NEXTN` 会被解析到 `EAGLE` 路径。
- 对 Gemma4 assistant 草稿模型，`NEXTN/EAGLE` 会被继续提升成 `FROZEN_KV_MTP`。
- 也就是说，`EAGLE` 在 CLI 层更多是一个兼容入口；真正选中的 worker 类型，要结合 draft 架构再决定。

### 2.2 自动补齐 draft 路径，说明 MTP 的 draft 常来自目标模型家族内部

MTP 与普通 STANDALONE speculative decode 的根本差异之一，是它的 draft 不是任意“小模型”，而通常是目标模型自带的 nextn/MTP 头，或者同家族导出的 draft 分支。因此历史提交里会出现：

- 自动设置 draft model path；
- 共享 target embedding 和 `lm_head`；
- 为特定模型重写 MTP 权重加载逻辑；

这些操作都说明 SGLang 把 MTP 视为“目标模型内生的 speculative 能力”，而不是“外接第二个独立 LLM”。

## 3. 第二层：worker 选型决定了 MTP 的真实执行形态

`speculative/spec_info.py` 的 worker 选择逻辑，是 MTP 架构最关键的分叉点。

### 3.1 三条主要分支

对 MTP 而言，当前系统里最重要的不是一个类，而是三类执行形态。

#### 3.1.1 多层 EAGLE 风格 MTP

当 speculative 算法仍走 `EAGLE`，并且 `enable_multi_layer_eagle=True` 时，会选择：

- V1：`MultiLayerEagleWorker`
- V2：`MultiLayerEagleWorkerV2`

这条路径本质上是“复用 EAGLE 的 draft/verify 合约，但 draft 模型不是普通外接 draft，而是多层 MTP 模块”。DeepSeek nextn、Step3.5 chain-style MTP 等都落在这一类附近。

#### 3.1.2 Frozen-KV MTP

当 draft 架构是 Gemma4 assistant 时，worker 会切到 `FrozenKVMTPWorker`。这是一条完全不同的实现哲学：

- assistant 读取 target KV；
- assistant 自己不扩展 KV；
- 仍复用 EAGLE verify 输入/输出契约；
- 但 seed、recurrent draft loop、位置处理、KV layer 映射都由 Frozen-KV MTP 单独掌控。

这说明 SGLang 已经不再把 MTP 视为单一“nextn 层”方案，而是允许不同模型选择不同的 draft-KV 关系。

#### 3.1.3 单层/专用 MTP 包装模型

Qwen3-Next、MiMo、Nemotron-H、EXAONE、Gemma4、Step3.5 等模型，都通过各自的 `*_mtp.py` 包装器，把 target trunk 外围再包一层 MTP 头或 assistant block，然后再挂入上述 worker 分支。

也就是说，SGLang 的 MTP 设计是两层组合：

- 上层 worker 决定 draft/verify 循环如何执行；
- 下层模型包装器决定“draft 能力”以什么结构暴露出来。

### 3.2 overlap scheduling 并不是所有 MTP 都支持

这里有个非常容易忽略的系统边界。

- 多层 EAGLE 风格 MTP 可以有 V1/V2 两套 worker。
- `FrozenKVMTPWorkerV2` 明确还是占位实现，直接抛 `NotImplementedError`。

因此，不能把 “MTP 支持 speculative v2” 当成统一结论。当前更准确的说法是：

- 一部分 MTP 路径可以进入 overlap scheduler；
- Frozen-KV MTP 暂时还不能。

## 4. 第三层：`ModelRunner` 把 MTP draft 看成模型的一个特殊切片

`model_executor/model_runner.py` 里有一段非常关键的逻辑：对于带 MTP 层的模型，如果当前进程是 draft worker，就用 `num_nextn_predict_layers` 决定 layer 数，而不是直接沿用 target 的全部 hidden/attention layer 数。

这背后的设计含义是：

- target 模型仍然是完整 trunk；
- draft worker 加载的是目标模型里的 MTP/nextn 子结构，而不是完整 target 的一份重复副本；
- 对部分模型，如 `MiMoV2MTP`、`Step3p5MTP`，draft 视图会进一步退化到单层包装；
- pipeline parallelism 明确不兼容 MTP 模型。

这也解释了为什么 SGLang 历史上反复修 MTP 的 weight loading、quantization、backend 兼容问题。因为这里不是“再起一个小模型”那么简单，而是要把模型内部的一部分层当成 draft 模块单独装配。

## 5. 统一执行框架：MTP 仍然遵守 speculative decode 的 draft -> verify -> accept 循环

虽然不同模型的 MTP 头形态不同，但执行骨架并没有脱离 speculative decode。

### 5.1 KV 预分配仍按 speculative decode 的 worst case 来做

`managers/utils.py` 的 `get_alloc_len_per_decode()` 明确说明：只要开了 speculative algorithm，decode 侧的 KV 分配长度就不再是 1，而要按：

- spec v1：draft decode 期间按 `topk * num_steps` 预分配；verify 期间按 `num_draft_tokens` 预分配；
- spec v2：按二者最大值预分配。

MTP 只是其中一种 speculative 算法，所以它天然继承了这套“先多分配、后回收”的运行时语义。

### 5.2 request 级别也按 speculative 路径管理过量 KV

`schedule_batch.py` 中请求对象提供了：

- `pop_committed_kv_cache()`
- `pop_overallocated_kv_cache()`
- speculative acceptance histogram

这说明系统从 request 生命周期层面就接受一个事实： speculative decode 会比最终真正接受的 token 更多地申请 KV。MTP 并没有绕开这件事，只是把“谁来提出 draft token”换成了内生 MTP 模块。

### 5.3 verify 阶段的输入输出契约仍然沿用 EAGLE 体系

即使是 Frozen-KV MTP，其 `FrozenKVMTPDraftInput`、`FrozenKVMTPVerifyInput` 仍然继承自 `EagleDraftInput`/`EagleVerifyInput`。这件事很关键，它说明：

- tree verify 的输入输出协议在系统中已经足够稳定；
- MTP 并没有重写 speculative 的外部接口，而是在内部换了 draft 生成机制。

因此，MTP 在 SGLang 中更像是“speculative framework 的一个 draft backend 家族”。

## 6. 三种主形态：DeepSeek nextn、多层 chain MTP、Frozen-KV MTP

如果只把 MTP 说成“模型自带多 token 头”，是不够的。通读模型实现后，至少可以分出三种主形态。

### 6.1 DeepSeek / 早期 nextn：EAGLE 框架中的内生 draft 层

DeepSeek V3/R1 的第一波 MTP 支持，从 2025-02 的 “Support NextN (MTP) speculative decoding for DeepSeek-V3/R1” 开始。它的核心不是改 scheduler，而是：

- 增加 nextn draft 模型/导出逻辑；
- 把 nextn 接进 EAGLE worker；
- 后续再逐步补 flashinfer MLA、FP8 KV、AMD、B200、DeepSeek V3.2、NPU 等兼容层。

这里的设计哲学是：target 仍是 target；draft 来自同家族 nextn 模块；运行时尽量复用 EAGLE。

### 6.2 Step3.5 式链式多层 MTP：每一步消费前一步 hidden states

`step3p5_mtp.py` 清楚地定义了另一类更“链式”的 MTP：

- 第 0 个 MTP 层消费 target hidden states；
- 第 `i+1` 个 MTP 层消费第 `i` 个 MTP 层产生的 hidden states；
- `MultiLayerEagleDraftWorker` 通过 `chain_mtp_hidden_states` 标志，在 speculative steps 之间覆写 `forward_batch.spec_info.hidden_states`。

这类模型的重点不在 KV 共享，而在 hidden-state 递推链。也因此它天然更依赖多层 draft worker，而不是简单的一层 nextn 包装。

### 6.3 Gemma4 Frozen-KV MTP：assistant 直接借 target KV，不建草稿 KV 历史

Gemma4 是当前 MTP 体系里最不一样的一类。

- `Gemma4AssistantForCausalLM` 会构造 `FrozenKVMTPContext`，把 assistant 逻辑层映射到 target 物理层；
- 每个 assistant layer 绑定 target 的 KV owner 层；
- `FrozenKVMTPWorker` 读取 target KV，但 assistant 自己不维护独立 KV 扩展路径；
- worker 注释里明确写了：它重用 EAGLE 的 verify 契约，但 draft 循环自己掌控，因为 assistant side 没有 KV extension。

这条路径的工程意义很大。它把 MTP 从“附加 nextn 层”进一步推广为“assistant 可以直接消费 target KV 视图”的通用设计。

## 7. hidden states 是 MTP 与 PD 耦合的关键介质

MTP 的数据流不只有 token 和 KV，还有 hidden states。

`schedule_batch.py` 里 `hidden_states_tensor` 的注释写得很直白：在 PD + MTP 场景下，使用 tensor 而不是 Python list 来传 `hidden_states`。这说明在分离式部署下，MTP 的跨节点/跨进程状态不只是 token 序列，还包括后续 draft 步骤要继续使用的隐藏状态。

`scheduler.py` 在 disaggregation 初始化时也留了一个非常关键的注释：

- 当前默认假设 MTP 只在 decode node 上开；
- 因而不主动在 prefill/decode 之间传 draft KV；
- 但这件事本身被显式标注为潜在问题。

这说明 MTP 与 PD 的耦合点至少有两层：

- target KV 如何在 P/D 节点之间传；
- MTP draft 额外依赖的 hidden states 或 draft 相关索引如何传。

从后续历史提交也能看到，这一块在 2026 年仍在持续修正，尤其是 NSA disaggregation、PD accuracy、PD + DP + MTP 的组合问题。

## 8. MTP 与注意力后端、CUDA graph、压缩路径是深度耦合的

从提交历史和代码可见，MTP 的主要复杂度并不只在 speculative 控制流，而在“它把更多组合路径变成了真实生产路径”。

### 8.1 attention backend 不是被动兼容，而要单独适配

早期提交很快就补了：

- flashinfer MLA + nextn
- FlashMLA + MTP + FP8 KV cache
- DeepSeek V3.2 的 NSA backend + MTP
- trtllm_mla / triton 等不同后端下的 MTP 性能与正确性修复

这说明 MTP 对 attention backend 的要求不是“能 decode 就行”，而是必须支持：

- draft extend / verify 所需的 metadata 组织；
- topk>1 时的树状验证布局；
- 某些模型特定的 KV/conv/mamba 状态更新。

### 8.2 CUDA graph 也要分 prefill / decode / draft_extend 单独处理

围绕 MTP 的大量 bugfix 都在处理 CUDA graph：

- capture batch size 与 padding；
- draft extend graph；
- V1/V2 模式差异；
- PD + DP + MTP + GLM 这类复杂组合；

这意味着 MTP 不只是给现有 decode 图再加一个分支，而是要求图捕获系统理解 speculative 的多个子阶段。

### 8.3 某些压缩/缓存优化路径直接与 MTP 不兼容

`pool_configurator.py` 里明确禁止了 `SGLANG_OPT_USE_ONLINE_COMPRESS` 与 speculative decode 并用，因为在线压缩路径假设严格 forward-only，而 speculative decode 需要 rollback/replay。这个结论对 MTP 同样成立。

因此，MTP 对系统的要求不是“更少的逻辑”，而是“对回滚、重放、额外状态视图更敏感”。

## 9. 与并行体系的关系：不是完全独立，但兼容边界清晰

### 9.1 PP 不兼容

`model_runner.py` 里已经通过断言给出明确边界：PP 与 MTP 模型不兼容。原因也很自然，MTP 的 draft 路径要求 target 和 draft 之间共享更紧密的层/hidden/KV 关系，而不是在 PP 边界上自由切开。

### 9.2 DP attention、CP、EP 可以逐步兼容，但需要逐项补洞

提交历史里能看到一条很典型的扩张轨迹：

- 先支持基础 TP8/单机 DeepSeek MTP；
- 再逐步补 DP attention、AMD、NPU、FlashMLA、FP8/NVFP4；
- 然后继续修 CP compatibility、EP MoE、two-batch-overlap、radix cache conflict、NCCL all-gather hang。

这说明 MTP 不是天然独立于并行系统，而是会把每个已有并行/后端组合重新走一遍。

### 9.3 大小 batch 的收益点也与模型和后端相关

DeepSeek 文档给出的经验结论是：

- 小 batch 下，MTP 速度提升尤其明显；
- 大 batch 下，也能受益，但需要调大 `max-running-requests`、补充 `cuda-graph-bs`；

这与前文分析一致。MTP 的收益来自减少 target 串行 decode 轮数，但代价是更多 speculative bookkeeping、verify、metadata 和 graph 维护。因此它不是“无条件越大 batch 越快”的单调优化，而是强依赖模型结构、后端和图捕获配置。

## 10. 系统视角下的控制流

把上述层次合并起来，可以把一次 MTP decode 轮次概括为下面的系统流程。

1. 用户通过 `--speculative-algorithm` 开启 speculative decode；server args 根据 draft 架构把它解析成 `EAGLE` 或 `FROZEN_KV_MTP` 等内部语义。
2. `spec_info.py` 根据 speculative 算法和 overlap 配置选择具体 worker：`MultiLayerEagleWorker(V2)`、`FrozenKVMTPWorker`、或其他 EAGLE 分支。
3. `ModelRunner` 按 MTP 模型的结构加载 target 与 draft 视图，必要时只加载 nextn 层或单层 assistant 包装。
4. decode 前，系统按 speculative worst case 预分配 KV，并准备 request 级回收边界。
5. draft worker 运行一轮或多轮 draft：
   - 多层 MTP 可能递推 hidden states；
   - Frozen-KV MTP 则直接读取 target KV；
   - 生成 verify 所需的 draft tokens / tree 结构 / bonus token 输入。
6. target worker 进行 verify，计算可接受的 draft token 数，并产出下一轮 draft 输入。
7. scheduler/request 状态更新 committed KV、overallocated KV、acceptance histogram；若涉及 PD，还需同步 hidden states tensor 与相关 metadata。

整个过程里，MTP 并没有摆脱 speculative decode 的控制骨架；它改变的是 draft 的来源、运行路径和状态依赖。

## 11. 提交历史时间线：从 DeepSeek nextn 到多模型、多后端、多形态 MTP

从 79 条相关提交中，可以读出比较清晰的五个阶段。

### 11.1 第一阶段：把 DeepSeek nextn 接进 speculative 框架

- `862dd76c`：首次支持 DeepSeek-V3/R1 的 NextN(MTP) speculative decoding。
- `9fafa62d`：共享 target embed 和 head，说明 draft 与 target 的关系不是完全独立模型。
- `9fb48f95`：补 flashinfer MLA 支持。

这一阶段的主题是“能把 DeepSeek 的 nextn 层接进 EAGLE 路径并跑起来”。

### 11.2 第二阶段：从能跑通走向多平台、多后端兼容

- `af6535e7`：AMD 支持。
- `2e4babdb`：FlashMLA + MTP + FP8 KV。
- 多条提交修正 fp4、MoE backend、two-batch-overlap、weight loading、router gemm 等问题。

这一阶段说明 MTP 真正难的不是算法定义，而是把它并入已有后端矩阵。

### 11.3 第三阶段：DeepSeek V3.2 / NSA 把 MTP 推进第二波

- `efa47334`：DeepSeek V3.2 MTP 支持。
- 后续提交继续清理 V3.2 的 V1/V2、flashmla_auto、NSA metadata、CP compatibility。

这表明 MTP 在 SGLang 中已经不再是单一 DeepSeek-V3 nextn 特性，而开始和新一代 attention backend、长上下文/稀疏路径一起演化。

### 11.4 第四阶段：从 nextn 扩张到更多模型族

- Qwen3-Next/Qwen3.5：逐步支持 MTP、spec_v2、DP、NCCL 修复。
- Step3.5：标准 multi-layer chain MTP。
- Gemma4：引入 Frozen-KV MTP。
- Nemotron-H、LongCat、GLM-5 等也逐步进入支持矩阵。

这代表系统从“为单模型加特例”过渡到“抽象出多种 MTP 执行形态”。

### 11.5 第五阶段：进入系统集成与组合爆炸治理期

2026 年的大量提交集中在：

- PD + DP + MTP 组合；
- radix cache conflict；
- NPU/AMD/B200/TRTLLM/FP4/NVFP4 等硬件与量化路径；
- hidden state / KV 传输遗漏；
- overlap scheduler 与 cuda graph 的组合问题。

这说明 MTP 已经从“一个模型特性”变成 SGLang runtime 的长期组合复杂度来源之一。

## 12. 结论：MTP 的本质是“统一 speculative 骨架上的多形态内生 draft 架构”

如果只用一句话概括 SGLang 中的 MTP，最准确的说法是：

> MTP 不是独立于 speculative decoding 的另一套解码器，而是把模型内生的 draft 能力通过不同 worker 和模型包装器接进统一 speculative 骨架的一组架构。

基于这个结论，可以把几个最容易混淆的问题说清楚。

- MTP 不是简单等于 EAGLE：CLI 常复用 `EAGLE` 入口，但内部可以分流到多层 EAGLE worker 或 Frozen-KV MTP worker。
- MTP 不是统一模型形态：至少存在 nextn 单层/少层、链式多层 MTP、Frozen-KV MTP 三类主实现。
- MTP 不是只改 draft 头：它同时影响 worker 选型、weight loading、KV 预分配、hidden state 传递、attention backend、cuda graph、PD/DP/CP/EP 兼容。
- MTP 也不是“天然全兼容”：PP 明确不兼容，Frozen-KV MTP 目前不支持 spec v2，PD/DP/CP 等组合仍在持续补洞。

因此，从全系统角度看，MTP 在 SGLang 中最值得强调的不是“能一次猜多个 token”，而是它如何把“模型内部的 draft 能力”规范化接入了 speculative runtime，同时允许不同模型保留各自最自然的 draft-KV-hidden-state 关系。

## 13. 参考锚点

- 代码：`python/sglang/srt/server_args.py`
- 代码：`python/sglang/srt/speculative/spec_info.py`
- 代码：`python/sglang/srt/model_executor/model_runner.py`
- 代码：`python/sglang/srt/speculative/multi_layer_eagle_worker.py`
- 代码：`python/sglang/srt/speculative/multi_layer_eagle_worker_v2.py`
- 代码：`python/sglang/srt/speculative/frozen_kv_mtp_worker.py`
- 代码：`python/sglang/srt/speculative/frozen_kv_mtp_info.py`
- 代码：`python/sglang/srt/models/step3p5_mtp.py`
- 代码：`python/sglang/srt/models/gemma4_mtp.py`
- 代码：`python/sglang/srt/models/qwen3_next_mtp.py`
- 文档：`docs/advanced_features/speculative_decoding.md`
- 文档：`docs/basic_usage/deepseek_v3.md`
- 文档：`docs/basic_usage/deepseek_v32.md`