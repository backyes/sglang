# SGLang 基础分布式架构设计报告

本文单独分析 SGLang 的基础分布式架构，不与 chunk prefill 主题合并。重点不是某一个单独优化点，而是从全文代码和相关文档出发，解释 SGLang 如何把配置参数、进程拓扑、调度控制面、并行维度和 prefill/decode 两阶段组织成一套可扩展的分布式 serving runtime。

## 0. 重审范围与方法

这次分析专门围绕 SGLang 的基础分布式架构展开，分成三部分执行：

1. 通读分布式 runtime 的主入口和控制面代码，包括 `engine.py`、`ray/engine.py`、`data_parallel_controller.py`、`disaggregation/prefill.py`、`disaggregation/decode.py`、`server_args.py`。
2. 通读与基础并行能力和模型部署建议相关的文档，包括 `docs/basic_usage/deepseek_v3.md`、`docs/basic_usage/deepseek_v32.md`、`docs/advanced_features/pipeline_parallelism.md`。
3. 按时间顺序通读并行相关提交序列，并对关键拐点提交展开 `git show --stat` 复核。

需要说明的是，当前仓库总提交数为 12439。逐 patch 通读整个仓库全部提交超出单轮可验证范围，也会引入大量与本主题无关的噪声。因此，本次“基于全文”的严格范围收敛为：

- 路径范围：`python/sglang/srt`、`docs/basic_usage/deepseek_v3.md`、`docs/basic_usage/deepseek_v32.md`、`docs/advanced_features/pipeline_parallelism.md`
- 主题范围：`tensor parallel`、`pipeline parallel`、`data parallel`、`dp attention`、`context parallel`、`expert parallel`、`disaggregation`、`prefill`、`decode`

在这个范围内，实际命中的相关提交共有 401 条，本文已完整通读这 401 条提交序列，并额外复核了以下关键拐点提交的 diff/stat：

- `0463f7fb52f06dcae2b10b7ca2a18a86ac135f96` `Support data parallelism (static) (#480)`
- `23cc66f7b65f885969d4608fd4964e0ba98fb7f5` `Add back data parallelism (#1635)`
- `11383cec3c08e7912c4398838e33eafe529e1732` `[PP] Add pipeline parallelism (#5724)`
- `711efe781426dad242e88fe71d6eefe866fe3866` `Integrating PD disaggregation with DP attention and DeepEP (#5435)`
- `384f8ab5ce2220caf00bb0815e08d33068ec5c06` `[PD] Support PD disaggregation with Prefill PP (#8846)`
- `d368c7451a48f2c58aa19e6f20a5725296094551` `(1/n)support context parallel with deepseekv3.2-DSA (#12065)`
- `05dfef92a1f03cf4f308b9bd529bfb149d737e67` `[DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled (#16380)`
- `13a2cd748db5f83926ba43e8f17380aab77097e3` `[Ray] Add data parallel (DP) and DP attention support to RayEngine (#21887)`

## 1. 结论先行

SGLang 的基础分布式架构不是“若干并行参数的堆叠”，而是一个以 scheduler 为中心、以阶段分工为原则的运行时系统。

它的核心结构可以概括为：

1. `ServerArgs` 定义逻辑并行维度和约束。
2. `Engine` 或 `RayEngine` 把逻辑维度实例化成具体进程或 actor 拓扑。
3. `Scheduler` 维持请求队列、batch 形成、KV 生命周期和跨阶段状态推进。
4. `DataParallelController` 与 `PD disaggregation` 等控制面组件负责把请求分发到合适的分布式 worker 组合。
5. 模型实现和 attention/MoE backend 决定某种拓扑组合是否真的能兑现性能收益。

因此，SGLang 的分布式架构有两个比“支持哪些并行”更本质的判断：

- 第一，真正的一等公民不是 TP/DP/PP/CP/EP 这些轴本身，而是 prefill 和 decode 两个资源画像不同的阶段。
- 第二，真正控制分布式行为的不是单个 kernel，而是参数校验、拓扑生成、请求分发、KV 传输、batch 调度这整条控制链路。

## 2. 基础分层架构

从 runtime 视角看，SGLang 的基础分布式架构可以拆成六层。

### 2.1 配置与约束层

关键文件：`python/sglang/srt/server_args.py`

这一层负责定义并行语义，而不是只是声明开关。核心参数包括：

- `tp_size`
- `pp_size`
- `dp_size`
- `ep_size`
- `attn_cp_size`
- `enable_dp_attention`
- `enable_nsa_prefill_context_parallel`
- `enable_prefill_context_parallel`
- `disaggregation_mode`
- `prefill_attention_backend`
- `decode_attention_backend`

这组参数说明 SGLang 的架构默认承认两件事：

- 并行维度不是平铺的，部分维度会重解释世界大小。
- prefill 和 decode 可以采用不同 backend，甚至不同部署角色。

也就是说，SGLang 从参数层面就已经把“阶段分工”写进了系统模型里。

### 2.2 进程编排层

关键文件：`python/sglang/srt/entrypoints/engine.py`

本地运行时的基本结构仍然是：

1. `TokenizerManager`
2. `Scheduler`
3. `DetokenizerManager`

这说明无论是单机还是多机，SGLang 的核心控制点始终是 scheduler。HTTP server、Engine、TokenizerManager 在主进程里协调请求进入；真正决定请求进入哪个 batch、何时 forward、何时传输 KV 的，仍然是 scheduler 所在控制面。

### 2.3 分布式拓扑实例化层

关键文件：`python/sglang/srt/ray/engine.py`

`RayEngine` 把逻辑并行维度展开成 actor 拓扑：

- 当 `dp_size == 1` 时，直接按 `tp_size * pp_size` 启动 scheduler actors。
- 当 `dp_size > 1` 时，先启动 `RayDataParallelController`，再由它驱动 per-DP-rank schedulers。
- 当 `enable_dp_attention` 开启时，`total_gpus` 的解释变成 `tp_size * pp_size`，而不是 `dp_size * tp_size * pp_size`。

这里的关键架构含义是：在 SGLang 里，世界大小不是固定常量，而是会被某些并行策略重新解释。

### 2.4 请求分发与负载均衡层

关键文件：`python/sglang/srt/managers/data_parallel_controller.py`

`DataParallelController` 是基础分布式架构里的核心控制面组件之一。它负责：

- 启动多个 DP worker 或 DP-attention worker 组。
- 接收 tokenizer 送来的请求。
- 依据 `load_balance_method` 决定请求发往哪个 `dp_rank`。
- 通过 `WatchLoadUpdateReq` 回收各 worker 的 load 信息，维护 `DPBudget`。

从实现上看，SGLang 不是把 data parallel 做成“多个完全独立服务器副本”，而是做成一个带统一 dispatch 和 load tracking 的中控层。这也是为什么后面的 DP attention、PD disaggregation、Ray DP controller 能继续沿同一条控制线演进。

### 2.5 阶段解耦与 KV 传输层

关键文件：

- `python/sglang/srt/disaggregation/prefill.py`
- `python/sglang/srt/disaggregation/decode.py`

这两份文件直接把 PD disaggregation 的阶段分工写成了两个不同的生命周期：

- Prefill server：`Bootstrap Queue -> Waiting Queue -> Inflight Queue`
- Decode server：`PreallocQueue -> TransferQueue -> WaitingQueue -> RunningBatch`

这比单纯说“prefill 和 decode 分离”更重要，因为它说明两边不是共享同一个状态机，只是跑不同 forward，而是拥有不同队列模型、不同预分配策略、不同传输完成条件。

其中最关键的基础事实是：decode 侧可以先做预分配和 metadata 准备，再在 transfer 完成后并入 running batch；prefill 侧则要先完成 bootstrap 和 sender 初始化，再把请求送入 waiting queue。SGLang 的 PD 不是简单 RPC，而是显式围绕 KV 生命周期构建的双端状态机。

### 2.6 模型与 backend 约束层

关键文件：`python/sglang/srt/arg_groups/deepseek_v4_hook.py`

模型不是被动接受并行参数，而是会主动重写或拒绝某些组合。例如 DeepSeek V4：

- PD disaggregation 要求 `pp_size == 1`
- CP 只支持 `round-robin-split`
- 会强制写入 `enable_dp_attention = True`
- 还要求 `dp_size == 1` 且 `tp_size <= 8`

这说明 SGLang 的基础分布式架构并不是“框架定义一次，模型全都复用”，而是框架提供骨架，模型 hook 再把可行拓扑收窄到真正稳定可跑的区域。

## 3. 基础并行维度的语义

理解 SGLang 的分布式架构，不能把 TP、DP、PP、CP、EP、PD 当作完全对称的能力，它们在系统里的职责不同。

### 3.1 TP：最基础的模型切分维度

TP 是最基础也最普适的切分方式，用来解决模型装载和单轮大算子并行。SGLang 的很多世界大小计算、rank 划分和 worker 启动默认都以 TP 为基础维度。

### 3.2 DP：统一请求入口下的多副本调度

DP 在 SGLang 里不是“复制多个完全隔离服务”，而是统一入口下的多 worker 调度。`DataParallelController` 的存在说明它天然和 load balance、活跃 rank 状态、控制消息广播绑定在一起。

### 3.3 DP attention：把 DP 语义折叠进 attention 路径

`enable_dp_attention` 的特殊之处在于，它不是简单再加一层物理副本，而是重写 attention 侧的布局语义，使 KV duplication 行为发生变化。这也是为什么 Ray 拓扑中的 `total_gpus` 解释会跟普通 DP 不同。

### 3.4 PP：跨 stage 的流水式执行

PP 的主要意义不是“再多一个并行轴”，而是改变跨节点通信模式。相比大 TP 的全量同步，PP 更多只在 stage 边界通信，因此更适合长上下文 prefill 这种单轮计算很重的场景。

### 3.5 CP：面向长序列 prefill 的上下文切分

从当前实现和文档看，CP 主要是 prefill 侧能力，尤其面向 DeepSeek V3.2/V4 这类超长上下文模型。它不是普适默认路径，而是模型特化较强的维度。

### 3.6 EP：面向 MoE 的专家维度切分

EP 的价值主要在 MoE 模型，尤其是在 decode 节点。因为 decode 的 steady-state 吞吐很容易被 KV 常驻和专家路由拖住，EP 更适合作为生成阶段的主要放大器。

### 3.7 PD：不是并行轴，而是阶段级角色拆分

PD disaggregation 与其说是一种“并行轴”，不如说是一种角色架构。它把 prefill 和 decode 拆成两个服务角色，然后允许每个角色选择更合适的 TP/PP/DP/CP/EP 组合。

## 4. Prefill 与 Decode 的基础分工

SGLang 的基础分布式架构之所以必须区分 prefill 和 decode，是因为两个阶段面对的主瓶颈不同。

### 4.1 Prefill 的系统画像

prefill 阶段的特征是：

- 单轮 token 数大
- attention 复杂度随上下文长度增长
- 用户更关注 TTFT 和长输入吞吐
- 激活峰值、页分配、跨节点大算子通信更敏感

因此 prefill 更偏好的基础策略通常是：

- 先用 TP 解决装载和单轮大算子并行
- 多节点长上下文时优先用 PP
- 极长上下文和特定模型上再启用 CP
- 需要把 prompt ingest 从生成流量里解耦时使用 PD prefill 节点

### 4.2 Decode 的系统画像

decode 阶段的特征是：

- 单轮只生成一个或少量 token
- 调度频率极高
- KV cache 常驻成本成为主要压力
- 小额同步和专家分发会被持续放大

因此 decode 更偏好的基础策略通常是：

- 通过 DP attention 降低 KV duplication
- 对 MoE 模型使用 EP
- 通过 PD decode 节点把生成调度独立出来
- 在超长上下文下进一步叠加 decode-side offload 或 HiSparse

### 4.3 阶段分工先于参数调优

SGLang 文档和代码最重要的共同结论不是“某个参数值最佳”，而是要先决定这个节点到底是 prefill 节点还是 decode 节点。只有先确定节点角色，TP/PP/DP/CP/EP 的甜点才有意义。

## 5. 三种基础部署形态

### 5.1 单机或单角色 unified runtime

这是最基础的形态：TokenizerManager、Scheduler、DetokenizerManager 和模型 worker 协同工作，prefill 与 decode 共享同一调度面。适合小规模部署和不需要角色拆分的场景。

### 5.2 带 DP controller 的统一入口多副本形态

当 `dp_size > 1` 时，SGLang 进入“统一入口 + 中控分发 + 多 worker”形态。这里仍然是单一服务角色，但请求调度已经不再是单 scheduler 内部问题，而是 controller 决定目标 `dp_rank`，worker 再各自调度自己的批次。

### 5.3 PD disaggregation 形态

在 PD 形态下，系统拆成 prefill server 和 decode server。两边的队列、内存预分配、KV 传输和 merge 逻辑不同。这个形态也是 SGLang 真正把“prefill 和 decode 是两个阶段”落实到运行时结构上的地方。

## 6. 模型级并行甜点

### 6.1 DeepSeek V3 / V3.1 / R1

从 `docs/basic_usage/deepseek_v3.md` 和相关实现可以看出，DeepSeek V3 系列更像“两阶段双甜点”模型：

- prefill 甜点：`PP + chunked prefill + dynamic chunking`
- decode 甜点：`DP attention`，必要时再叠加 `PD` 或 `EP`

原因是 MLA 模型在 decode 时最容易被 KV duplication 限制，而在长上下文 prefill 时又很适合通过 PP 降低跨节点全量同步。

### 6.2 DeepSeek V3.2 / V4

从 `docs/basic_usage/deepseek_v32.md` 和 `deepseek_v4_hook.py` 看，DeepSeek DSA 路线的甜点更明确地按阶段分家：

- prefill 甜点：`CP` 或 `PP + CP`
- decode 甜点：`EP` 或专用 `PD decode` 节点

V4 甚至把部分约束直接写进 hook，说明这些甜点已经不是经验总结，而是实现级约束。

### 6.3 Qwen3-235B-A22B-FP8

`docs/advanced_features/pipeline_parallelism.md` 给出的推荐更偏向深 PP：`TP4 + PP8`。这说明不同模型结构会直接改变最优 PP 深度，Qwen3-235B-A22B-FP8 更适合用更多 stage 换取更好的跨节点扩展性。

### 6.4 一般性判断

对小模型、短上下文、单机低延迟场景，SGLang 并不鼓励默认打开复杂并行。更合理的路径通常是：

- 先用单机 TP 或单卡
- 模型装不下或 TTFT 恶化时再引入 PP
- decode 并发被 KV 限制时再引入 DP attention、PD 或 EP

## 7. 从提交时间线看架构演进

把 401 条相关提交按时间串起来，可以看到 SGLang 对分布式架构的理解经历了四个阶段。

### 7.1 第一阶段：先补齐 DP 控制面

代表提交：

- `0463f7fb52f06dcae2b10b7ca2a18a86ac135f96` `Support data parallelism (static) (#480)`
- `23cc66f7b65f885969d4608fd4964e0ba98fb7f5` `Add back data parallelism (#1635)`

这一阶段的重点是把 DP 从单纯副本概念做成带 controller 的控制面能力。

### 7.2 第二阶段：PP 成为长上下文 prefill 主线

代表提交：

- `11383cec3c08e7912c4398838e33eafe529e1732` `[PP] Add pipeline parallelism (#5724)`
- `11553c1a3727ce20ca4b85ea767f46fcdcb7661d` `Add pipeline parallelism for Qwen2 and Qwen3 Model (#6250)`

这一阶段说明 PP 已经从框架级能力走向模型落地能力，并开始服务超长 prefill 场景。

### 7.3 第三阶段：PD 把 prefill/decode 真正拆成两个角色

代表提交：

- `711efe781426dad242e88fe71d6eefe866fe3866` `Integrating PD disaggregation with DP attention and DeepEP (#5435)`
- `384f8ab5ce2220caf00bb0815e08d33068ec5c06` `[PD] Support PD disaggregation with Prefill PP (#8846)`

这条线意味着 SGLang 不再追求统一拓扑覆盖所有阶段，而是允许 prefill 和 decode 各自选择更合适的组合。

### 7.4 第四阶段：CP、Ray 和复杂组合逐步补齐

代表提交：

- `d368c7451a48f2c58aa19e6f20a5725296094551` `(1/n)support context parallel with deepseekv3.2-DSA (#12065)`
- `05dfef92a1f03cf4f308b9bd529bfb149d737e67` `[DeepSeek 3.2] Support and optimize pipeline parallelis when context pipeline enabled (#16380)`
- `13a2cd748db5f83926ba43e8f17380aab77097e3` `[Ray] Add data parallel (DP) and DP attention support to RayEngine (#21887)`

到这个阶段，SGLang 的基础分布式架构已经不再只是本地 mp 模式能力，而是要求 Ray、多阶段角色、模型特化约束都能统一表达。

## 8. 对维护者最有价值的排查入口

如果后续继续分析或扩展 SGLang 的基础分布式架构，建议优先从下面这些位置入手：

1. `python/sglang/srt/server_args.py`
   - 看并行语义、合法性校验和 backend/模型默认值如何互相影响。

2. `python/sglang/srt/entrypoints/engine.py`
   - 看本地模式下 TokenizerManager、Scheduler、DetokenizerManager 的基本编排。

3. `python/sglang/srt/ray/engine.py`
   - 看 Ray 模式下世界大小、placement group 和 controller/actor 拓扑如何生成。

4. `python/sglang/srt/managers/data_parallel_controller.py`
   - 看统一入口下的 DP 分发与负载均衡。

5. `python/sglang/srt/disaggregation/prefill.py`
   - 看 prefill 角色的 bootstrap、waiting、inflight 三段式生命周期。

6. `python/sglang/srt/disaggregation/decode.py`
   - 看 decode 角色的 prealloc、transfer、merge 与 running batch 生命周期。

7. `python/sglang/srt/arg_groups/deepseek_v4_hook.py`
   - 看模型如何反向约束并行组合。

## 9. 总结

从全文代码和相关文档综合看，SGLang 的基础分布式架构最值得注意的不是“并行轴很多”，而是它把分布式 serving 的真正控制问题放在了正确的位置。

第一，系统首先区分阶段，再区分并行轴。prefill 和 decode 的分工先于 TP/DP/PP/CP/EP 的调优。

第二，系统首先建立控制面，再扩张执行面。`Engine`、`RayEngine`、`DataParallelController`、PD 的双端队列模型，才是这些并行策略能够长期演进而不失控的原因。

第三，系统首先承认模型差异，再讨论统一框架。DeepSeek V3、DeepSeek V3.2/V4、Qwen3-235B-A22B-FP8 的甜点不同，不是偶然，而是基础架构、模型 hook 和提交演进共同收敛出的结果。

因此，如果要理解或继续扩展 SGLang 的分布式能力，最有效的顺序不是“还支持什么并行”，而是：这个节点处于哪个阶段、它的控制面由谁主导、它的 KV 生命周期如何推进、以及当前模型允许什么拓扑。这才是 SGLang 基础分布式架构真正稳定的骨架。