# 双 micro-batch 的实现与边界

## 插件与存储适配

`pyproject.toml` 注册 `vllm.general_plugins:dsv32_demo → demo_patch:register`。
插件仅在 `DSV32_SHAPE_DEMO=1` 时安装进程内补丁，不修改 site-packages 文件。
`DSV32_DUAL_BATCH=1` 再包装 `AscendSFAImpl.forward`。

保留 61 层、hidden size 7168、128 attention heads，前 3 层 dense MLP、后 58 层 MoE。
原生归一化、投影、Lightning Indexer、SparseFlashAttention、MoE 路由和融合计算仍执行。
为了在单卡放下结构，采用以下改变语义的适配：

- dummy 随机权重；不加载检查点，不启用 MTP，配置去除 FP8 声明，采用 BF16。
- 256 维路由打分仍计算，top-8 专家 ID 对 8 取余，映射到 8 个物理专家。
- 这组物理专家参数在 58 个 MoE 层间共享。
- 每请求每层 attention KV 仅保留前 2048 tokens；Indexer key cache 保留完整逻辑上下文。
- 真实 Indexer 仍执行，但送入 SFA 的索引覆盖为固定前缀，超出范围用 -1 padding。

因此任何数值一致性仅说明“拆分版与同一 demo 的未拆分版一致”，不能证明与真实 DeepSeek 一致。

## 请求拆分

仅纯 decode、实际请求数大于 1，且 metadata 和张量形状匹配时拆分。
切分点为 `(B+1)//2`，两组大小为 `ceil(B/2)`、`floor(B/2)`。
例如 B=7 为 4+3；B=8 为 4+4。batch=1、prefill、混合步骤和不支持的 padding 回退到原路径。

每层的执行顺序：

```text
common：共享 QKV 低秩投影、q_c 归一化、当前 token KV 写入
   ↓
I0：第 0 组 Indexer 全阶段（K/Q 投影、RoPE、cache 写入、LightningIndexer）
   ↓
I1：第 1 组 Indexer 全阶段
   ↓
P ：完整 batch 的 attention Q 投影、BMM、RoPE、layout 和固定索引准备
   ↓
A0：第 0 组原生 SparseFlashAttention
   ↓
A1：第 1 组原生 SparseFlashAttention
   ↓
merge：按原请求顺序合并、V 上投影、O 投影
   ↓
原有 residual、FFN/MoE
```

每组保留原请求的物理 KV slot，子组累计 query 长度从 1 重新计数。
共享 Q 前处理合批执行一次，没有将整个模型分别跑两遍。
普通模式不增加全局设备同步；数值验证模式为了读取并比较结果会同步。

## 当前没有实现的功能

当前只使用一条计算流，KV 已在 HBM，没有传输流、真实 KV 加载、模拟等待、
`aclrtValueWait` 同步或双组并行计算。I0/I1 完成位置是未来接入分组加载的候选边界。
MLAPO、DSA-CP、量化 Indexer、KV producer 与多卡均不在此实现范围内。

## 测量定义

稳定 decode 的每层应出现 `Indexer0 → Indexer1 → SFA0 → SFA1` 的设备 kernel 顺序。

```text
W0 = start(SFA0) - end(Indexer0)
W1 = start(SFA1) - end(Indexer1)
```

`elapsed_us` 是实际设备时间间隔，包含 kernel 间空隙；`busy_us` 是同一图实例中
kernel 时间区间的并集，排除空闲，不把空隙算作有效计算。
逐层逐步直接读取 profiler 时间，不用整模型延迟除以 61。

这些是未来传输可能利用的时间窗口，不是已经实现的传输隐藏量。
两组如果共用传输路径，B0 传输尚未结束会压缩 B1 的实际时间预算；不能把 W0/W1 当作独立额度。
当前未测量页面映射、真实复制、提交开销、复制与计算的资源竞争。profiler 本身也有开销。
