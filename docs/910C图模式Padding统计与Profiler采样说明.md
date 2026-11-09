# 910C 图模式 Padding 统计与 Profiler 采样说明

本文记录 DeepSeek V3.2 W8A8 在 16 卡 Ascend 910C 上使用 DP=2、TP=8、`enable_mlapo=false` 和 `FULL_DECODE_ONLY` 图模式时，如何查看实际调度 token 数、图执行容量和 padding 数。

## 1. 完整启动命令

启动前退出旧服务，保留当前用于控制 batch 拆分和 attention 准备后移的环境变量设置。下面的命令不改变这些开关。

```bash
mkdir -p /workspace/vllm_profile
set -o pipefail

PYTHONUNBUFFERED=1 vllm serve /models/DeepSeek-V3.2-W8A8 \
  --additional-config '{"enable_mlapo":false}' \
  --host 0.0.0.0 \
  --port 8000 \
  --data-parallel-size 2 \
  --tensor-parallel-size 8 \
  --quantization ascend \
  --seed 1024 \
  --served-model-name deepseek_v3_2 \
  --enable-expert-parallel \
  --max-num-seqs 64 \
  --max-model-len 33000 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --no-enable-prefix-caching \
  --gpu-memory-utilization 0.97 \
  --cudagraph-metrics \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[8,16,32,64]}' \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/workspace/vllm_profile","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true,"ignore_frontend":true,"delay_iterations":0,"max_iterations":10}' \
  2>&1 | tee /workspace/vllm_profile_decode_start.log
```

主要新增参数为 `--cudagraph-metrics`。`tee` 将服务终端输出同时保存到日志文件；重复执行此命令会覆盖同名日志，需保留旧结果时请先改名或备份。

## 2. 查看 padding 统计

启动服务并运行 benchmark 后，在同一容器的另一个终端中执行：

```bash
grep -A 20 'CUDAGraph Stats' /workspace/vllm_profile_decode_start.log
```

实时查看服务输出：

```bash
tail -f /workspace/vllm_profile_decode_start.log
```

预期表头和示例数据如下。此行仅用于解释字段，并非实测结果；运行模式的具体字符串可能随版本变化。

```text
Unpadded Tokens | Padded Tokens | Num Paddings | Runtime Mode | Count
20             | 32            | 12           | FULL         | 50
```

| 字段 | 含义 | 示例 |
| --- | --- | --- |
| Unpadded Tokens | 本次执行实际调度的 token 数 | 20 |
| Padded Tokens | 补齐后的图执行 token 容量 | 32 |
| Num Paddings | 补齐位置数，即前两者之差 | 12 |
| Runtime Mode | 实际执行模式 | FULL |
| Count | 日志统计周期内该组合出现的次数 | 50 |

普通 decode 每条请求每步生成一个 token 时，20 个有效 token 对应这次执行中的 20 条请求。Prefill、混合调度或推测解码不能直接将 token 数当作请求数。

图容量属于每个 DP 副本的执行批次，TP=8 的各 rank 共同处理该批次，不再将 batch 除以 8。阅读 DP=2 的日志时应保留进程、engine 或 DP 标识，不能把某个副本的统计当作全局请求数。

## 3. 与 profiler 输出的区别

| 输出 | 内容 |
| --- | --- |
| `/workspace/vllm_profile_decode_start.log` | 服务日志；启用并正常接通统计功能后，包含周期汇总的图容量及 padding 数据 |
| `/workspace/vllm_profile/` | Profiler 采样及解析输出，通常包含各 worker 的子目录 |
| `kernel_details.csv` | 算子时间、执行形状、核数等；默认没有有效请求数或 batch padding 数字段 |

`--cudagraph-metrics` 不需要先调用 `/start_profile`，也不会自动给 `kernel_details.csv` 增加 padding 字段。

仅看到 Indexer 输入第一维为 32，无法区分“32 个有效 token”与“20 个有效 token 加 12 个 padding”。`record_shapes=true` 记录形状，不记录张量内容中的有效行数。

图统计是日志周期内的组合计数，不是逐 step 的时间戳记录。要精确对应某一次 profiler 执行，需要在调度执行处额外记录时间戳、step ID、DP 编号、实际调度 token 数和补齐后的容量。

## 4. 每轮 profiler 采样都手动停止

本启动命令保留了 `max_iterations=10`。它是否会在 Ascend worker 上自动停止采样，取决于实际安装版本，不能只根据参数已被接受就判断生效；这里的 iterations 也不能直接理解成某条请求生成的 token 数。

建议每轮采用以下流程，并检查两个 HTTP 请求均成功。若上一轮未停止，先调用 `/stop_profile` 并等待返回，再开始新一轮。

```bash
# 1. 开始采样，检查响应状态和正文
curl --noproxy '*' -sS -i -X POST \
  http://127.0.0.1:8000/start_profile

# 2. 在此运行 benchmark，或让正在运行的请求继续执行

# 3. 停止采样，等待导出
curl --noproxy '*' -sS -i -X POST \
  http://127.0.0.1:8000/stop_profile

# 4. 递归查找解析结果
find /workspace/vllm_profile -type f \
  \( -name 'kernel_details.csv' \
     -o -name 'trace_view.json' \
     -o -name 'trace_view.json.gz' \)
```

不要在采集或导出期间删除输出目录。仅调用 `/start_profile` 不保证立即出现解析后的 CSV；若未达到自动停止条件、版本不支持自动停止或解析尚未完成，结果可能尚未生成。

若仍无结果，检查服务日志：

```bash
grep -nEi 'profil|trace|export|permission|no space|error|exception' \
  /workspace/vllm_profile_decode_start.log | tail -100
```

## 5. 已确认范围与版本限制

2026-11-09 对辅助 910B 环境进行了只读检查，确认其源码存在：

```python
if self.vllm_config.observability_config.cudagraph_metrics:
    cudagraph_stats = CUDAGraphStat(
        num_unpadded_tokens=num_tokens,
        num_padded_tokens=batch_descriptor.num_tokens,
        num_paddings=batch_descriptor.num_tokens - num_tokens,
        runtime_mode=str(cudagraph_mode),
    )
```

同时找到相应日志汇总代码，但没有进行新一轮运行验证。该环境的 Ascend worker profiler 实现未见 `max_iterations` / `delay_iterations` 的处理或 profiler 步进调用，因此不能依靠它的 `max_iterations=10` 自动结束采样。

以上检查不能替代 910C 容器的版本验证。若启动时提示 `unrecognized arguments: --cudagraph-metrics`，说明需要核查当前版本的参数支持；若参数被接受但没有统计表，还需检查 Ascend worker 到日志统计的传递路径及日志配置。

参考：[vLLM 官方 Profiler 使用说明](https://github.com/vllm-project/vllm/blob/main/docs/contributing/profiling.md)。官方当前文档要求即使 worker 达到自动停止上限，也应调用 `/stop_profile` 关闭会话后再开始下一轮；历史版本的具体行为可能不同。
