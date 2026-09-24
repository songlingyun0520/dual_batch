# 验证记录

## 2026-09-24 迁移复测

迁移后在原 910B2 服务器上创建了一个新的临时目录和 `--system-site-packages` 虚拟环境，
使用本仓库的 `pip install -e .` 安装插件，并在独立端口启动。
没有修改原实验目录，也没有安装或替换系统的 torch/vLLM 包。
这验证了路径迁移、插件注册及本仓库启动器；**没有在第二台物理服务器或全新软件镜像上测试**。

已完成的检查：

- Python 文件语法编译，Bash 脚本语法检查。
- 5 个无需 NPU 的启动器测试：图配置、跨目录/空格路径、eager 验证模式、baseline 开关、非法模式拒绝。
- 原环境 API 预检查、64 GB 级 NPU 可见、插件 entry point 可发现。
- 默认图模式启动成功，`/health` 返回 HTTP 200。
- 完整 HTTP 回归覆盖 batch 1/2/3/4/7/8、2048 边界、超过 2048、不同输入长度及缓存 slot 复用。
- eager 与图模式各 10 组请求全部通过，生成 token 完全一致，logprob 最大绝对差为 0。
- eager 中拆分/未拆分路径完成 488 次逐层比较，覆盖全部 61 层；最大绝对输出误差为 0，所有检查的缓存行相同。
- 运行时审计确认 `max_decode_batch_seen=8`。

迁移复测的数值结果摘要另存于 [migration-validation.json](migration-validation.json)。
测试完成后停止临时服务，保留原服务器原有部署文件。

## 依赖来源检查

从原机记录了关键文件 SHA-256，并与 GitHub 上 vLLM 与 vLLM-Ascend 的 `v0.18.0` tag 比较。
以下 5 个文件完全一致：

```text
vllm/model_executor/models/deepseek_v2.py
vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py
vllm_ascend/attention/sfa_v1.py
vllm_ascend/worker/model_runner_v1.py
vllm_ascend/ops/fused_moe/fused_moe.py
```

清单见 [source-environment.json](../environment/source-environment.json)。
这一检查不等于所有安装文件或二进制 wheel 均已逐字比较。
原机 torch-npu 的 git version 也记录在清单中，便于排查目标环境差异。

## 原实验历史记录（2026-09-22）

迁移源的 `DUAL_BATCH.md` 及结果文件记录：

- 488 次层级输出/缓存对比，覆盖 61 层，记录的最大绝对输出误差为 0，缓存行相同。
- 未拆分、eager 拆分和 ACL Graph 拆分的 10 组请求，生成 token 和 logprob 一致。
- 在 32K 上下文、batch 4/8 上采集过单层 Indexer/SFA 窗口。

这些属于原版本实验记录，不将它们冒充为本次迁移版的新测量。
本仓库不提交原日志、profiling 大文件和历史生成文本；可按运行指南重新生成。

## 在目标服务器再次验证

```bash
python3 -m unittest discover -s tests -v
source scripts/env.sh
python3 scripts/check_env.py
```

随后按 [运行指南第 5～7 节](RUNNING.md) 进行真实设备上的回归、数值比较与 profiler 采集。
迁移版保留原计算算法，只修改路径、输出目录、启动器、插件安装和客户端参数/检查。
一致性通过也不代表具备真实权重模型的语义正确性。
