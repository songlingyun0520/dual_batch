# DeepSeek V3.2 双 micro-batch 单卡实验

将纯 decode 阶段的请求 batch 分为两组，在一张 Ascend 910B2 上执行
`Indexer0 → Indexer1 → Q 前处理 → Attention0 → Attention1`。
仓库包含迁移后的代码、模型结构配置、tokenizer、启动脚本及验证/测量工具。
**不包含、也不下载 DeepSeek 模型权重。** 启动固定使用 `--load-format dummy` 生成随机参数。

这是推理流程与设备时间窗口的实验 demo。输出文本没有语义，不用于模型精度或真实
DeepSeek V3.2 吞吐评测。保留 61 层结构，但采用专家参数共享、路由映射、attention KV 截断等适配。
全部 KV 已驻留 HBM；当前两组在一条计算流中顺序执行，尚未实现 KV 传输与计算重叠。

## 从这里开始

- [详细部署与运行指南](docs/RUNNING.md)：新服务器环境、安装、启动、验证、停止及排错。
- [实现说明](docs/DESIGN.md)：双 batch 的拆分位置、缓存语义、时间窗口定义与限制。
- [验证记录](docs/VALIDATION.md)：历史验证与本次迁移复测的区别。
- [原服务器环境](environment/source-environment.json)：版本与关键源文件的 SHA-256。

## 快速运行

前提：Linux、空闲的 64 GB Ascend 910B2，以及配置好的 CANN/NNAL 9.0.0、
vLLM/vLLM-Ascend 0.18.0 和兼容的 PyTorch 环境。**普通 NVIDIA CUDA 镜像不能运行本代码。**

```bash
git clone https://github.com/songlingyun0520/dual_batch.git
cd dual_batch

# 在已有 Ascend 推理环境上建立隔离的 demo 插件环境
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source scripts/env.sh
python3 -m pip install --no-deps --no-build-isolation -e .
python3 scripts/check_env.py

# 前台启动；默认启用双 batch、decode 图模式，监听 127.0.0.1:8000
bash launch.sh
```

另开终端，在仓库目录中执行：

```bash
curl -f http://127.0.0.1:8000/health
python3 verify_dual_service.py --short --output runtime/short-checks.json
```

只有看到健康检查成功后再发送测试请求。首次初始化、随机参数分配和图捕获可能需要数分钟。
前台启动窗口按 `Ctrl+C` 停止。

## 常用开关

```bash
# 普通执行模式，不捕获图
bash launch.eager.sh

# 不拆分的对照组（仍是同一个随机权重、KV 截断 demo）
DSV32_DUAL_BATCH=0 bash launch.sh

# 拆分前后逐层数值/缓存检查，只能用于 eager
DSV32_DUAL_BATCH_VALIDATE=1 bash launch.eager.sh

# 更换监听地址/端口和输出目录
HOST=0.0.0.0 PORT=18000 DSV32_OUTPUT_DIR=/data/dual-batch-run bash launch.sh
```

每次修改模式都需要先停止旧服务，再启动新进程。默认最大 batch=8；batch=1、prefill、
混合步骤及不符合条件的 padding 情况走原 demo 路径。

## 文件结构

```text
demo_patch.py               单卡参数共享、compact KV 与 vLLM 插件入口
dual_batch.py               decode 请求拆分和逐层一致性检查
launch.sh / launch.eager.sh 图模式与 eager 启动脚本
pyproject.toml              注册 vllm.general_plugins 插件
model/                     结构配置和 tokenizer，无权重
scripts/check_env.py        版本、设备、插件与内部 API 预检查
verify_dual_service.py      HTTP 回归与对照结果比较
profile_dual_windows.py     稳定 decode 窗口采集
analyze_dual_windows.py     按设备 kernel 时间线计算窗口
docs/                      部署、设计与验证说明
environment/               原环境清单与版本约束参考
runtime/                   启动后生成；已被 Git 忽略
```

源代码依赖 vLLM 和 vLLM-Ascend 的内部 API；同版本号不一定代表本地包内容完全一致。
不要直接升级到最新版。源服务器和本仓库的实测范围见验证记录。

模型配置与 tokenizer 的第三方说明见 [model/README.md](model/README.md)。
