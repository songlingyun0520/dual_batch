# 在另一台服务器运行

## 1. 确认目标与硬件

本仓库复现的是 DeepSeek V3.2 **结构/执行流程实验**，不是加载真实权重的语言模型服务。
不需要下载几百 GB 的检查点，不要将真实权重放入 `model/`。

已验证的硬件是单张 Ascend 910B2，HBM 64 GB，Linux aarch64。
模型随机参数的历史分配约 34.90 GiB，另有 KV、图捕获和运行时内存；需要一张空闲的 64 GB 卡。
未验证 NVIDIA GPU、310P、32 GB NPU、多卡或多机。固定配置 TP=1，不支持直接改为多卡。
主机内存需求未做下限测试；建议预留充足内存，长上下文和 profiler 会额外占用主机内存与磁盘。

```bash
npu-smi info
python3 --version
uname -m
```

容器内要能访问目标 `/dev/davinciN`、管理设备以及驱动库。物理 NPU 编号可能是 4 或 7，
容器/PyTorch 中的逻辑编号仍可能为 0；不要照抄源服务器的物理编号。
多卡环境可在启动前设置 `ASCEND_RT_VISIBLE_DEVICES` 选择空闲卡，并重新运行预检查确认。

## 2. 准备软件环境

原服务器实测组合：

| 组件 | 版本 |
|---|---|
| 操作系统 / 架构 | openEuler 24.03 LTS-SP2 / aarch64 |
| Python | 3.11.6 |
| 驱动（npu-smi） | 25.2.1 |
| CANN / NNAL ATB | 9.0.0 / 9.0.0 |
| vLLM / vLLM-Ascend | 0.18.0 / 0.18.0 |
| torch / torch-npu | 2.9.0 / 2.9.0 |
| transformers / numpy | 4.57.6 / 1.26.4 |

### 路径 A：已有可运行的 Ascend vLLM 环境

优先复用已有的匹配环境或内部镜像，然后只安装本仓库的轻量插件。
`--system-site-packages` 让新虚拟环境复用现有 torch、torch-npu 和 vLLM，避免重新下载大型依赖。
它不隔离系统软件栈，只隔离本次插件安装；需要在源环境仍可用的条件下运行。

### 路径 B：从零配置

按 [vLLM-Ascend 0.18.0 官方安装指南](https://docs.vllm.ai/projects/ascend/en/v0.18.0/installation.html)
安装与硬件匹配的驱动、固件、CANN 9.0.0、910B 算子包、NNAL 9.0.0，以及 vLLM/vLLM-Ascend 0.18.0。
可以使用该指南中的 Ascend CANN 或 vLLM-Ascend 容器路线；设备挂载和驱动兼容性按目标宿主机配置。
不要使用 CUDA 版 PyTorch/vLLM 镜像替代 Ascend 环境。

官方 0.18.0 安装页推荐的 torch-npu 是 **2.9.0.post2**，而源服务器实际装的是 **2.9.0**。
本仓库将这一差异明确保留，不声称两种环境已经完成同等验证。优先使用已验证环境；使用官方
post2 组合时，先预检查，再执行本指南的 HTTP、逐层数值与 baseline 对比测试。
`environment/constraints-source.txt` 是原环境记录，**不是一键安装驱动/CANN 的安装器**。

如已有兼容的软件栈，只需对齐 Python 级依赖，可按需使用：

```bash
python3 -m pip install 'transformers==4.57.6' 'numpy==1.26.4'
```

vLLM/Ascend 的安装与编译选项请使用版本化官方指南，避免让通用 pip 命令拉入不匹配的 CUDA 依赖。

## 3. 克隆并安装 demo 插件

```bash
git clone https://github.com/songlingyun0520/dual_batch.git
cd dual_batch
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
source scripts/env.sh
python3 -m pip install --no-deps --no-build-isolation -e .
python3 scripts/check_env.py
```

若构建后端缺失，在虚拟环境安装 `setuptools>=68` 和 `wheel` 后重试。
仓库可以放在任意可写目录，启动脚本从自身位置解析路径；无需放在 `/root`。
**不能只复制两个 Python 文件然后直接启动服务**：`pip install -e .` 会注册
`vllm.general_plugins`，插件才能在 API server 和 worker 中被发现。

自定义安装路径时，在 `source scripts/env.sh` 或启动命令之前设置：

```bash
export CANN_ENV_FILE=/your/cann/set_env.sh
export ATB_ENV_FILE=/your/nnal/atb/set_env.sh
```

默认查找 `/usr/local/Ascend/cann/set_env.sh`，其次查找
`/usr/local/Ascend/ascend-toolkit/set_env.sh`；ATB 默认路径为
`/usr/local/Ascend/nnal/atb/set_env.sh`。

预检查必须显示 `Preflight PASSED`。它会检查版本、插件 entry point、设备可用性、HBM 容量及部分内部 API。
它不代替实际推理测试，也不保证目标机器上的二进制算子与驱动完全兼容。

## 4. 启动服务

在上述已激活的虚拟环境中：

```bash
bash launch.sh
```

默认配置：

| 项目 | 值 |
|---|---|
| 服务地址 | `127.0.0.1:8000` |
| API 模型名 | `deepseek-v3.2-shape-demo` |
| 模型数据 | 仓库 `model/`；dummy 随机参数；不联网加载权重 |
| 双 batch | 开启，`DSV32_DUAL_BATCH=1` |
| 执行模式 | `FULL_DECODE_ONLY`，捕获 batch 1～8 |
| dtype / TP / seed | BF16 / 1 / 0 |
| 最大序列长度 / 并发 | 32896 / 8 |
| 每步 token 预算 / 长 prefill chunk | 1024 / 128 |
| attention KV | 每请求每层仅保留前 2048 tokens |
| 运行输出 | 仓库 `runtime/` |

初始化会分配随机权重并捕获图。启动成功需同时确认日志中的：

```text
DEMO_PATCH active: ...
DUAL_BATCH enabled: request halves, I0 I1 P A0 A1, all KV resident; ...
Application startup complete.
```

另开终端检查：

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/v1/models
```

如需从其他机器访问，显式设置 `HOST=0.0.0.0` 并按你的网络环境设置访问权限。
更换端口、输出目录或 Python：

```bash
HOST=0.0.0.0 PORT=18000 DSV32_OUTPUT_DIR=/data/dual-batch-run \
  PYTHON="$VIRTUAL_ENV/bin/python3" bash launch.sh
```

`PYTHON` 应指向已安装插件的同一个解释器。脚本使用该 Python 的 API server 模块启动，
避免系统 `vllm` 可执行文件的 shebang 意外绕过虚拟环境。

## 5. 发送请求与验证 batch 拆分

```bash
curl -f http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v3.2-shape-demo","prompt":"Hello","max_tokens":8,"temperature":0,"ignore_eos":true}'

# 从仓库目录运行，覆盖 batch 1/2/3/4/7/8
python3 verify_dual_service.py --short --output runtime/short-checks.json

# 再覆盖 2048 边界、越界上下文、不同长度、slot 复用
python3 verify_dual_service.py --output runtime/full-checks.json
```

修改端口时，给客户端加 `--base-url http://127.0.0.1:18000`。
检查 `runtime/batch_audit.json` 中的 `max_decode_batch_seen`，确认实际出现多请求 decode，
不能只根据 API 请求里放了多条 prompt 就认定设备实际执行过拆分。
返回文本无意义是预期；测试验证返回数量、生成长度、有限 logprob 等流程性质。

## 6. 原版与拆分版一致性检查

所有实验使用同一套软件栈、同一个配置、固定 seed=0；每次切换模式都要停止旧进程。
下列“baseline”也是本仓库的随机权重/固定 KV demo，不是真实权重 DeepSeek 基准。

1. 启动不拆分服务：`DSV32_DUAL_BATCH=0 bash launch.eager.sh`。
2. 在另一终端运行：

   ```bash
   python3 verify_dual_service.py --output runtime/baseline.json
   ```

3. 在服务终端按 `Ctrl+C`，确认端口释放。
4. 启动拆分并打开逐层检查：

   ```bash
   DSV32_DUAL_BATCH=1 DSV32_DUAL_BATCH_VALIDATE=1 bash launch.eager.sh
   ```

5. 在另一终端比较：

   ```bash
   python3 verify_dual_service.py \
     --reference runtime/baseline.json --output runtime/dual-eager.json
   ```

6. 停止服务；关闭数值验证并改用图模式：

   ```bash
   DSV32_DUAL_BATCH_VALIDATE=0 bash launch.sh
   ```

7. 比较图模式：

   ```bash
   python3 verify_dual_service.py \
     --reference runtime/baseline.json --output runtime/dual-graph.json
   ```

客户端要求 token 一致，logprob 最大绝对差默认不超过 `1e-3`，可用 `--logprob-atol` 显式调整。
逐层检查写入 `runtime/dual_batch_validation.jsonl`，比较层输出及有效缓存行；输出采用
rtol=0.01、atol=0.001、RMS 相对误差≤0.005，缓存要求完全一致。
验证模式会额外执行原版路径、读取设备结果并同步，**不能拿其耗时作性能结论**。
此模式只允许 eager；启动器会拒绝同时启用图捕获。

## 7. 采集单层时间窗口

使用默认图模式、双 batch 开启、数值验证关闭的空闲服务。采集期间不要混入其他请求。
在服务器本机仓库目录执行，保证能访问服务生成的 profiler 文件：

```bash
python3 profile_dual_windows.py --batch 4 --context 32768
python3 profile_dual_windows.py --batch 8 --context 32768
```

自定义输出位置时，客户端必须使用同样的目录：

```bash
python3 profile_dual_windows.py --batch 4 --context 32768 \
  --base-url http://127.0.0.1:18000 --run-dir /data/dual-batch-run
```

每条请求至少输出 8 tokens 后启动 profiler，至少 24 tokens 后停止，默认总共输出 64 tokens。
终端打印 `trace_dirs`，manifest 位于 `runtime/dual_measurements/`。
从对应 trace 中找到 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`：

```bash
find runtime/profiles -name kernel_details.csv
python3 analyze_dual_windows.py \
  runtime/profiles/实际的trace目录/ASCEND_PROFILER_OUTPUT \
  --batch 4 --output runtime/analysis-b4
```

分析器支持偶数 batch 2/4/8，严格要求完整的 61 层 decode 步骤与 I0/I1/A0/A1 顺序。
如果 profiler 边界截到半个步骤，会明确报错；重新采集完整窗口，不要删断言来生成结果。
输出逐层统计、逐次样本 CSV 和 JSON，定义见 [DESIGN.md](DESIGN.md)。

## 8. 停止、后台运行与输出

建议先以前台模式验证；按 `Ctrl+C` 正常停止 API server 和 worker，随后用 `npu-smi info`
确认显存释放。不要对整台服务器执行 `pkill python` 或 `pkill vllm`。

需要后台运行时，在激活的环境、仓库目录执行：

```bash
mkdir -p runtime
nohup bash launch.sh > runtime/server.log 2>&1 < /dev/null &
echo $! > runtime/server.pid
tail -f runtime/server.log
```

停止前确认 PID 仍属于这个仓库的 API server，避免使用已经过期的 PID 文件：

```bash
ps -fp "$(cat runtime/server.pid)"
# 确认上面输出为本次 demo 后执行
kill -TERM "$(cat runtime/server.pid)"
```

运行产生的日志、缓存审计、profiler 和测试结果均放在 `runtime/`（或 `DSV32_OUTPUT_DIR`），
不提交到 Git。更换 `DSV32_OUTPUT_DIR` 不会自动改写 `--output` 参数；客户端结果路径按需显式设置。

## 9. 常见问题

| 现象 | 检查方向 |
|---|---|
| 找不到 `libc_sec.so`、`libascendcl.so`、`libatb.so` | 在同一 shell 加载 CANN 和 NNAL；核对 env 脚本路径、驱动库挂载 |
| 找不到 `dsv32_demo` 插件，没有 `DEMO_PATCH active` | 用启动服务的同一 Python 执行 `pip install -e .`；检查旧 PYTHONPATH；若设置过 `VLLM_PLUGINS`，确保包含 `dsv32_demo` |
| `SharedFusedMoE`、SFA 或 model runner 成员不存在 | 内部 API 版本不一致；对照环境清单，不要使用 vLLM 最新版 |
| 自定义 Ascend 算子无法导入/执行 | 检查 vLLM-Ascend 算子构建、目标 SoC、CANN/torch-npu ABI；预检查不能覆盖所有二进制问题 |
| OOM 或初始化失败 | 先确认 64 GB 卡空闲；不要任意修改层数/专家数后仍称为原 demo；小显存需要另行设计配置 |
| 图捕获失败但 eager 成功 | 先用 `launch.eager.sh` 排查，核对软件栈和固定地址 metadata 支持 |
| `output` 文本乱码或无意义 | dummy 随机参数的预期表现 |
| 生成了多条结果，但没有拆分 | 查看 `max_decode_batch_seen`、运行时 `DSV32_DUAL_BATCH`；prefill 和 batch=1 不拆分 |
| 8000 端口占用 | 改 `PORT`，客户端同时设置 `--base-url`；不要终止不属于本实验的服务 |
| profiler 输出为空或分析报完整步骤错误 | 等稳定 decode、保持服务独占，核对 trace 路径和 batch，再采集 |

## 10. 无外网目标机

可以将仓库打包复制到目标机；配置和 tokenizer 已包含，不需要下载模型权重。
但 CANN、NNAL、torch-npu、vLLM-Ascend 及其依赖仍需提前用适配目标系统/架构的镜像或离线包准备好。
启动器设置 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`；不会从 Hugging Face 自动拉取检查点。
