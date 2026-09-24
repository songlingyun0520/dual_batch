"""Run with the same Python/environment that will launch the server."""
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
errors = []
print(f"Python: {sys.version.split()[0]}, architecture: {platform.machine()}")
if not ((3, 10) <= sys.version_info[:2] < (3, 12)):
    errors.append("Use Python 3.10 or 3.11; original server used 3.11.6")
for name, expected in {"vllm": "0.18.0", "vllm-ascend": "0.18.0",
                       "torch": "2.9.0", "transformers": "4.57.6"}.items():
    try:
        version = metadata.version(name)
        print(f"{name}: {version}")
        if version.split("+")[0] != expected:
            errors.append(f"{name}: expected {expected}, got {version}")
    except metadata.PackageNotFoundError:
        errors.append(f"Missing {name}")
try:
    version = metadata.version("torch-npu")
    print(f"torch-npu: {version} (source host: 2.9.0; upstream recommends 2.9.0.post2)")
    if not version.startswith("2.9.0"):
        errors.append("torch-npu must be from the compatible 2.9.0 series")
except metadata.PackageNotFoundError:
    errors.append("Missing torch-npu")
entries = metadata.entry_points(group="vllm.general_plugins")
if not any(e.name == "dsv32_demo" and e.value == "demo_patch:register" for e in entries):
    errors.append("Plugin not installed: python3 -m pip install --no-deps --no-build-isolation -e .")
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    path = root / "model" / name
    if not path.is_file():
        errors.append(f"Missing {path}")
    else:
        json.loads(path.read_text())
try:
    import torch
    import torch_npu
    if not torch.npu.is_available():
        errors.append("No available NPU; check device mapping and driver/CANN environment")
    else:
        props = torch.npu.get_device_properties(0)
        print(f"NPU: {torch.npu.get_device_name(0)}, memory: {props.total_memory / 2**30:.1f} GiB")
        if props.total_memory < 60 * 2**30:
            errors.append("This fixed configuration requires a 64 GiB-class NPU")
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm.model_executor.models import deepseek_v2
    for cls, names in [(AscendSFAImpl, ["exec_kv", "forward", "indexer_select_post_process", "_execute_sparse_flash_attention_process"]),
                       (AscendSFAMetadataBuilder, ["build"]),
                       (NPUModelRunner, ["_allocate_kv_cache_tensors", "_reshape_kv_cache_tensors", "_update_states", "_zero_block_ids"]),
                       (deepseek_v2, ["SharedFusedMoE"])]:
        for name in names:
            if not hasattr(cls, name):
                errors.append(f"Incompatible internal API: {cls.__name__}.{name}")
except Exception as exc:
    errors.append(f"Import/device check failed: {type(exc).__name__}: {exc}")
if errors:
    print("\nPreflight FAILED:\n- " + "\n- ".join(errors), file=sys.stderr)
    sys.exit(1)
print("\nPreflight PASSED. This is an API/environment check; run the HTTP regression after startup.")
