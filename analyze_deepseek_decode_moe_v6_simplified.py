"""Analyze one rank's kernel_details.csv with enable_mlapo disabled.

The additive layer wall-time partition is: attention-layer start -> Indexer
start (all preceding preparation), Indexer kernel, Indexer end
-> SFA end (including deferred attention KV/Q, if present, plus layout and
sparse attention), SFA end -> MoE norm start (V up,
O projection and communication), then MoE norm start -> next layer start.
The O quantization/matmul span is also reported separately inside the output
path. All times are device timeline wall spans, not sums of kernel durations.

decode_original_order_timeline.csv additionally reports each stable decode
layer in the observed kernel order, with separate preparation, Indexer,
SFA, V/O, communication and MoE spans. A missing anchor stays unresolved;
it is never filled with a duration inferred from a neighboring layer.
The historical "original_order" filenames now describe the OBSERVED order
for both before-Indexer and after-Indexer attention preparation. Preparation
spans are diagnostics nested in the additive partition, not extra summands.

Capacity-aware revision: safe Indexer Q/block-table shape parsing, per-contiguous-
capacity regime detection and statistics, no TPOT across regimes or missing steps.
Capacity means execution rows (possibly graph padding), not live request count.
Single-token prefill can share decode shapes; shape classification is heuristic.
Core-count revision: preserve Block Num / Mix Block Num, report per-kernel
Cube/Vector launch configurations by capacity regime and observed module.
Launch counts do not measure simultaneous physical-core occupancy.

Recent-2K revision: V5/V5.1 UNSPLIT ring attention cache, full Indexer history.
Requires ring_profiler_support.py beside this file. Auto-detects mapping,
handles CSV/TSV and headerless excerpts, separates mapping/history/commit work.
Full 58-MoE stable-step statistics retain the previous conservative rules.

"""

import argparse
import os
import re
import json
from collections import Counter
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd
import ring_profiler_support as ring_support

def core_counts(core, block, mix):
    """Reported launch blocks for AIC/AIV, not simultaneous busy physical cores.

    Primary Block=0 may mean task-time L0 did not collect it. Secondary Mix=0
    is accepted only with a valid positive primary count. No inference for HCCL.
    """
    def count(value):
        try:
            value=Decimal(str(value).strip().strip('"'))
            if not value.is_finite():
                return np.nan
        except (InvalidOperation, ValueError, TypeError):
            return np.nan
        return int(value) if value is not None and value>=0 and value==value.to_integral_value() else np.nan
    primary,secondary=count(block),count(mix)
    kind=str(core).strip().upper()
    if kind in {"COMMUNICATION","AI_CPU","AICPU","CPU"}:
        return np.nan,np.nan,"not_applicable"
    if kind not in {"AI_CORE","AI_VECTOR_CORE","MIX_AIC","MIX_AIV"}:
        return np.nan,np.nan,"unavailable"
    if pd.isna(primary) or primary==0:
        return np.nan,np.nan,"unavailable"
    if kind=="AI_CORE":
        return primary,0,"reported"
    if kind=="AI_VECTOR_CORE":
        return 0,primary,"reported"
    status="partial" if pd.isna(secondary) else "reported"
    return (primary,secondary,status) if kind=="MIX_AIC" else (secondary,primary,status)


def core_profile(rows):
    """Per-kernel launch configurations. Counts of calls are never core sums."""
    counted=rows[rows.core_count_status.isin(["reported","partial","unavailable"])]
    known=counted[counted.core_count_status=="reported"]
    pairs=Counter((int(r.cube_cores),int(r.vector_cores)) for r in known.itertuples())
    configs=[dict(cube=c,vector=v,calls=n) for (c,v),n in sorted(pairs.items())]
    return dict(core_launch_profiles=json.dumps(configs),
                core_counted_calls=len(known),core_unknown_calls=len(counted)-len(known),
                core_not_applicable_calls=len(rows)-len(counted),
                cube_cores_per_kernel_min=counted.cube_cores.min(),
                cube_cores_per_kernel_max=counted.cube_cores.max(),
                vector_cores_per_kernel_min=counted.vector_cores.min(),
                vector_cores_per_kernel_max=counted.vector_cores.max())


def core_profile_text(profile):
    configs=json.loads(profile["core_launch_profiles"])
    text="; ".join(f"{p['cube']}/{p['vector']} ({p['calls']} calls)" for p in configs[:8]) or "N/A"
    if len(configs)>8:
        text+=f"; +{len(configs)-8} configurations (see CSV)"
    return ("Cube/Vector="+text+
            f"; unknown={profile['core_unknown_calls']}; non-core={profile['core_not_applicable_calls']}")


# ============================================================
# 0. User config
# ============================================================

parser = argparse.ArgumentParser(
    description="Analyze stable DeepSeek-V3.2 decode with enable_mlapo disabled."
)
parser.add_argument("--csv", default="kernel_details.csv", help="kernel_details.csv path")
parser.add_argument("--output-dir", default="deepseek_decode_no_mlapo_analysis")
parser.add_argument("--debug", action="store_true", help="Also write per-step/layer timing detail")
parser.add_argument("--kv-scheme", choices=["auto", "full", "ring2k"], default="auto",
                    help="auto detects V5/V5.1 recent-2K mapping; full uses legacy classification")
parser.add_argument("--analysis-scope", choices=["auto", "full", "excerpt"], default="auto",
                    help="auto falls back to observed ring modules if no complete 58-MoE forward exists")
parser.add_argument("--device-id", help="Select one device before constructing timelines")
parser.add_argument("--max-decode-capacity", type=int,
                    help="Optional analysis limit for stable decode capacity; larger executions remain in classification but not TPOT")
parser.add_argument(
    "--expect-attention-prep", choices=["auto", "before", "after"], default="auto",
    help="Auto-detect KV/Q preparation position; warn if it differs from this expectation",
)
args = parser.parse_args()
if args.max_decode_capacity is not None and args.max_decode_capacity < 1:
    parser.error('--max-decode-capacity must be positive')
CSV_PATH = args.csv
OUTPUT_DIR = args.output_dir

# DeepSeek-V3.2: 3 dense layers + 58 MoE layers, 0-based layer id = 3..60
FIRST_MOE_LAYER = 3
NUM_MOE_LAYERS = 58
LAST_MOE_LAYER = FIRST_MOE_LAYER + NUM_MOE_LAYERS - 1

# Decode detection is independent of input_len/output_len.
#
# Two-stage logic:
#   1) kernel structure confirms a complete/stable 58-MoE forward regime;
#   2) wall time locates the START of the final pure-Decode latency plateau.
#      After the plateau start is found, later structural matches are kept even
#      if their wall time is a true long-tail outlier.
MIN_DECODE_STEPS = 3
TAIL_REFERENCE_STEPS = 8
KERNEL_COUNT_ABS_TOL = 4
KERNEL_COUNT_REL_TOL = 0.001

# Pure-Decode plateau detector.
# The tolerance is adaptive: max(relative tolerance, robust MAD tolerance,
# minimum absolute tolerance). These values are intentionally conservative.
PLATEAU_MIN_REL_TOL = 0.03          # at least +/-3% around tail median
PLATEAU_MAD_MULT = 6.0              # robust sigma multiplier
PLATEAU_MIN_ABS_TOL_US = 200.0      # lower bound for very fast workloads
PLATEAU_CONFIRM_WINDOW = 5          # local window used to confirm plateau start
PLATEAU_CONFIRM_INLIER_FRAC = 0.80  # e.g. 4/5 points should be baseline-like

# Wall-time sanity check is diagnostic only. It never removes a step after the
# Decode plateau has started.
WALL_DIAGNOSTIC_RATIO_LOW = 0.45
WALL_DIAGNOSTIC_RATIO_HIGH = 2.50

# Forward boundary: after the previous step's last MoE Gating, the next first
# Gating is preceded by Indexers for dense layers 0/1/2 and MoE layer 3.
MIN_INDEXERS_BETWEEN_STEPS = 4

# Terminal stages used to find the real end of Layer60.
# We infer the dominant terminal stage from normal layers 3..59, then use the
# same pattern to stop Layer60 before LM Head / sampling / scheduler work.
PREFERRED_TERMINAL_STAGES = ["AddRmsNorm", "Add"]

# Export one decode-step/layer timeline. -1 means last detected decode step.
TARGET_DECODE_INDEX = -1
TARGET_MODEL_LAYER = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. Read CSV + detect columns
# ============================================================

# Preserve the original timestamp text even when CSV fields have no embedded
# quote characters. Parsing a ~1e15-us epoch into float before Decimal would
# already lose sub-microsecond precision and can invent boundary overlaps.
df = ring_support.read_kernel_table(CSV_PATH, args.device_id)
device_col = next((c for c in df.columns if str(c).lower() in {"device_id", "device id"}), None)
if device_col is not None and df[device_col].nunique(dropna=True) > 1:
    raise ValueError("CSV 含多个 Device_id；请先按单个 device/rank 拆分，避免交错层时间线。")

print("Columns:")
for i, c in enumerate(df.columns):
    print(i, c)


def find_col(candidates):
    for cand in candidates:
        for col in df.columns:
            if cand.lower() in str(col).lower():
                return col
    return None


name_col = find_col(["kernel name", "op name", "name", "op_name"])
stream_col = find_col(["stream id", "stream_id", "stream"])
start_col = find_col(["start time", "start_time", "start time(us)", "start time(µs)"])
duration_col = find_col(["duration", "task_duration"])
shape_col = find_col(["input shapes", "input_shapes", "input shape", "input dims"])

print("\nDetected:")
print("name       =", name_col)
print("stream     =", stream_col)
print("start      =", start_col)
print("duration   =", duration_col)
print("inputshape =", shape_col)

if None in [name_col, stream_col, start_col, duration_col]:
    raise RuntimeError("关键列名识别失败，请根据打印列名修改 find_col candidates。")

cols = [name_col, stream_col, start_col, duration_col]
if shape_col is not None:
    cols.append(shape_col)

data = df[cols].copy()
rename_map = {
    name_col: "name",
    stream_col: "stream",
    start_col: "start_raw",
    duration_col: "duration_raw",
}
if shape_col is not None:
    rename_map[shape_col] = "input_shapes"

data = data.rename(columns=rename_map)
# Exact matching prevents Block Num accidentally selecting Mix Block Num.
def exact_column(*choices):
    normalized={re.sub(r"[\s_]", "", c.lower()) for c in choices}
    return next((c for c in df.columns if re.sub(r"[\s_]", "", str(c).lower()) in normalized), None)

for target,choices in {
    "core": ("Accelerator Core",),
    "block_num_raw": ("Block Num", "Block Dim"),
    "mix_block_num_raw": ("Mix Block Num", "Mix Block Dim"),
    "output_shapes": ("Output Shapes",),
    "model_id": ("Model ID",),
    "op_state": ("OP State",),
}.items():
    source=exact_column(*choices)
    data[target]=df[source].fillna("") if source is not None else ""
data["source_row"]=df['_source_record'].to_numpy()
core_columns=["cube_cores","vector_cores","core_count_status"]
data[core_columns]=pd.DataFrame(
    [core_counts(c,b,m) for c,b,m in zip(data.core,data.block_num_raw,data.mix_block_num_raw)],
    index=data.index,columns=core_columns)

if "input_shapes" not in data.columns:
    data["input_shapes"] = ""


# ============================================================
# 2. Clean + high-precision relative timestamp
# ============================================================

data["name"] = data["name"].astype(str)
data["input_shapes"] = data["input_shapes"].fillna("").astype(str)
data["stream"] = pd.to_numeric(data["stream"], errors="coerce")
data["start_raw"] = (
    data["start_raw"].astype(str).str.replace('"', "", regex=False).str.strip()
)


def parse_decimal(x):
    try:
        return Decimal(x)
    except (InvalidOperation, ValueError, TypeError):
        return None


data["_start_dec"] = data["start_raw"].map(parse_decimal)
data["duration_us"] = pd.to_numeric(data["duration_raw"], errors="coerce")
data = data.dropna(subset=["_start_dec", "duration_us"]).copy()
data = data[data['_start_dec'].map(lambda v: v.is_finite()) & data.duration_us.ge(0)].copy()
if data.empty:
    raise ValueError("没有有效的时间戳/持续时间记录")

# Normalize duration to us.
duration_name = str(duration_col).lower()
if "ms" in duration_name and "us" not in duration_name and "µs" not in duration_name:
    data["duration_us"] *= 1000.0
elif "ns" in duration_name:
    data["duration_us"] /= 1000.0

# Subtract a Decimal base first, then convert to float.
t0_dec = min(data["_start_dec"])
data["start"] = data["_start_dec"].map(lambda x: float(x - t0_dec))
data["end"] = data["start"] + data["duration_us"]
data = data.sort_values(["start", "stream"], na_position="last").reset_index(drop=True)


# ============================================================
# 3. Kernel classification
# ============================================================

def classify(name):
    n = str(name).lower()

    # Attention / DSA
    if "mla_preprocess" in n:
        return "MLA_Preprocess"
    if "lightningindexer" in n:
        return "LightningIndexer"
    if "sparseflashattention" in n:
        return "SparseFlashAttention"

    # Routed MoE
    if "moegatingtopk" in n:
        return "Gating"
    if "moedistributedispatch" in n or "moeinitroutingcustom" in n:
        return "Dispatch"
    if "groupedmatmulswiglu" in n:
        return "Expert_GEMM1"
    if "groupedmatmul" in n and "swiglu" not in n:
        return "Expert_GEMM2"
    if "moedistributecombine" in n or "moetokenunpermute" in n:
        return "Combine"

    # Communication
    # AI-CPU collective kernels are background/runtime tasks on another stream.
    # Their lifetime is not the corresponding HCCL operation's wall span.
    # Keep them visible, but never attribute them to O communication or remap.
    compact = n.replace("_", "")
    if "aicpu" in compact and any(
        op in compact for op in ("allreduce", "allgather", "reducescatter", "alltoall")
    ):
        return "CommunicationRuntime"
    if "allgather" in n:
        return "AllGather"
    if "allreduce" in n:
        return "AllReduce"
    if "reducescatter" in n or "reduce_scatter" in n:
        return "ReduceScatter"
    if "alltoall" in n or "all_to_all" in n:
        return "AllToAll"

    # Norm / Quant / GEMM
    if "addrmsnormbias" in n or "addrmsnorm" in n:
        return "AddRmsNorm"
    if "dequantswigluquant" in n:
        return "DequantSwiGLUQuant"
    if "dynamicquant" in n:
        return "DynamicQuant"
    if "quantbatchmatmul" in n or "quantmatmul" in n:
        return "QuantMatmul"
    if "batch_matmul_transpose" in n or "batchmatmultranspose" in n or "transposebatchmatmul" in n:
        return "BatchMatmulTranspose"
    if "swiglu" in n:
        return "SwiGLU"
    if "matmul" in n:
        return "Matmul"
    if n.strip() == "add" or "aclnnadd_" in n:
        return "Add"

    return "Other"


data["stage"] = data["name"].apply(classify)
if (data["stage"] == "MLA_Preprocess").any():
    raise ValueError("检测到 MLA_Preprocess；此脚本用于 enable_mlapo=false，请使用备份的 MLAPO 分析脚本。")


# ============================================================
# 4. Conservative HCCL / AivKernel dedup
# ============================================================

def is_explicit_comm(name):
    n = str(name).lower()
    return any(
        k in n
        for k in [
            "hcom_",
            "allreduce",
            "allgather",
            "reducescatter",
            "reduce_scatter",
            "alltoall",
            "all_to_all",
        ]
    )


def is_generic_aiv_shadow(name):
    n = str(name).lower().strip()
    return n == "aivkernel" or n.startswith("aivkernel ")


def dedup_priority(name):
    if is_explicit_comm(name):
        return 0
    if not is_generic_aiv_shadow(name):
        return 1
    return 2


# Exact duplicate rows.
data = data.drop_duplicates(
    subset=["name", "stream", "start", "duration_us"], keep="first"
).copy()

# Same stream + identical start/duration can be a HCCL/AivKernel mirror.
data["_dedup_priority"] = data["name"].map(dedup_priority)
data = (
    data.sort_values(["stream", "start", "duration_us", "_dedup_priority"])
    .drop_duplicates(subset=["stream", "start", "duration_us"], keep="first")
    .drop(columns=["_dedup_priority"])
    .sort_values(["start", "stream"], na_position="last")
    .reset_index(drop=True)
)

print(f"\nAfter cleaning: {len(data)} kernel records")

RING_MODE = args.kv_scheme == 'ring2k' or (
    args.kv_scheme == 'auto' and ring_support.ring_signature(data)
)
RING_OBSERVATIONS = []
RING_INDEXER_INFO = {}
if RING_MODE or args.analysis_scope == 'excerpt':
    RING_OBSERVATIONS = ring_support.export_observed(data, OUTPUT_DIR, CSV_PATH, t0_dec)
    RING_INDEXER_INFO = {info['indexer_start_us']: info for info, _, _, _ in RING_OBSERVATIONS
                        if np.isfinite(info['indexer_start_us'])}
    if any(info['phase'] == 'unsupported_split' for info, _, _, _ in RING_OBSERVATIONS):
        raise ValueError('检测到同层多次 Indexer/SFA；本版本适用于 V5/V5.1 未拆分模式，不能套用到双batch。')

def incomplete_trace(message):
    if RING_OBSERVATIONS and args.analysis_scope != 'full':
        print('[INFO] '+message)
        print('仅输出 ring_observed_* 片段分析；不生成稳定 Decode TPOT 或绝对模型层号。')
        raise SystemExit(0)
    raise RuntimeError(message)

if args.analysis_scope == 'excerpt':
    incomplete_trace('用户选择 excerpt；模块耗时按实际可见边界统计。')


# ============================================================
# 5. Helpers
# ============================================================

def nearest_previous(values, t):
    pos = np.searchsorted(values, t, side="left") - 1
    if pos < 0:
        return None
    return float(values[pos])


def next_after(values, t):
    pos = np.searchsorted(values, t, side="right")
    if pos >= len(values):
        return None
    return float(values[pos])


def find_attention_starts(indexer_starts, norm_starts):
    """Last norm between consecutive Indexers is the attention-layer start.

    Preserve NaN for a clipped first layer; never substitute the first kernel
    in a pasted fragment as its layer start.
    """
    result = []
    for i, indexer_start in enumerate(indexer_starts):
        previous_indexer = indexer_starts[i - 1] if i else -np.inf
        candidates = norm_starts[
            (norm_starts > previous_indexer) & (norm_starts < indexer_start)
        ]
        result.append(float(candidates[-1]) if len(candidates) else np.nan)
    return np.asarray(result, dtype=float)


def robust_mode_int(values):
    vals = [int(v) for v in values if v is not None and not pd.isna(v) and int(v) > 0]
    if not vals:
        return np.nan
    c = Counter(vals)
    best_count = max(c.values())
    best_vals = sorted(k for k, v in c.items() if v == best_count)
    return int(best_vals[0])


def robust_mode_str(values):
    vals = [str(v) for v in values if v is not None and str(v)]
    if not vals:
        return None
    c = Counter(vals)
    best_count = max(c.values())
    best_vals = sorted(k for k, v in c.items() if v == best_count)
    return best_vals[0]


def robust_wall_reference(values):
    """Return (median, MAD, robust_sigma) for finite wall-time samples."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    robust_sigma = 1.4826 * mad
    return med, mad, robust_sigma


def final_true_run_start(mask):
    """Start index of the final contiguous True run; None if the tail is False."""
    arr = np.asarray(mask, dtype=bool)
    if len(arr) == 0 or not arr[-1]:
        return None
    i = len(arr) - 1
    while i > 0 and arr[i - 1]:
        i -= 1
    return int(i)


def find_decode_plateau_start(
    walls,
    structure_ok,
    tail_reference_steps=8,
    min_decode_steps=3,
    min_rel_tol=0.03,
    mad_mult=6.0,
    min_abs_tol_us=200.0,
    confirm_window=5,
    confirm_inlier_frac=0.80,
):
    """
    Find the start of the final pure-Decode latency plateau.

    Important design rule:
      wall time is used ONLY to locate the plateau start. Once the start is
      found, later steps are NOT filtered by wall time, so real Decode long-tail
      steps remain in the sample set.

    Returns a dict with start index, baseline/tolerance diagnostics and an
    inlier mask used only for onset detection.
    """
    walls = np.asarray(walls, dtype=float)
    structure_ok = np.asarray(structure_ok, dtype=bool)
    n = len(walls)

    result = {
        "start_idx": None,
        "structural_suffix_start": None,
        "ref_wall_us": np.nan,
        "mad_us": np.nan,
        "robust_sigma_us": np.nan,
        "tol_us": np.nan,
        "tol_rel": np.nan,
        "inlier_mask": np.zeros(n, dtype=bool),
        "confidence": "none",
    }

    if n == 0:
        return result

    structural_start = final_true_run_start(structure_ok)
    result["structural_suffix_start"] = structural_start
    if structural_start is None:
        return result

    structural_len = n - structural_start
    if structural_len < min_decode_steps:
        return result

    tail_n = min(int(tail_reference_steps), structural_len)
    tail_walls = walls[n - tail_n : n]
    ref, mad, robust_sigma = robust_wall_reference(tail_walls)
    if not np.isfinite(ref) or ref <= 0:
        return result

    tol_us = max(
        float(min_abs_tol_us),
        float(ref) * float(min_rel_tol),
        float(robust_sigma) * float(mad_mult),
    )

    inlier = structure_ok & np.isfinite(walls) & (np.abs(walls - ref) <= tol_us)

    result.update(
        {
            "ref_wall_us": ref,
            "mad_us": mad,
            "robust_sigma_us": robust_sigma,
            "tol_us": tol_us,
            "tol_rel": tol_us / ref,
            "inlier_mask": inlier,
        }
    )

    # Earliest point that itself looks baseline-like and whose following local
    # window is dominated by baseline-like points. This rejects a gradually
    # shrinking prefill/mixed region while tolerating an occasional Decode
    # long-tail inside the confirmation window.
    latest_start = n - min_decode_steps
    for i in range(structural_start, latest_start + 1):
        if not inlier[i]:
            continue

        j = min(n, i + int(confirm_window))
        win_len = j - i
        if win_len < min_decode_steps:
            continue

        required = max(
            min_decode_steps,
            int(np.ceil(float(confirm_inlier_frac) * win_len)),
        )
        required = min(required, win_len)

        win_inliers = int(inlier[i:j].sum())
        win_med = float(np.median(walls[i:j]))
        win_med_ok = abs(win_med - ref) <= tol_us

        if win_inliers >= required and win_med_ok:
            result["start_idx"] = int(i)
            result["confidence"] = (
                "high" if win_len >= confirm_window and win_inliers == win_len
                else "medium"
            )
            return result

    # Conservative fallback: find the first run of min_decode_steps wall inliers
    # inside the final structural suffix.
    for i in range(structural_start, latest_start + 1):
        if np.all(inlier[i : i + min_decode_steps]):
            result["start_idx"] = int(i)
            result["confidence"] = "fallback"
            return result

    return result


def find_terminal_kernel(layer_data, combine_end, preferred_stages=None):
    """
    Find a measured terminal residual/norm kernel after Combine.
    Returns (stage, end_time) or (None, None).
    """
    if preferred_stages is None:
        preferred_stages = PREFERRED_TERMINAL_STAGES

    post = layer_data[layer_data["start"] >= combine_end].copy()
    if post.empty:
        return None, None

    for stage in preferred_stages:
        tmp = post[post["stage"] == stage]
        if not tmp.empty:
            # The final occurrence is safer for a normal layer.
            row = tmp.sort_values("end").iloc[-1]
            return stage, float(row["end"])

    return None, None


# ============================================================
# 6. Detect complete forwards by 58 Gating + Indexer structure
#    Independent of input_len/output_len.
# ============================================================

gating_rows = data[data["stage"] == "Gating"].sort_values("start").reset_index()
indexer_rows_all = data[data["stage"] == "LightningIndexer"].sort_values("start").reset_index()
norm_rows_all = data[data["stage"] == "AddRmsNorm"].sort_values("start").reset_index()

if len(gating_rows) == 0:
    incomplete_trace("没有找到 MoeGatingTopK / Gating。")
if len(indexer_rows_all) == 0:
    incomplete_trace("没有找到 LightningIndexer；请确认这是关闭 enable_mlapo 的采样。")
if len(norm_rows_all) == 0:
    incomplete_trace("没有找到 AddRmsNormBias，无法可靠定位 attention 层起点。")

indexer_starts = indexer_rows_all["start"].to_numpy(dtype=float)
norm_starts = norm_rows_all["start"].to_numpy(dtype=float)
# The last AddRmsNormBias before each Indexer is the attention-side layer
# start. The earlier AddRmsNormBias between two Indexers belongs to MoE.
attention_starts = find_attention_starts(indexer_starts, norm_starts)
if np.isnan(attention_starts).any():
    print(f"[WARN] {np.isnan(attention_starts).sum()} 个 Indexer 缺少前置 AddRmsNormBias；相关 forward 会被丢弃。")
attention_starts = np.unique(attention_starts[np.isfinite(attention_starts)])
gating_starts = gating_rows["start"].to_numpy(dtype=float)

boundary_indices = []
for i, g in enumerate(gating_starts):
    prev_g = -np.inf if i == 0 else gating_starts[i - 1]
    indexers_between = int(np.sum((indexer_starts > prev_g) & (indexer_starts < g)))
    if indexers_between >= MIN_INDEXERS_BETWEEN_STEPS:
        boundary_indices.append(i)

if not boundary_indices:
    incomplete_trace(
        "没有识别到 forward step 边界。检查 Indexer 分类及 MIN_INDEXERS_BETWEEN_STEPS。"
    )

step_groups = []
for k, start_idx in enumerate(boundary_indices):
    end_idx = boundary_indices[k + 1] if k + 1 < len(boundary_indices) else len(gating_rows)
    grp = gating_rows.iloc[start_idx:end_idx].copy()

    if len(grp) == NUM_MOE_LAYERS:
        step_groups.append((k, grp))
    else:
        print(
            f"[WARN] 丢弃不完整/异常 forward: gating_count={len(grp)}, "
            f"expected={NUM_MOE_LAYERS}, range=[{start_idx},{end_idx})"
        )

if not step_groups:
    incomplete_trace("没有找到包含完整 58 个 MoE Gating 的 forward step。")

print(f"Detected complete forward steps: {len(step_groups)}")


# ============================================================
# 7. Rebuild 58 MoE layers for each forward
#
# Bug fix #1:
#   Layer3..59: current attention AddRmsNormBias -> next layer start
#   Layer60: DO NOT use next step Layer0 start as end.
#            Infer the normal post-Combine terminal pattern from Layer3..59,
#            then stop Layer60 at the matching measured residual/norm kernel.
# ============================================================

step_records = []
moe_layer_records = []

for raw_step_id, grp in step_groups:
    grp = grp.sort_values("start").reset_index(drop=True)
    if len(grp) != NUM_MOE_LAYERS:
        continue

    regular_layers = []
    valid = True

    # --------------------------------------------------------
    # 7.1 Layer3..59: use next attention start as a measured boundary
    # --------------------------------------------------------
    for moe_idx in range(NUM_MOE_LAYERS - 1):
        gating = grp.iloc[moe_idx]
        model_layer = FIRST_MOE_LAYER + moe_idx
        gating_start = float(gating["start"])

        layer_start = nearest_previous(attention_starts, gating_start)
        if layer_start is None:
            valid = False
            break

        layer_end = next_after(attention_starts, layer_start)
        if layer_end is None or layer_end <= layer_start:
            valid = False
            break

        ld = data[(data["start"] >= layer_start) & (data["start"] < layer_end)].copy()
        combine_rows = ld[ld["stage"] == "Combine"]
        if len(combine_rows) == 0:
            print(
                f"[WARN] raw_step={raw_step_id} layer={model_layer}: 缺少 Combine，丢弃该step"
            )
            valid = False
            break

        combine_end = float(combine_rows["end"].max())
        terminal_stage, terminal_end = find_terminal_kernel(ld, combine_end)

        regular_layers.append(
            {
                "raw_step_id": raw_step_id,
                "moe_index": moe_idx,
                "model_layer": model_layer,
                "start": layer_start,
                "end": layer_end,
                "gating_start": gating_start,
                "combine_end": combine_end,
                "post_combine_tail_us": layer_end - combine_end,
                "terminal_stage": terminal_stage,
                "terminal_end": terminal_end,
                "terminal_to_next_attention_gap_us": (
                    layer_end - terminal_end if terminal_end is not None else np.nan
                ),
                "layer_end_source": "next_attention_start",
                "layer_end_measured": True,
            }
        )

    if not valid or len(regular_layers) != NUM_MOE_LAYERS - 1:
        continue

    # --------------------------------------------------------
    # 7.2 Learn normal Layer tail pattern from Layer3..59
    # --------------------------------------------------------
    tail_ref_us = float(np.median([x["post_combine_tail_us"] for x in regular_layers]))

    gap_vals = [
        x["terminal_to_next_attention_gap_us"]
        for x in regular_layers
        if pd.notna(x["terminal_to_next_attention_gap_us"])
        and x["terminal_to_next_attention_gap_us"] >= 0
    ]
    terminal_gap_ref_us = float(np.median(gap_vals)) if gap_vals else 0.0

    terminal_stage_ref = robust_mode_str(
        [x["terminal_stage"] for x in regular_layers if x["terminal_stage"] is not None]
    )

    # --------------------------------------------------------
    # 7.3 Layer60: stop before LM Head/sampling/next-step work
    # --------------------------------------------------------
    gating = grp.iloc[-1]
    moe_idx = NUM_MOE_LAYERS - 1
    model_layer = LAST_MOE_LAYER
    gating_start = float(gating["start"])
    layer_start = nearest_previous(attention_starts, gating_start)

    if layer_start is None:
        continue

    # This is only an upper bound. It may be next step Layer0 attention and MUST NOT
    # be used directly as Layer60's end.
    upper_bound = next_after(attention_starts, layer_start)
    if upper_bound is None:
        upper_bound = float(data["end"].max())

    search = data[
        (data["start"] >= layer_start)
        & (data["start"] < upper_bound)
    ].copy()

    combine_rows = search[search["stage"] == "Combine"]
    if len(combine_rows) == 0:
        print(f"[WARN] raw_step={raw_step_id} layer=60: 缺少 Combine，丢弃该step")
        continue

    combine_end = float(combine_rows["end"].max())

    # Find all measured terminal-stage candidates after Combine, then choose the
    # one whose post-Combine tail best matches normal Layer3..59 behavior.
    candidate_rows = pd.DataFrame()
    if terminal_stage_ref is not None:
        candidate_rows = search[
            (search["start"] >= combine_end)
            & (search["stage"] == terminal_stage_ref)
        ].copy()

    # If dominant terminal stage cannot be found, try preferred stages.
    if candidate_rows.empty:
        for st in PREFERRED_TERMINAL_STAGES:
            tmp = search[
                (search["start"] >= combine_end)
                & (search["stage"] == st)
            ].copy()
            if not tmp.empty:
                candidate_rows = tmp
                terminal_stage_ref = st
                break

    layer60_end_source = None
    layer60_measured = False
    layer60_terminal_end = np.nan

    if not candidate_rows.empty:
        candidate_rows = candidate_rows.copy()
        candidate_rows["candidate_layer_end"] = (
            candidate_rows["end"].astype(float) + terminal_gap_ref_us
        )
        candidate_rows["candidate_layer_end"] = np.minimum(
            candidate_rows["candidate_layer_end"].to_numpy(dtype=float),
            np.full(len(candidate_rows), upper_bound - 1e-9),
        )
        candidate_rows["candidate_tail_us"] = (
            candidate_rows["candidate_layer_end"] - combine_end
        )
        candidate_rows["tail_error_us"] = np.abs(
            candidate_rows["candidate_tail_us"] - tail_ref_us
        )

        best = candidate_rows.sort_values(["tail_error_us", "end"]).iloc[0]
        layer_end = float(best["candidate_layer_end"])
        layer60_terminal_end = float(best["end"])
        layer60_end_source = f"terminal_stage:{terminal_stage_ref}+median_gap"
        # The terminal kernel is observed, but the layer boundary adds a
        # median gap learned from other layers. It is therefore estimated.
        layer60_measured = False
    else:
        # Last-resort fallback: use the median normal post-Combine tail.
        # This is explicitly marked as estimated and a warning is printed.
        layer_end = min(combine_end + tail_ref_us, upper_bound - 1e-9)
        layer60_end_source = "estimated_from_layer3_59_median_tail"
        print(
            f"[WARN] raw_step={raw_step_id} layer=60: 未找到可测 terminal stage，"
            f"使用 median tail={tail_ref_us:.3f} us 估计结束边界"
        )

    if layer_end <= combine_end or layer_end <= layer_start:
        print(f"[WARN] raw_step={raw_step_id} layer=60: 结束边界异常，丢弃该step")
        continue

    layer60 = {
        "raw_step_id": raw_step_id,
        "moe_index": moe_idx,
        "model_layer": model_layer,
        "start": layer_start,
        "end": layer_end,
        "gating_start": gating_start,
        "combine_end": combine_end,
        "terminal_stage": terminal_stage_ref,
        "terminal_end": layer60_terminal_end,
        "terminal_to_next_attention_gap_us": np.nan,
        "layer_end_source": layer60_end_source,
        "layer_end_measured": layer60_measured,
    }

    this_layers = regular_layers + [layer60]

    ids = [x["model_layer"] for x in this_layers]
    if ids != list(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1)):
        print(f"[WARN] raw_step={raw_step_id}: 58层编号结构异常，丢弃")
        continue

    moe_layer_records.extend(this_layers)

    step_start = float(this_layers[0]["start"])
    step_end = float(this_layers[-1]["end"])
    sd = data[(data["start"] >= step_start) & (data["start"] < step_end)]

    stage_counts = sd["stage"].value_counts()

    step_records.append(
        {
            "raw_step_id": raw_step_id,
            "moe58_start": step_start,
            "moe58_end": step_end,
            "moe58_wall_us": step_end - step_start,
            "moe58_kernel_count": len(sd),
            "gating_count": int(stage_counts.get("Gating", 0)),
            "dispatch_count": int(stage_counts.get("Dispatch", 0)),
            "expert_gemm1_count": int(stage_counts.get("Expert_GEMM1", 0)),
            "expert_gemm2_count": int(stage_counts.get("Expert_GEMM2", 0)),
            "combine_count": int(stage_counts.get("Combine", 0)),
            "layer60_end_source": layer60_end_source,
            "layer60_end_measured": layer60_measured,
            "normal_post_combine_tail_ref_us": tail_ref_us,
            "normal_terminal_gap_ref_us": terminal_gap_ref_us,
            "normal_terminal_stage_ref": terminal_stage_ref,
        }
    )

step_summary = pd.DataFrame(step_records)
moe_layer_summary = pd.DataFrame(moe_layer_records)

if step_summary.empty or moe_layer_summary.empty:
    incomplete_trace("58层 MoE step/layer 重建失败。")

# Re-number valid steps consecutively.
valid_raw_ids = step_summary["raw_step_id"].tolist()
step_id_map = {raw: i for i, raw in enumerate(valid_raw_ids)}
step_summary["step_id"] = step_summary["raw_step_id"].map(step_id_map)
moe_layer_summary = moe_layer_summary[
    moe_layer_summary["raw_step_id"].isin(valid_raw_ids)
].copy()
moe_layer_summary["step_id"] = moe_layer_summary["raw_step_id"].map(step_id_map)

step_summary = step_summary.sort_values("step_id").reset_index(drop=True)
moe_layer_summary = moe_layer_summary.sort_values(
    ["step_id", "model_layer"]
).reset_index(drop=True)

step_summary["next_step_interval_us"] = (
    step_summary["moe58_start"].shift(-1) - step_summary["moe58_start"]
)


# ============================================================
# 8. Decode detection: kernel structure + final latency plateau
#
# Bug fix #3:
#   Kernel structure alone cannot distinguish prefill/mixed-forward from pure
#   Decode because both may execute the same 58 MoE layers and the same number
#   of kernels.
#
#   New logic:
#     A) use the tail to learn the stable Decode kernel structure;
#     B) within the final structural run, use robust wall-time statistics to
#        locate the START of the pure-Decode latency plateau;
#     C) after that start, keep every structural match, regardless of wall time.
#        Therefore a real Decode long-tail is preserved rather than filtered.
# ============================================================

def shape_list(value):
    """Parse CANN shapes, including literal quoting left by CSV/Excel exports.

    Preserve empty argument slots; never eval text or extract arbitrary digits.
    """
    def unquote(text):
        text = text.strip()
        while len(text) >= 2 and text[0] in (chr(34), chr(39)) and text[-1] == text[0]:
            text = text[1:-1].strip()
        return text
    result = []
    for part in unquote(str(value)).split(";"):
        part = unquote(part)
        if len(part) >= 2 and (part[0], part[-1]) in (("[", "]"), ("(", ")")):
            part = part[1:-1].strip()
        if re.fullmatch(r"\d+(?:\s*,\s*\d+)*", part):
            result.append(tuple(int(x) for x in part.split(",")))
        else:
            result.append(())
    return result

def detect_capacity_regime(step_summary):
    print(f"Capacity regime {int(step_summary.iloc[0].regime_id)}: {int(step_summary.iloc[0].input_capacity)} tokens, {len(step_summary)} candidate steps")
    step_summary["step_kind"] = "other"
    step_summary["decode_detection_method"] = ""

    n_steps = len(step_summary)
    tail_n = min(TAIL_REFERENCE_STEPS, n_steps)
    tail = step_summary.tail(tail_n).copy()

    # ---------- A. Stable kernel-structure reference ----------
    ref_kernel_count = robust_mode_int(tail["moe58_kernel_count"].tolist())
    if pd.isna(ref_kernel_count):
        raise RuntimeError("无法建立 Decode kernel-count 尾部参考。")

    kernel_count_tol = max(
        int(KERNEL_COUNT_ABS_TOL),
        int(np.ceil(float(ref_kernel_count) * KERNEL_COUNT_REL_TOL)),
    )

    critical_cols = [
        "gating_count",
        "dispatch_count",
        "expert_gemm1_count",
        "expert_gemm2_count",
        "combine_count",
    ]
    critical_refs = {c: robust_mode_int(tail[c].tolist()) for c in critical_cols}

    step_summary["decode_ref_kernel_count"] = int(ref_kernel_count)
    step_summary["kernel_count_delta"] = (
        step_summary["moe58_kernel_count"] - int(ref_kernel_count)
    ).abs()
    step_summary["kernel_structure_ok"] = (
        step_summary["kernel_count_delta"] <= kernel_count_tol
    )

    for c in critical_cols:
        ref = critical_refs[c]
        if not pd.isna(ref):
            step_summary["kernel_structure_ok"] &= (step_summary[c] == int(ref))

    # ---------- B. Locate the final pure-Decode latency plateau ----------
    plateau = find_decode_plateau_start(
        walls=step_summary["moe58_wall_us"].to_numpy(dtype=float),
        structure_ok=step_summary["kernel_structure_ok"].to_numpy(dtype=bool),
        tail_reference_steps=TAIL_REFERENCE_STEPS,
        min_decode_steps=MIN_DECODE_STEPS,
        min_rel_tol=PLATEAU_MIN_REL_TOL,
        mad_mult=PLATEAU_MAD_MULT,
        min_abs_tol_us=PLATEAU_MIN_ABS_TOL_US,
        confirm_window=PLATEAU_CONFIRM_WINDOW,
        confirm_inlier_frac=PLATEAU_CONFIRM_INLIER_FRAC,
    )

    ref_wall_us = float(plateau["ref_wall_us"])
    wall_tol_us = float(plateau["tol_us"])
    wall_tol_rel = float(plateau["tol_rel"])

    step_summary["decode_plateau_ref_wall_us"] = ref_wall_us
    step_summary["decode_plateau_tol_us"] = wall_tol_us
    step_summary["decode_plateau_tol_rel"] = wall_tol_rel
    step_summary["wall_plateau_inlier"] = plateau["inlier_mask"]
    step_summary["wall_ratio_to_decode_tail"] = (
        step_summary["moe58_wall_us"] / ref_wall_us
        if np.isfinite(ref_wall_us) and ref_wall_us > 0
        else np.nan
    )
    step_summary["wall_time_sanity_ok"] = (
        (step_summary["wall_ratio_to_decode_tail"] >= WALL_DIAGNOSTIC_RATIO_LOW)
        & (step_summary["wall_ratio_to_decode_tail"] <= WALL_DIAGNOSTIC_RATIO_HIGH)
    )
    step_summary["decode_plateau_start"] = False
    step_summary["decode_after_plateau_start"] = False

    plateau_start = plateau["start_idx"]
    if plateau_start is None:
        print("\n[WARN] 没有可靠识别到 pure-Decode latency plateau。")
        print("       请检查 forward_classification.csv 中 wall/kernel structure。")
    else:
        plateau_start = int(plateau_start)
        step_summary.loc[plateau_start, "decode_plateau_start"] = True
        step_summary.loc[plateau_start:, "decode_after_plateau_start"] = True

        # ---------- C. Keep all structural matches after onset ----------
        # Do NOT filter by wall time here: this intentionally preserves true
        # long-tail Decode steps.
        decode_mask = (
            step_summary.index.to_numpy() >= plateau_start
        ) & step_summary["kernel_structure_ok"].to_numpy(dtype=bool)

        method = (
            "final_decode_plateau("
            f"start_step={int(step_summary.loc[plateau_start, 'step_id'])},"
            f"ref_wall_us={ref_wall_us:.3f},"
            f"tol_us={wall_tol_us:.3f},"
            f"tol_rel={wall_tol_rel:.4f},"
            f"confidence={plateau['confidence']};"
            f"ref_kernels={int(ref_kernel_count)},kernel_tol={kernel_count_tol})"
        )

        step_summary.loc[decode_mask, "step_kind"] = "decode"
        step_summary.loc[decode_mask, "decode_detection_method"] = method

        # Structural mismatch after plateau onset is suspicious and is deliberately
        # not labelled Decode; surface it loudly for manual inspection.
        post_onset_bad = step_summary.loc[plateau_start:, "kernel_structure_ok"] == False
        if post_onset_bad.any():
            bad_ids = step_summary.loc[
                step_summary.index[plateau_start:][post_onset_bad.to_numpy()], "step_id"
            ].tolist()
            print(
                f"[WARN] plateau起点之后存在 kernel structure 不匹配 step: {bad_ids}; "
                "这些step未纳入Decode统计。"
            )

    return step_summary

# Extract all 58 observed Indexer shapes; capacity is NOT live request count.
shape_diagnostics = []
layer_shapes = {}
li_data = data[data.stage == "LightningIndexer"].sort_values("start")
li_times = li_data.start.to_numpy()
for layer in moe_layer_summary.itertuples():
    part = li_data.iloc[np.searchsorted(li_times, layer.start):np.searchsorted(li_times, layer.gating_start)]
    capacities, requests = [], []
    for row in part.itertuples():
        shapes = shape_list(row.input_shapes)
        q = shapes[0][0] if shapes and shapes[0] else None
        req = shapes[5][0] if len(shapes)>5 and shapes[5] else None
        capacities.append(q)
        requests.append(req)
        shape_diagnostics.append(dict(step_id=layer.step_id, model_layer=layer.model_layer,
            input_shapes_repr=repr(row.input_shapes), parsed_shapes=json.dumps(shapes),
            q_tokens=q, request_rows=req, parse_ok=q is not None and req is not None))
    if len(part) != 1:
        phase = "unsupported_indexer_count"
    elif capacities[0] is None or requests[0] is None or min(capacities[0], requests[0]) <= 0:
        phase = "unknown_shape"
    elif capacities != requests:
        phase = "prefill_or_dp_padding"
    else:
        phase = "decode_candidate"
    if RING_MODE and len(part) == 1:
        observed_phase = RING_INDEXER_INFO.get(float(part.iloc[0].start), {}).get('phase')
        if observed_phase == 'prefill_or_mixed':
            phase = 'prefill_or_mixed'
        elif observed_phase == 'decode' and capacities[0] is not None and requests[0] is not None:
            # Native decode modulo path is stronger evidence than q==request
            # rows when DP adds token-only padding. Capacity still != live B.
            phase = 'decode_candidate'
    layer_shapes[(layer.step_id,layer.model_layer)] = (capacities, requests, phase)
shape_diagnostics = pd.DataFrame(shape_diagnostics)
if not shape_diagnostics.empty:
    failures=int((~shape_diagnostics.parse_ok).sum())
    print(f"Indexer shape parse failures: {failures}/{len(shape_diagnostics)}")
step_summary["input_capacity"] = np.nan
step_summary["shape_signature"] = ""
step_summary["shape_phase"] = "unknown_shape"
for idx, row in step_summary.iterrows():
    items=[layer_shapes[(row.step_id, layer)] for layer in range(FIRST_MOE_LAYER,LAST_MOE_LAYER+1)]
    signatures=sorted({tuple(x[0]) for x in items},key=str)
    step_summary.at[idx,"shape_signature"] = str(signatures)
    phases={x[2] for x in items}
    if phases == {"decode_candidate"} and len(signatures)==1:
        step_summary.at[idx,"input_capacity"] = signatures[0][0]
        step_summary.at[idx,"shape_phase"] = "decode_candidate"
    elif "prefill_or_mixed" in phases:
        step_summary.at[idx,"shape_phase"] = "prefill_or_mixed"
    elif "prefill_or_dp_padding" in phases:
        step_summary.at[idx,"shape_phase"] = "prefill_or_dp_padding"
    elif phases == {"decode_candidate"}:
        step_summary.at[idx,"shape_phase"] = "mixed_capacity"
    elif "unsupported_indexer_count" in phases:
        step_summary.at[idx,"shape_phase"] = "unsupported_indexer_count"

# A user-specified comparison limit does not redefine the execution phase.
# DP padding can invoke a decode path at a prefill-sized token capacity.
step_summary['decode_capacity_excluded'] = False
if args.max_decode_capacity is not None:
    step_summary['decode_capacity_excluded'] = (
        step_summary.shape_phase.eq('decode_candidate')
        & step_summary.input_capacity.gt(args.max_decode_capacity)
    )
    print(f"Stable decode capacity limit: {args.max_decode_capacity}; "
          f"excluded {int(step_summary.decode_capacity_excluded.sum())} candidate steps (retained in classification)")

# Same capacity reappearing later forms a NEW regime. Missing raw steps break it.
step_summary["regime_id"] = np.nan
regime=-1
previous=None
for idx,row in step_summary.iterrows():
    if row.shape_phase != "decode_candidate" or row.decode_capacity_excluded:
        previous=None
        continue
    if previous is None or previous[0] != row.input_capacity or previous[1]+1 != row.raw_step_id:
        regime+=1
    step_summary.at[idx,"regime_id"]=regime
    previous=(row.input_capacity,row.raw_step_id)
step_summary["step_kind"]="other"
step_summary.loc[step_summary.shape_phase.eq('prefill_or_mixed'),'step_kind']='prefill_or_mixed'
step_summary["decode_detection_method"]="shape_not_decode_candidate"
step_summary.loc[step_summary.decode_capacity_excluded,'decode_detection_method']='excluded_by_capacity_limit'
for key in ("kernel_structure_ok","wall_plateau_inlier","decode_plateau_start","decode_after_plateau_start"):
    step_summary[key]=False
for key in ("kernel_count_delta","wall_ratio_to_decode_tail"):
    step_summary[key]=np.nan
classified=[]
for _,group in step_summary[step_summary.regime_id.notna()].groupby("regime_id",sort=False):
    classified.append(detect_capacity_regime(group.copy().reset_index(drop=True)))
classified.append(step_summary[step_summary.regime_id.isna()])
step_summary=pd.concat(classified,ignore_index=True).sort_values("step_id").reset_index(drop=True)

# Merge step labels back to layers.
moe_layer_summary = moe_layer_summary.merge(
    step_summary[
        [
            "step_id",
            "step_kind", "regime_id", "input_capacity", "shape_signature", "shape_phase",
            "decode_detection_method",
            "moe58_wall_us",
            "next_step_interval_us",
            "kernel_count_delta",
            "wall_ratio_to_decode_tail",
            "wall_plateau_inlier",
            "decode_plateau_start",
            "decode_after_plateau_start",
        ]
    ],
    on="step_id",
    how="left",
)


# ============================================================
# 8.5. Same Layer3 anchor in consecutive forwards gives a profiler-side
#      TPOT approximation. Only pairs of complete stable Decode steps enter
#      the main summary below.
# ============================================================

step_summary["prev_step_id"] = step_summary["step_id"].shift(1)
same_regime_pair = (
    step_summary.regime_id.notna() & step_summary.regime_id.eq(step_summary.regime_id.shift(1))
    & step_summary.input_capacity.eq(step_summary.input_capacity.shift(1))
    & step_summary.raw_step_id.eq(step_summary.raw_step_id.shift(1)+1)
    & step_summary.step_kind.eq("decode") & step_summary.step_kind.shift(1).eq("decode")
)
step_summary["profiler_tpot_from_prev_us"] = step_summary.moe58_start.diff().where(same_regime_pair)


# ============================================================
# 9. One non-MLAPO timing definition for each Decode layer
# ============================================================

decode_layers = moe_layer_summary[
    moe_layer_summary["step_kind"] == "decode"
].copy()

ADDITIVE_METRICS = [
    "pre_indexer_wall_us",
    "lightning_indexer_wall_us",
    "sfa_path_wall_us",
    "output_path_wall_us",
    "moe_wall_us",
]
DIAGNOSTIC_METRICS = [
    "indexer_to_sfa_gap_us",
    "sfa_kernel_us",
    "v_up_kernel_us",
    "o_projection_wall_us",
    "o_matmul_kernel_us",
    "moe_gating_to_combine_us",
]
ALL_TIMING_METRICS = ["layer_wall_us"] + ADDITIVE_METRICS + DIAGNOSTIC_METRICS


def measure_layer(layer):
    step_id = int(layer["step_id"])
    model_layer = int(layer["model_layer"])
    layer_start, layer_end = float(layer["start"]), float(layer["end"])
    gating_start = float(layer["gating_start"])
    combine_end = float(layer["combine_end"])
    ad = data[(data["start"] >= layer_start) & (data["start"] < gating_start)]
    indexers = ad[ad["stage"] == "LightningIndexer"].sort_values("start")
    sfas = ad[ad["stage"] == "SparseFlashAttention"].sort_values("start")
    indexer = indexers.iloc[0] if len(indexers) == 1 else None
    sfa = sfas.iloc[0] if len(sfas) == 1 else None
    indexer_start = float(indexer["start"]) if indexer is not None else np.nan
    indexer_end = float(indexer["end"]) if indexer is not None else np.nan
    sfa_start = float(sfa["start"]) if sfa is not None else np.nan
    sfa_end = float(sfa["end"]) if sfa is not None else np.nan

    moe_norms = ad[
        (ad["stage"] == "AddRmsNorm")
        & (ad["start"] >= sfa_end)
        & (ad["start"] < gating_start)
    ].sort_values("start")
    moe_start = float(moe_norms.iloc[-1]["start"]) if len(moe_norms) else np.nan

    anchor_ok = bool(
        indexer is not None and sfa is not None and np.isfinite(moe_start)
        and layer_start < indexer_start < indexer_end <= sfa_start < sfa_end
        <= moe_start < gating_start < combine_end < layer_end
    )
    if not anchor_ok:
        reason = (
            f"indexers={len(indexers)},sfa={len(sfas)},"
            f"moe_norms={len(moe_norms)},anchor_order_invalid"
        )
    else:
        reason = ""

    # O's operator span is nested inside the SFA-end-to-MoE output path.
    # In W8A8 this is AscendQuantV2 + QuantBatchMatmulV3. The generic
    # Matmul fallback also supports unquantized traces.
    post_sfa = ad[(ad["start"] >= sfa_end) & (ad["start"] < moe_start)]
    v_up = post_sfa[
        post_sfa["stage"] == "BatchMatmulTranspose"
    ].sort_values("start")
    v_up_row = v_up.iloc[0] if len(v_up) else None
    # Adjacent kernels on a captured stream can overlap by a fraction of a
    # microsecond in profiler timestamps. Start after V-up *begins* so the O
    # matmul is not lost when its recorded start is just before V-up's end.
    o_search_start = float(v_up_row["start"]) if v_up_row is not None else sfa_end
    o_matmuls = post_sfa[
        (post_sfa["stage"] == "QuantMatmul")
        & (post_sfa["start"] >= o_search_start)
    ].sort_values("start")
    if o_matmuls.empty:
        o_matmuls = post_sfa[
            (post_sfa["stage"] == "Matmul")
            & (post_sfa["start"] >= o_search_start)
        ].sort_values("start")
    o_matmul = o_matmuls.iloc[0] if len(o_matmuls) else None
    o_matmul_start = float(o_matmul["start"]) if o_matmul is not None else np.nan
    o_matmul_end = float(o_matmul["end"]) if o_matmul is not None else np.nan
    o_quants = post_sfa[
        post_sfa["name"].str.contains("ascendquant", case=False, na=False)
        & (post_sfa["start"] >= o_search_start)
        & (post_sfa["start"] < o_matmul_start)
    ].sort_values("start")
    o_start = float(o_quants.iloc[-1]["start"]) if len(o_quants) else o_matmul_start

    if anchor_ok:
        additive = {
            "pre_indexer_wall_us": indexer_start - layer_start,
            "lightning_indexer_wall_us": indexer_end - indexer_start,
            "sfa_path_wall_us": sfa_end - indexer_end,
            "output_path_wall_us": moe_start - sfa_end,
            "moe_wall_us": layer_end - moe_start,
        }
        diagnostics = {
            "indexer_to_sfa_gap_us": sfa_start - indexer_end,
            "sfa_kernel_us": float(sfa["duration_us"]),
            "v_up_kernel_us": (
                float(v_up_row["duration_us"]) if v_up_row is not None else np.nan
            ),
            "o_projection_wall_us": (
                o_matmul_end - o_start if np.isfinite(o_start) else np.nan
            ),
            "o_matmul_kernel_us": (
                float(o_matmul["duration_us"]) if o_matmul is not None else np.nan
            ),
            "moe_gating_to_combine_us": combine_end - gating_start,
        }
        decomposition_error = (layer_end - layer_start) - sum(additive.values())
    else:
        additive = {name: np.nan for name in ADDITIVE_METRICS}
        diagnostics = {name: np.nan for name in DIAGNOSTIC_METRICS}
        decomposition_error = np.nan

    return {
        "step_id": step_id,
        "model_layer": model_layer,
        "stable_decode_step": layer.get("step_kind", "decode" if layer.get("wall_plateau_inlier", False) else "other") == "decode",
        "regime_id": layer.get("regime_id", np.nan),
        "input_capacity": layer.get("input_capacity", np.nan),
        "shape_signature": layer.get("shape_signature", ""),
        "anchors_ok": anchor_ok,
        "anchor_error": reason,
        "layer_end_estimated": not bool(layer["layer_end_measured"]),
        "layer_end_source": str(layer["layer_end_source"]),
        "layer_start_us": layer_start,
        "indexer_start_us": indexer_start,
        "indexer_end_us": indexer_end,
        "sfa_start_us": sfa_start,
        "sfa_end_us": sfa_end,
        "moe_start_us": moe_start,
        "gating_start_us": gating_start,
        "combine_end_us": combine_end,
        "layer_end_us": layer_end,
        "layer_wall_us": layer_end - layer_start,
        **additive,
        **diagnostics,
        "decomposition_error_us": decomposition_error,
    }


decode_layer_timing_detail = pd.DataFrame(
    [measure_layer(layer) for _, layer in decode_layers.iterrows()]
)

# Keep a step in main statistics only if all 58 MoE layers have valid anchors.
# Layer60 is included: its pre-MoE stages are observed, while layer_wall_us
# and moe_wall_us use the marked, estimated terminal boundary.
complete_step_ids = []
if not decode_layer_timing_detail.empty:
    for step_id, group in decode_layer_timing_detail.groupby("step_id"):
        ids = sorted(group["model_layer"].tolist())
        if (
            ids == list(range(FIRST_MOE_LAYER, LAST_MOE_LAYER + 1))
            and group["stable_decode_step"].all()
            and group["anchors_ok"].all()
            and group["decomposition_error_us"].abs().max() < 1e-6
        ):
            complete_step_ids.append(int(step_id))

stable_layer_timing = decode_layer_timing_detail[
    decode_layer_timing_detail["step_id"].isin(complete_step_ids)
].copy() if not decode_layer_timing_detail.empty else pd.DataFrame()

if not decode_layer_timing_detail.empty:
    bad = decode_layer_timing_detail[~decode_layer_timing_detail["anchors_ok"]]
    if len(bad):
        print(f"[WARN] {len(bad)} 个 decode 层锚点不完整；使用 --debug 查看明细。")


# ============================================================
# 9.5. Observed-order stages, matching the HTML timeline categories.
#      Stage spans are first kernel start -> last kernel end. They need not
#      add to layer_wall_us because boundaries can contain gaps or overlaps.
# ============================================================

ORIGINAL_ORDER_STAGES = [
    ("required_qkv_indexer_k", "输入归一化/QKV/Indexer K 准备", "必须在 Indexer 前"),
    ("movable_attention_kv_q", "attention KV 写入与 attention Q 准备", "可后移至 Indexer 后"),
    ("required_indexer_cache_qw", "Indexer cache 写入与 Q/W 准备", "必须在 Indexer 前"),
    ("lightning_indexer", "Lightning Indexer", "Lightning Indexer"),
    ("indexer_output_preparation", "Indexer 输出重映射与 SFA 前准备", "Attention / SFA"),
    ("sfa_q_transpose", "SFA 前 Q 转置", "Attention / SFA"),
    ("sparse_flash_attention", "SparseFlashAttention", "Attention / SFA"),
    ("v_up_projection", "V 上投影", "V / O 投影"),
    ("o_input_quant", "O 投影输入量化", "V / O 投影"),
    ("o_matmul", "O 投影 MatMul", "V / O 投影"),
    ("output_communication", "O 后通信", "通信"),
    ("moe_tail", "MoE 与层末尾", "MoE / 层末尾"),
]


def original_order_for_layer(layer):
    """Build one layer's measured stage intervals without moving any kernels."""
    layer_start, layer_end = float(layer["layer_start_us"]), float(layer["layer_end_us"])
    indexer_start, indexer_end = float(layer["indexer_start_us"]), float(layer["indexer_end_us"])
    sfa_start, sfa_end = float(layer["sfa_start_us"]), float(layer["sfa_end_us"])
    moe_start = float(layer["moe_start_us"])
    ld = data[(data["start"] >= layer_start) & (data["start"] < layer_end)].copy()
    if RING_MODE:
        return ring_support.analyze_layer(
            ld, layer_start, layer_end, True, int(layer['step_id']), int(layer['model_layer']),
            bool(layer['layer_end_estimated']),
        )[1]
    found = {}

    def set_stage(stage_id, first, last, evidence, status="measured"):
        if not (np.isfinite(first) and np.isfinite(last) and last >= first):
            return
        found[stage_id] = (float(first), float(last), str(evidence), status)

    indexer_rows = ld[ld["stage"] == "LightningIndexer"].sort_values("start")
    sfa_rows = ld[ld["stage"] == "SparseFlashAttention"].sort_values("start")
    if len(indexer_rows) != 1 or len(sfa_rows) != 1:
        return []
    indexer = indexer_rows.iloc[0]
    sfa = sfa_rows.iloc[0]
    compute_stream = indexer["stream"]
    compute = ld[ld["stream"] == compute_stream].sort_values("start").reset_index(drop=True)

    # Native preparation has its own cache and does not have to precede LI.
    # Locate the KV anchor across streams, then inspect its own stream for
    # Cast/Slice and attention Q's final InterleaveRope. Do not infer an absent
    # block's cost from the old trace or from the entire LI->SFA interval.
    pre = compute[compute["start"] < indexer_start].reset_index(drop=True)
    attention_rows = ld[ld["start"] < sfa_start].sort_values("start")
    kv_rows = attention_rows[
        attention_rows["name"].str.contains("kvrmsnormropecache", case=False, na=False)
    ]
    prep_location = "unknown"
    movable_kernel_ids = set()
    if len(kv_rows) == 1:
        kv_row = kv_rows.iloc[0]
        if kv_row["start"] < indexer_start:
            prep_location = "before_indexer"
        elif kv_row["start"] >= indexer_end - 1e-6:
            prep_location = "after_indexer"
        kv_stream = attention_rows[attention_rows["stream"] == kv_row["stream"]]
        kv_positions = np.flatnonzero(kv_stream.index == kv_row.name)
        kv_pos = int(kv_positions[0])
        cast_slice = (
            kv_pos >= 2
            and "cast" in kv_stream.iloc[kv_pos - 2]["name"].lower()
            and "slice" in kv_stream.iloc[kv_pos - 1]["name"].lower()
        )
        if cast_slice:
            movable_pos = kv_pos - 2
            if prep_location == "before_indexer" and kv_row["stream"] == compute_stream:
                scatter_positions = [
                    i for i in range(kv_pos + 1, len(kv_stream))
                    if kv_stream.iloc[i]["start"] < indexer_start
                    and "scatterndupdate" in kv_stream.iloc[i]["name"].lower()
                ]
                if scatter_positions:
                    scatter_pos = scatter_positions[0]
                    block = kv_stream.iloc[movable_pos:scatter_pos]
                    if movable_pos > 0 and not block.empty:
                        movable_kernel_ids.update(block.index)
                        set_stage(
                            "required_qkv_indexer_k", layer_start,
                            kv_stream.iloc[movable_pos - 1]["end"],
                            "layer start -> kernel before attention KV Cast/Slice",
                        )
                        set_stage(
                            "movable_attention_kv_q", block.iloc[0]["start"], block["end"].max(),
                            "Cast + Slice + KvRmsNormRopeCache -> before Indexer ScatterNdUpdate",
                        )
                        set_stage(
                            "required_indexer_cache_qw", kv_stream.iloc[scatter_pos]["start"],
                            pre["end"].max(),
                            "Indexer ScatterNdUpdate -> last compute kernel before LightningIndexer",
                        )
            elif prep_location == "after_indexer":
                # This is the rope_single endpoint in the audited native patch.
                # A different/fused endpoint must be adapted explicitly, not
                # silently treated as Indexer output preparation.
                rope_positions = [
                    i for i in range(kv_pos + 1, len(kv_stream))
                    if "interleaverope" in kv_stream.iloc[i]["name"].lower()
                ]
                if len(rope_positions) == 1:
                    block = kv_stream.iloc[movable_pos:rope_positions[0] + 1]
                    has_q_matmul = block["stage"].isin(["Matmul", "QuantMatmul"]).any()
                    # start and (start + duration) have distinct floating-point
                    # rounding after the Decimal-origin subtraction.
                    if block.iloc[0]["start"] >= indexer_end - 1e-6 and has_q_matmul:
                        movable_kernel_ids.update(block.index)
                        set_stage(
                            "movable_attention_kv_q", block.iloc[0]["start"], block["end"].max(),
                            "after LI: Cast + Slice + KvRmsNormRopeCache -> attention Q InterleaveRope",
                        )

    if prep_location == "after_indexer":
        # Native non-CP, non-quantized indexer cache path supported by the patch.
        # Multiple pre-LI scatter candidates are ambiguous; leave unresolved.
        scatters = pre.index[
            pre["name"].str.contains("scatterndupdate", case=False, na=False)
        ].tolist()
        if len(scatters) == 1 and scatters[0] > 0:
            pos = scatters[0]
            set_stage("required_qkv_indexer_k", layer_start, pre.iloc[:pos]["end"].max(),
                      "layer start -> before Indexer ScatterNdUpdate (attention KV/Q deferred)")
            set_stage("required_indexer_cache_qw", pre.iloc[pos]["start"], pre.iloc[pos:]["end"].max(),
                      "Indexer ScatterNdUpdate -> last compute kernel before LightningIndexer")

    set_stage("lightning_indexer", indexer_start, indexer_end, indexer["name"])

    # Search all streams: stream changes occur in the real baseline. Exclude
    # deferred KV/Q kernels by their row identities, not a broad time mask.
    between = ld[
        (ld["start"] >= indexer_end) & (ld["start"] < sfa_start)
        & ~ld.index.isin(movable_kernel_ids)
        & ~ld["stage"].isin([
            "CommunicationRuntime", "AllReduce", "AllGather", "ReduceScatter", "AllToAll",
        ])
    ].sort_values("start").reset_index(drop=True)
    transpose_positions = between.index[
        between["name"].str.contains("transpose", case=False, na=False)
        & ~between["name"].str.contains("batch_matmul_transpose", case=False, na=False)
    ].tolist()
    post_prep_resolved = (
        prep_location == "before_indexer"
        or (prep_location == "after_indexer" and "movable_attention_kv_q" in found)
    )
    if transpose_positions:
        transpose_pos = transpose_positions[-1]
        transpose = between.iloc[transpose_pos]
        if transpose_pos > 0 and post_prep_resolved:
            remap = between.iloc[:transpose_pos]
            # One CSV row can describe only one contiguous module span. If
            # remaining work straddles the delayed block, do not bridge across
            # it and double-count that block as remapping.
            movable = found.get("movable_attention_kv_q")
            straddles = (
                prep_location == "after_indexer" and movable is not None
                and remap.iloc[0]["start"] < movable[0]
                and remap.iloc[-1]["start"] > movable[1]
            )
            if not straddles:
                set_stage(
                    "indexer_output_preparation", remap.iloc[0]["start"], remap["end"].max(),
                    "post-LI kernels before Q Transpose, excluding identified attention KV/Q",
                )
            else:
                post_prep_resolved = False
        set_stage("sfa_q_transpose", transpose["start"], transpose["end"], transpose["name"])
    else:
        post_prep_resolved = False

    set_stage("sparse_flash_attention", sfa_start, sfa_end, sfa["name"])

    post_sfa = ld[
        (ld["start"] >= sfa_end) & (ld["start"] < moe_start)
    ].sort_values("start")
    v_rows = post_sfa[post_sfa["stage"] == "BatchMatmulTranspose"]
    if len(v_rows):
        v_row = v_rows.iloc[0]
        set_stage("v_up_projection", v_row["start"], v_row["end"], v_row["name"])
        o_search_start = float(v_row["start"])
    else:
        o_search_start = sfa_end

    o_candidates = post_sfa[
        (post_sfa["start"] >= o_search_start) & (post_sfa["stage"] == "QuantMatmul")
    ]
    if o_candidates.empty:
        o_candidates = post_sfa[
            (post_sfa["start"] >= o_search_start) & (post_sfa["stage"] == "Matmul")
        ]
    if len(o_candidates):
        o_row = o_candidates.iloc[0]
        set_stage("o_matmul", o_row["start"], o_row["end"], o_row["name"])
        q_candidates = post_sfa[
            (post_sfa["start"] >= o_search_start)
            & (post_sfa["start"] < o_row["start"])
            & post_sfa["name"].str.contains("ascendquant", case=False, na=False)
        ]
        if len(q_candidates):
            q_row = q_candidates.iloc[-1]
            set_stage("o_input_quant", q_row["start"], q_row["end"], q_row["name"])
        comm = ld[
            ld["stage"].isin(["AllReduce", "ReduceScatter", "AllGather"])
            & (ld["start"] >= o_row["start"])
            & (ld["start"] < moe_start)
        ].sort_values("start")
        if len(comm):
            set_stage(
                "output_communication", comm.iloc[0]["start"], comm["end"].max(),
                ",".join(comm["stage"].drop_duplicates()),
            )

    set_stage(
        "moe_tail", moe_start, layer_end, layer["layer_end_source"],
        "estimated_layer_end" if bool(layer["layer_end_estimated"]) else "measured",
    )

    stages = list(ORIGINAL_ORDER_STAGES)
    if prep_location == "after_indexer":
        movable_stage = stages.pop(1)
        stages.insert(3, movable_stage)  # after LightningIndexer
    result = []
    for order, (stage_id, label, category) in enumerate(stages, start=1):
        observed = found.get(stage_id)
        absent_status = (
            "not_present" if stage_id in {
                "indexer_output_preparation", "o_input_quant", "output_communication"
            } else "anchor_not_identified"
        )
        if stage_id == "indexer_output_preparation" and not post_prep_resolved:
            absent_status = "anchor_not_identified"
        if stage_id == "movable_attention_kv_q" and prep_location == "after_indexer":
            category = "已后移至 Indexer 后"
        first, last, evidence, status = observed if observed else (np.nan, np.nan, "", absent_status)
        result.append({
            "step_id": int(layer["step_id"]),
            "model_layer": int(layer["model_layer"]),
            "stage_order": order,
            "attention_prepare_location": prep_location,
            "execution_layout": {
                "before_indexer": "original", "after_indexer": "attention_deferred"
            }.get(prep_location, "unresolved") if "movable_attention_kv_q" in found else "unresolved",
            "stage_id": stage_id,
            "stage_name": label,
            "category": category,
            "status": status,
            "layer_wall_us": layer_end - layer_start,
            "layer_end_estimated": bool(layer["layer_end_estimated"]),
            "relative_start_us": first - layer_start if np.isfinite(first) else np.nan,
            "relative_end_us": last - layer_start if np.isfinite(last) else np.nan,
            "span_us": last - first if np.isfinite(first) and np.isfinite(last) else np.nan,
            "trace_start_us": first,
            "trace_end_us": last,
            "boundary_evidence": evidence,
            "boundary_delta_to_next_us": np.nan,
        })
    # Sort measured rows by actual timestamps. Missing optional rows are placed
    # immediately before the next observed stage in the expected layout.
    for i, row in enumerate(result):
        row["_sort_start"] = next(
            (r["trace_start_us"] for r in result[i:] if np.isfinite(r["trace_start_us"])), np.inf
        )
    result.sort(key=lambda row: (row["_sort_start"], row["stage_order"]))
    for order, row in enumerate(result, start=1):
        row["stage_order"] = order
        del row["_sort_start"]
    present = [i for i, row in enumerate(result) if np.isfinite(row["trace_start_us"])]
    for current, following in zip(present, present[1:]):
        result[current]["boundary_delta_to_next_us"] = (
            result[following]["trace_start_us"] - result[current]["trace_end_us"]
        )
    return result


decode_original_order_timeline = pd.DataFrame(
    record
    for _, layer in stable_layer_timing.iterrows()
    for record in original_order_for_layer(layer)
)

# Enrich the coarse, additive layer partition with an independently identified
# KV/Q module span. These diagnostic spans MUST NOT be added to that partition.
PREPARATION_METRICS = [
    "attention_prepare_wall_us", "post_indexer_attention_prepare_wall_us",
]
ALL_TIMING_METRICS += PREPARATION_METRICS
if not decode_original_order_timeline.empty:
    prep = decode_original_order_timeline[
        decode_original_order_timeline["stage_id"] == "movable_attention_kv_q"
    ][["step_id", "model_layer", "execution_layout", "attention_prepare_location",
       "status", "span_us"]].rename(columns={
           "status": "attention_prepare_status", "span_us": "attention_prepare_wall_us",
       })
    prep["post_indexer_attention_prepare_wall_us"] = np.where(
        prep["attention_prepare_location"] == "after_indexer", prep["attention_prepare_wall_us"],
        np.where(prep["attention_prepare_location"] == "before_indexer", 0.0, np.nan),
    )
    stable_layer_timing = stable_layer_timing.merge(prep, on=["step_id", "model_layer"], how="left")
    decode_layer_timing_detail = decode_layer_timing_detail.merge(
        prep, on=["step_id", "model_layer"], how="left",
    )
    unresolved = int((prep["attention_prepare_status"] != "measured").sum())
    if unresolved:
        print(f"[WARN] {unresolved} 个稳定 decode 层的 attention KV/Q 完整边界未识别；不猜测耗时。")
    if args.expect_attention_prep != "auto":
        expected = args.expect_attention_prep + "_indexer"
        mismatch = int((prep["attention_prepare_location"] != expected).sum())
        if mismatch:
            print(f"[WARN] attention KV/Q 预期位置 {expected}：{mismatch}/{len(prep)} 层不符或未识别。")

RING_METRICS = ['ring_index_remap_wall_us', 'ring_history_gather_wall_us', 'ring_kv_commit_wall_us']
if RING_MODE and not decode_original_order_timeline.empty:
    ring_records = []
    for (step_id, model_layer), stages in decode_original_order_timeline.groupby(['step_id','model_layer']):
        record = dict(step_id=step_id, model_layer=model_layer)
        for sid in ['ring_index_remap','ring_history_gather','ring_kv_commit']:
            row=stages[stages.stage_id.eq(sid)].iloc[0]
            record[sid+'_wall_us'] = (float(row.span_us) if row.status=='measured'
                                      else 0.0 if row.status=='not_present_decode' else np.nan)
        ring_records.append(record)
    ring_metrics=pd.DataFrame(ring_records)
    stable_layer_timing=stable_layer_timing.merge(ring_metrics,on=['step_id','model_layer'],validate='one_to_one')
    decode_layer_timing_detail=decode_layer_timing_detail.merge(ring_metrics,on=['step_id','model_layer'],how='left')
    ALL_TIMING_METRICS += RING_METRICS


# ============================================================
# 10. Seven user-facing CSVs; optional debug detail
# ============================================================

if not decode_original_order_timeline.empty:
    decode_original_order_timeline = decode_original_order_timeline.drop(columns=['input_capacity'], errors='ignore')
    decode_original_order_timeline = decode_original_order_timeline.merge(
        step_summary[["step_id", "regime_id", "input_capacity", "shape_signature"]],
        on="step_id", how="left", validate="many_to_one")

pair_complete = step_summary.step_id.isin(complete_step_ids) & step_summary.prev_step_id.isin(complete_step_ids)
step_summary.loc[~pair_complete,"profiler_tpot_from_prev_us"] = np.nan

forward_classification = step_summary[
    [
        "step_id", "raw_step_id", "regime_id", "input_capacity", "shape_signature", "shape_phase",
        "step_kind", "profiler_tpot_from_prev_us", "moe58_wall_us", "next_step_interval_us",
        "moe58_kernel_count", "kernel_structure_ok", "wall_plateau_inlier",
        "decode_plateau_start", "wall_ratio_to_decode_tail", "layer60_end_source",
        "decode_capacity_excluded", "decode_detection_method",
    ]
].copy().rename(columns={"next_step_interval_us": "model_step_interval_to_next_us"})
forward_classification["complete_58_layer_timing"] = (
    forward_classification["step_id"].isin(complete_step_ids)
)
forward_classification['reconstructed_58_moe_layers'] = True
forward_classification['stable_complete_decode_step'] = forward_classification['complete_58_layer_timing']


def percentile_99(values):
    return values.quantile(0.99)


if not stable_layer_timing.empty:
    aggregation = {
        "samples": ("step_id", "count"),
        "estimated_layer_end_samples": ("layer_end_estimated", "sum"),
        "attention_prepare_identified_samples": ("attention_prepare_wall_us", "count"),
        "attention_prepare_after_indexer_samples": (
            "attention_prepare_location", lambda x: (x == "after_indexer").sum()
        ),
    }
    for metric in ALL_TIMING_METRICS:
        aggregation[f"{metric}_mean"] = (metric, "mean")
        aggregation[f"{metric}_median"] = (metric, "median")
        aggregation[f"{metric}_p99"] = (metric, percentile_99)
    decode_per_layer = stable_layer_timing.groupby(
        ["regime_id", "input_capacity", "shape_signature", "model_layer"], as_index=False
    ).agg(**aggregation)

    per_step_rows = []
    for step_id, group in stable_layer_timing.groupby("step_id"):
        record = {
            "step_id": int(step_id),
            "regime_id": int(group.iloc[0]["regime_id"]),
            "input_capacity": int(group.iloc[0]["input_capacity"]),
            "shape_signature": group.iloc[0]["shape_signature"],
            "measured_or_estimated_layers": int(group["model_layer"].nunique()),
            "estimated_layer_ends": int(group["layer_end_estimated"].sum()),
            "o_projection_identified_layers": int(group["o_projection_wall_us"].notna().sum()),
            "attention_prepare_identified_layers": int(group["attention_prepare_wall_us"].notna().sum()),
            "attention_prepare_before_indexer_layers": int((group["attention_prepare_location"] == "before_indexer").sum()),
            "attention_prepare_after_indexer_layers": int((group["attention_prepare_location"] == "after_indexer").sum()),
            "attention_prepare_unknown_position_layers": int((group["attention_prepare_location"] == "unknown").sum()),
        }
        for metric in ALL_TIMING_METRICS:
            values = group[metric]
            record[f"{metric}_sum"] = (
                float(values.sum()) if values.notna().all() else np.nan
            )
        per_step_rows.append(record)
    decode_per_step = pd.DataFrame(per_step_rows).merge(
        step_summary[["step_id", "moe58_wall_us", "profiler_tpot_from_prev_us"]],
        on="step_id", how="left",
    )
    decode_per_step["layer_sum_minus_moe58_wall_us"] = (
        decode_per_step["layer_wall_us_sum"] - decode_per_step["moe58_wall_us"]
    )
    if decode_per_step["layer_sum_minus_moe58_wall_us"].abs().max() > 1e-6:
        raise RuntimeError("58层分段之和与 step wall time 不一致，请检查层边界。")

    summaries=[]
    for regime_id,rg in stable_layer_timing.groupby("regime_id",sort=False):
        group_steps=decode_per_step[decode_per_step.regime_id.eq(regime_id)]
        stable_step_set = set(rg.step_id.unique())
        stable_intervals = step_summary[
            step_summary["step_id"].isin(stable_step_set)
            & step_summary["prev_step_id"].isin(stable_step_set)
        ]["profiler_tpot_from_prev_us"].dropna().astype(float)
        summary = {
            "decode_steps_all": int(((step_summary["step_kind"] == "decode") & step_summary.regime_id.eq(regime_id)).sum()),
            "stable_complete_steps": len(stable_step_set),
            "moe_layers_per_step": NUM_MOE_LAYERS,
            "layer60_end_estimated": True,
            "attention_prepare_identified_layer_samples": int(rg["attention_prepare_wall_us"].notna().sum()),
            "attention_prepare_before_indexer_layer_samples": int((rg["attention_prepare_location"] == "before_indexer").sum()),
            "attention_prepare_after_indexer_layer_samples": int((rg["attention_prepare_location"] == "after_indexer").sum()),
            "attention_prepare_unknown_position_layer_samples": int((rg["attention_prepare_location"] == "unknown").sum()),
            "moe58_wall_us_mean": float(group_steps["moe58_wall_us"].mean()),
            "layer_wall_us_sum_mean": float(group_steps["layer_wall_us_sum"].mean()),
            "profiler_tpot_us_mean": (
                float(stable_intervals.mean()) if len(stable_intervals) else np.nan
            ),
            "profiler_tpot_us_median": (
                float(stable_intervals.median()) if len(stable_intervals) else np.nan
            ),
        }
        for metric in ALL_TIMING_METRICS:
            summary[f"{metric}_mean_per_layer"] = float(rg[metric].mean())
            summary[f"{metric}_sum_mean_per_step"] = float(
                group_steps[f"{metric}_sum"].mean()
            )
        summary.update(regime_id=int(regime_id), input_capacity=int(rg.iloc[0].input_capacity),
                       shape_signature=rg.iloc[0].shape_signature, profiler_tpot_pairs=len(stable_intervals))
        summaries.append(summary)
    decode_summary=pd.DataFrame(summaries)

else:
    decode_per_layer = pd.DataFrame()
    decode_per_step = pd.DataFrame()
    decode_summary = pd.DataFrame()
    print("[WARN] 未找到 58 层锚点完整的稳定 Decode step；仅输出 forward 分类。")

selected_layer_timeline = pd.DataFrame()
selected_layer_original_order = pd.DataFrame()
if complete_step_ids:
    step_ids = sorted(complete_step_ids)
    if TARGET_DECODE_INDEX == -1:
        target_step = step_ids[-1]
    elif 0 <= TARGET_DECODE_INDEX < len(step_ids):
        target_step = step_ids[TARGET_DECODE_INDEX]
    else:
        raise ValueError(f"TARGET_DECODE_INDEX 超出稳定 Decode 范围 0..{len(step_ids)-1}")
    if not FIRST_MOE_LAYER <= TARGET_MODEL_LAYER <= LAST_MOE_LAYER:
        raise ValueError("TARGET_MODEL_LAYER 必须位于 Layer3..60")
    target = stable_layer_timing[
        (stable_layer_timing["step_id"] == target_step)
        & (stable_layer_timing["model_layer"] == TARGET_MODEL_LAYER)
    ].iloc[0]
    layer_start, layer_end = float(target["layer_start_us"]), float(target["layer_end_us"])
    timeline = data[(data["start"] >= layer_start) & (data["start"] < layer_end)].copy()
    timeline["region"] = np.select(
        [
            timeline["start"] < target["indexer_start_us"],
            timeline["start"] < target["indexer_end_us"],
            timeline["start"] < target["sfa_end_us"],
            timeline["start"] < target["moe_start_us"],
        ],
        ["pre_indexer", "lightning_indexer", "sfa_path", "output_path"],
        default="moe",
    )
    timeline.insert(0, "step_id", target_step)
    timeline.insert(1, "model_layer", TARGET_MODEL_LAYER)
    timeline["execution_layout"] = target["execution_layout"]
    for key in ("regime_id", "input_capacity", "shape_signature"):
        timeline[key] = target[key]
    timeline["relative_start_us"] = timeline["start"] - layer_start
    timeline["relative_end_us"] = timeline["end"] - layer_start
    selected_layer_timeline = timeline[
        [
            "step_id", "regime_id", "input_capacity", "shape_signature", "model_layer", "relative_start_us", "relative_end_us",
            "duration_us", "stream", "region", "execution_layout", "stage", "name", "input_shapes",
            "source_row", "core", "block_num_raw", "mix_block_num_raw",
            "cube_cores", "vector_cores", "core_count_status",
        ]
    ].sort_values(["relative_start_us", "stream"], na_position="last")
    selected_layer_original_order = decode_original_order_timeline[
        (decode_original_order_timeline["step_id"] == target_step)
        & (decode_original_order_timeline["model_layer"] == TARGET_MODEL_LAYER)
    ].sort_values("stage_order")


def build_core_usage(kernels, layers, order):
    """Assign each stable-layer kernel once; ambiguous/uncovered modules stay unresolved.

    Ring modules use explicit source-record membership; legacy modules use
    kernel starts in measured spans. This does not measure concurrent occupancy.
    """
    keys=["regime_id","input_capacity","shape_signature","execution_layout","stage_id","stage_name"]
    details=[]
    starts=kernels.start.to_numpy()
    stage_groups={key:g for key,g in order.groupby(["step_id","model_layer"])} if not order.empty else {}
    for layer in layers.to_dict("records"):
        lo,hi=np.searchsorted(starts,[layer["layer_start_us"],layer["layer_end_us"]],side="left")
        part=kernels.iloc[lo:hi].copy()
        part["step_id"]=layer["step_id"]
        part["model_layer"]=layer["model_layer"]
        for key in keys[:4]:
            part[key]=layer[key]
        part["relative_start_us"]=part.start-layer["layer_start_us"]
        part["relative_end_us"]=part.end-layer["layer_start_us"]
        part["stage_id"]="unresolved"
        part["stage_name"]="未确定模块"
        matches=np.zeros(len(part),dtype=int)
        stages=stage_groups.get((layer["step_id"],layer["model_layer"]))
        if stages is not None:
            for stage in stages.to_dict("records"):
                first,last=stage["trace_start_us"],stage["trace_end_us"]
                if not (np.isfinite(first) and np.isfinite(last) and last>first):
                    continue
                if RING_MODE:
                    mask=part.source_row.isin(json.loads(stage.get('source_rows', '[]')))
                else:
                    mask=(part.start>=first) & (part.start<last)
                matches+=mask.to_numpy(dtype=int)
                part.loc[mask,"stage_id"]=stage["stage_id"]
                part.loc[mask,"stage_name"]=stage["stage_name"]
        part["module_assignment"]=np.where(matches==1,"source_matched" if RING_MODE else "span_matched",np.where(matches==0,"uncovered","ambiguous"))
        part.loc[matches!=1,["stage_id","stage_name"]]=["unresolved","未确定模块"]
        details.append(part)
    detail_columns=["step_id","model_layer"]+keys+[
        "source_row","relative_start_us","relative_end_us","duration_us","stream","name","input_shapes",
        "module_assignment","core","block_num_raw","mix_block_num_raw","cube_cores","vector_cores","core_count_status"]
    detail=pd.concat(details,ignore_index=True)[detail_columns] if details else pd.DataFrame(columns=detail_columns)
    profiles=[]
    for values,group in detail.groupby(keys,dropna=False,sort=False):
        profiles.append(dict(zip(keys,values)) | dict(calls=len(group),steps=group.step_id.nunique(),
            layer_samples=len(group[["step_id","model_layer"]].drop_duplicates()),**core_profile(group)))
    modules=pd.DataFrame(profiles,columns=keys+["calls","steps","layer_samples"]+list(core_profile(detail.iloc[:0])))
    op_keys=keys+["name","core","block_num_raw","mix_block_num_raw","cube_cores","vector_cores","core_count_status"]
    ops=detail.groupby(op_keys,dropna=False,sort=False).agg(calls=("source_row","size"),
        steps=("step_id","nunique")).reset_index() if not detail.empty else pd.DataFrame(columns=op_keys+["calls","steps"])
    stage_profiles=[]
    for values,group in detail.groupby(["step_id","model_layer","stage_id"],sort=False):
        stage_profiles.append(dict(zip(["step_id","model_layer","stage_id"],values)) | core_profile(group))
    stage_profiles=pd.DataFrame(stage_profiles,columns=["step_id","model_layer","stage_id"]+list(core_profile(detail.iloc[:0])))
    return detail,modules,ops,stage_profiles


core_details,core_modules,core_ops,core_stage_profiles=build_core_usage(data,stable_layer_timing,decode_original_order_timeline)
selected_core=core_details[
    core_details.step_id.eq(target_step) & core_details.model_layer.eq(TARGET_MODEL_LAYER)
].copy() if complete_step_ids else core_details.iloc[:0].copy()
if not decode_original_order_timeline.empty:
    decode_original_order_timeline=decode_original_order_timeline.merge(
        core_stage_profiles,on=["step_id","model_layer","stage_id"],how="left",validate="one_to_one")
    selected_layer_original_order=decode_original_order_timeline[
        decode_original_order_timeline.step_id.eq(target_step) & decode_original_order_timeline.model_layer.eq(TARGET_MODEL_LAYER)
    ].sort_values("stage_order")

outputs = []

def write_table(frame, filename, float_format=None):
    if frame.empty and len(frame.columns)==0:
        schemas={
            "decode_summary.csv": ["regime_id","input_capacity","shape_signature","stable_complete_steps","profiler_tpot_us_mean","profiler_tpot_pairs"],
            "decode_per_layer.csv": ["regime_id","input_capacity","model_layer","samples"],
            "decode_per_step.csv": ["step_id","regime_id","input_capacity","profiler_tpot_from_prev_us"],
        }
        frame=pd.DataFrame(columns=schemas.get(filename,["step_id","model_layer"]))
    path = os.path.join(OUTPUT_DIR, filename)
    frame.to_csv(path, index=False, float_format=float_format, encoding='utf-8-sig')
    outputs.append(path)


write_table(core_modules, "decode_module_core_usage.csv")
write_table(core_ops, "decode_kernel_core_usage.csv")
write_table(selected_core, "selected_layer_core_usage.csv")
write_table(forward_classification, "forward_classification.csv")
write_table(shape_diagnostics, "indexer_shape_diagnostics.csv")
write_table(decode_summary, "decode_summary.csv")
write_table(decode_per_layer, "decode_per_layer.csv")
write_table(decode_per_step, "decode_per_step.csv")
write_table(selected_layer_timeline, "selected_layer_timeline.csv")
write_table(decode_original_order_timeline, "decode_original_order_timeline.csv", "%.3f")
write_table(selected_layer_original_order, "selected_layer_original_order.csv", "%.3f")
if args.debug:
    write_table(decode_layer_timing_detail, "decode_layer_timing_detail.csv")

print("\n================ Decode Summary ================")
print(f"Decode steps (incl. long tail): {int((step_summary['step_kind'] == 'decode').sum())}")
print(f"Stable complete steps         : {len(complete_step_ids)}")
for _,row in decode_summary.iterrows():
    print(f"\n--- regime {int(row['regime_id'])}: capacity={int(row['input_capacity'])}, shape={row['shape_signature']} ---")
    print(f"Stable complete steps         : {int(row['stable_complete_steps'])}")
    print(f"Profiler TPOT pairs           : {int(row['profiler_tpot_pairs'])}")
    print(f"58-MoE wall mean              : {row['moe58_wall_us_mean']/1000:.3f} ms")
    if pd.notna(row["profiler_tpot_us_mean"]):
        print(f"Profiler TPOT mean            : {row['profiler_tpot_us_mean']/1000:.3f} ms")
    for metric in ["layer_wall_us"] + ADDITIVE_METRICS:
        print(f"{metric:30}: {row[f'{metric}_mean_per_layer']:.3f} us")
    print("sfa_path_wall_us = Indexer end -> SFA end; includes deferred KV/Q if present.")
    print("Preparation position (layer samples): " + ", ".join(
        f"{position}={int(row[f'attention_prepare_{position}_layer_samples'])}"
        for position in ["before_indexer", "after_indexer", "unknown_position"]
    ))
    print("Nested diagnostics (do not add to layer wall):")
    for metric in PREPARATION_METRICS + (RING_METRICS if RING_MODE else []) + ["indexer_to_sfa_gap_us", "sfa_kernel_us"]:
        print(f"  {metric:42}: {row[f'{metric}_mean_per_layer']:.3f} us")
    print("Core launch profiles by module (Cube/Vector per kernel; calls are occurrences):")
    for profile in core_modules[core_modules.regime_id.eq(row["regime_id"])].to_dict("records"):
        print(f"  {profile['stage_name']}: {core_profile_text(profile)}")
    print("Core counts are launch sizes, NOT module sums or simultaneous occupancy; N/A is not zero.")
    print("Layer60 full-layer and MoE times include an estimated terminal gap.")
if not selected_layer_original_order.empty:
    print("\n========= Selected layer: observed order =========")
    print("Layout:", selected_layer_original_order.iloc[0]["execution_layout"])
    for _, stage_row in selected_layer_original_order.iterrows():
        span = stage_row["span_us"]
        span_text = f"{span:.3f} us" if pd.notna(span) else stage_row["status"]
        print(f"{stage_row['stage_order']:2d}. {stage_row['stage_name']:<43} {span_text}")
        if pd.notna(stage_row.get("core_launch_profiles")):
            print("    "+core_profile_text(stage_row))
print("\nGenerated:")
for path in outputs:
    print(f"  {path}")
