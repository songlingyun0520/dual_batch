#!/usr/bin/env python3
"""Analyze DeepSeek V3.2 V3/V4 dual-batch kernel timelines (enable_mlapo=false).
Standalone: Python 3.10+, numpy, pandas. No NPU runtime required.
Reuses classification and robust plateau algorithm from v6_simplified; layer
boundaries group Indexers by attention-side norm, never by each Indexer.
See 双batch_profiler分析_使用说明.txt for definitions and limitations.
"""
import argparse
import csv
import json
import hashlib
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6
COMM = {"AllReduce", "AllGather", "ReduceScatter", "AllToAll"}

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
    if "allgather" in compact:
        return "AllGather"
    if "allreduce" in compact:
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
    if "batch_matmul_transpose" in n or "batchmatmultranspose" in n:
        return "BatchMatmulTranspose"
    if "swiglu" in n:
        return "SwiGLU"
    if "matmul" in n:
        return "Matmul"
    if n.strip() == "add" or "aclnnadd_" in n:
        return "Add"

    return "Other"

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

def robust_mode_int(values):
    vals = [int(v) for v in values if v is not None and not pd.isna(v) and int(v) > 0]
    if not vals:
        return np.nan
    c = Counter(vals)
    best_count = max(c.values())
    best_vals = sorted(k for k, v in c.items() if v == best_count)
    return int(best_vals[0])

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


def decimal_number(value):
    try:
        number = Decimal(str(value).replace('"', "").strip())
        return number if number.is_finite() else None
    except (InvalidOperation, ValueError, TypeError):
        return None


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


def load_kernels(path, device=None, separator="auto"):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(16384)
        handle.seek(0)
        if separator == "auto":
            sep = "\t" if sample.splitlines()[0].count("\t") > 10 else ","
        else:
            sep = "\t" if separator == "tab" else ","
        first = next(csv.reader(handle, delimiter=sep))
    headerless = len(first) >= 18 and decimal_number(first[0]) is not None and decimal_number(first[8]) is not None
    df = pd.read_csv(path, sep=sep, dtype=str, header=None if headerless else 0,
                     encoding="utf-8-sig", keep_default_na=False)
    if headerless:
        required = {0: "Device_id", 1: "Model ID", 2: "Task ID", 3: "Stream ID",
                    4: "Name", 5: "Type", 6: "OP State", 7: "Accelerator Core",
                    8: "Start Time(us)", 9: "Duration(us)", 14: "Input Shapes", 17: "Output Shapes"}
        df.columns = [required.get(i, f"extra_{i}") for i in range(len(df.columns))]
    def column(*choices):
        for choice in choices:
            normalized = re.sub(r"[\s_]", "", choice.lower())
            for c in df.columns:
                if re.sub(r"[\s_]", "", c.lower()) == normalized:
                    return c
        return None
    mapping = {
        "device": column("Device_id"), "model_id": column("Model ID"),
        "task_id": column("Task ID"), "stream": column("Stream ID"),
        "name": column("Name", "Kernel Name", "Op Name"), "op_type": column("Type"),
        "op_state": column("OP State"), "core": column("Accelerator Core"),
        "input_shapes": column("Input Shapes"), "output_shapes": column("Output Shapes"),
    }
    for field, term in (("start_raw", "start"), ("duration_raw", "duration")):
        mapping[field] = next((c for c in df.columns if term in c.lower()), None)
    for field in ("name", "stream", "start_raw", "duration_raw"):
        if mapping[field] is None:
            raise ValueError(f"Missing required column: {field}; columns={df.columns.tolist()}")
    data = pd.DataFrame({k: df[v] if v else "" for k, v in mapping.items()})
    data["source_row"] = np.arange(len(data)) + (1 if headerless else 2)
    if device is not None:
        data = data[data.device == str(device)].copy()
    if data.device.nunique() > 1:
        raise ValueError("Multiple devices: use --device ID; never merge rank timelines.")
    if data.empty:
        raise ValueError("No records for the requested device.")
    def factor(name):
        n = name.lower().replace("μ", "µ")
        return Decimal("0.001") if "(ns)" in n else Decimal(1000) if "(ms)" in n else Decimal(1)
    start_factor, duration_factor = factor(mapping["start_raw"]), factor(mapping["duration_raw"])
    starts = data.start_raw.map(decimal_number)
    durations = data.duration_raw.map(decimal_number)
    valid = starts.notna() & durations.notna()
    dropped_invalid = int((~valid).sum())
    data = data[valid].copy()
    starts, durations = starts[valid] * start_factor, durations[valid] * duration_factor
    if data.empty:
        raise ValueError("No valid timestamps/durations.")
    origin = min(starts)
    data["start"] = starts.map(lambda x: float(x - origin))
    data["duration_us"] = durations.map(float)
    if (data.duration_us < 0).any():
        raise ValueError("Negative kernel duration.")
    data["end"] = data.start + data.duration_us
    data["stage"] = data.name.map(classify)
    data["stream"] = data.stream.astype(str)
    before = len(data)
    # Do NOT dedup arbitrary different kernels just because their times match.
    data = data.drop_duplicates(subset=["device", "model_id", "task_id", "stream",
                                       "name", "start", "duration_us", "input_shapes", "output_shapes"])
    explicit = data[data.name.map(is_explicit_comm)]
    comm_keys = set(zip(explicit.stream, explicit.start, explicit.duration_us))
    shadows = [is_generic_aiv_shadow(r.name) and (r.stream, r.start, r.duration_us) in comm_keys
               for r in data.itertuples()]
    data = data.loc[~np.asarray(shadows)].sort_values(["start", "source_row"]).reset_index(drop=True)
    data["kernel_id"] = np.arange(len(data))
    if data.stage.eq("MLA_Preprocess").any():
        raise ValueError("MLA_Preprocess found: this analyzer requires enable_mlapo=false.")
    return data, {"input": str(Path(path).resolve()), "timestamp_origin_us": str(origin),
                  "headerless": headerless, "delimiter": repr(sep),
                  "invalid_rows_dropped": dropped_invalid, "duplicate_rows_dropped": before-len(data),
                  "kernels": len(data)}


def interval(data, start, end):
    times = data.start.to_numpy()
    return data.iloc[np.searchsorted(times, start, side="left"):
                     np.searchsorted(times, end, side="left")].copy()


def union_duration(rows, start, end):
    total, right = 0.0, start
    for row in rows.sort_values("start").itertuples():
        a, b = max(start, row.start), min(end, row.end)
        if b > max(right, a):
            total += b - max(right, a)
        right = max(right, b)
    return total


def reconstruct_layers(data):
    """A/B Indexers share the same preceding attention norm: one layer."""
    indexers = data[data.stage == "LightningIndexer"]
    norms = data[data.stage == "AddRmsNorm"]
    nt = norms.start.to_numpy()
    groups = []
    for kid, row in indexers.iterrows():
        pos = np.searchsorted(nt, row.start, side="left") - 1
        anchor = float(nt[pos]) if pos >= 0 else None
        # No AddRmsNorm between LI A and LI B => same layer, not another layer.
        if groups and groups[-1]["start"] == anchor:
            groups[-1]["li_ids"].append(kid)
        else:
            groups.append({"start": anchor, "li_ids": [kid]})
    layers = []
    for i, group in enumerate(groups):
        start = group["start"]
        visible_start = start if start is not None else float(data.start.min())
        upper = groups[i+1]["start"] if i+1 < len(groups) else None
        upper = float(upper) if upper is not None else float(data.end.max()) + EPS
        rows = interval(data, visible_start, upper)
        gates = rows[rows.stage == "Gating"]
        sfas = rows[rows.stage == "SparseFlashAttention"]
        combine = rows[rows.stage == "Combine"]
        li = data.loc[group["li_ids"]]
        kv = rows[rows.name.str.contains("kvrmsnormropecache", case=False)]
        q_tokens, request_rows = [], []
        for value in li.input_shapes:
            shapes = shape_list(value)
            q_tokens.append(shapes[0][0] if shapes and shapes[0] else None)
            # Audited native LI signature: Q, K-cache, W, actual_Q, actual_K, block_table.
            request_rows.append(shapes[5][0] if len(shapes)>5 and shapes[5] else None)
        schedule_ok = (len(sfas) in (1, 2) and len(li) == len(sfas)
                       and float(li.end.max()) <= float(sfas.start.min()) + EPS)
        prep_location = "unknown"
        if len(kv) == 1:
            if kv.iloc[0].start >= li.end.max() - EPS:
                prep_location = "after_indexers"
            elif kv.iloc[0].end <= li.start.min() + EPS:
                prep_location = "before_indexer"
        if schedule_ok and len(li) == 2 and prep_location == "after_indexers":
            layout = "dual_deferred"
        elif schedule_ok and len(li) == 1:
            layout = "single_deferred" if prep_location == "after_indexers" else (
                "single_original_order" if prep_location == "before_indexer" else "single_unresolved")
        else:
            layout = "partial_or_unsupported"
        shape_relation = "unknown"
        if all(x is not None for x in q_tokens+request_rows):
            shape_relation = "uniform" if q_tokens == request_rows else "nonuniform"
        if layout == "dual_deferred" and shape_relation == "uniform":
            phase = "decode_dual_schedule"
        elif layout.startswith("single") and shape_relation == "uniform":
            phase = "decode_candidate"
        elif shape_relation == "nonuniform":
            phase = "prefill_or_dp_padding"
        else:
            phase = "unknown"
        moe_norm = None
        if len(gates) == 1 and not sfas.empty:
            candidates = rows[(rows.stage == "AddRmsNorm") &
                              (rows.start >= sfas.end.max()-EPS) & (rows.start < gates.iloc[0].start)]
            if len(candidates) == 1:
                moe_norm = float(candidates.iloc[0].start)
        next_start = groups[i+1]["start"] if i+1 < len(groups) else None
        layers.append(dict(
            layer_uid=i, start=start, visible_start=visible_start, upper=upper,
            end=next_start, end_source="next_attention_start" if next_start is not None else "unresolved",
            end_estimated=False, li_ids=group["li_ids"], sfa_ids=sfas.index.tolist(),
            gating_count=len(gates), gating_start=float(gates.iloc[0].start) if len(gates)==1 else None,
            combine_end=float(combine.end.max()) if len(combine)==1 else None,
            moe_start=moe_norm, layout=layout, phase=phase, shape_relation=shape_relation,
            q_tokens=q_tokens, request_rows=request_rows, prep_location=prep_location,
            step_id=None, model_layer=None, stable_decode=False,
            body_complete=bool(schedule_ok and len(gates)==1 and len(combine)==1 and moe_norm is not None),
        ))
    return layers


def reconstruct_forwards(data, layers, first_moe=3, moe_count=58, estimate_gap=True):
    moe = [layer for layer in layers if layer["gating_count"] > 0]
    boundaries = []
    for i, layer in enumerate(moe):
        prev_uid = moe[i-1]["layer_uid"] if i else -1
        if layer["layer_uid"] - prev_uid >= first_moe + 1:
            boundaries.append(i)
    steps = []
    for ordinal, b in enumerate(boundaries):
        stop = boundaries[ordinal+1] if ordinal+1 < len(boundaries) else len(moe)
        group = moe[b:stop]
        step = dict(step_id=ordinal, moe_layer_count=len(group), complete=False, stable_decode=False,
                    phase="incomplete", layout="unknown", moe_start_us=np.nan, moe_end_us=np.nan,
                    moe_wall_us=np.nan, profiler_tpot_us=np.nan, plateau_confidence="none")
        for j, layer in enumerate(group):
            layer["step_id"], layer["model_layer"] = ordinal, first_moe+j
        dense_start = group[0]["layer_uid"]-first_moe
        if dense_start >= 0:
            for j in range(first_moe):
                layers[dense_start+j]["step_id"] = ordinal
                layers[dense_start+j]["model_layer"] = j
        if len(group) != moe_count or not all(x["body_complete"] and x["start"] is not None for x in group):
            step["reason"] = "missing_or_extra_MoE_layers_or_anchors"
            steps.append(step)
            continue
        if any(group[j+1]["layer_uid"] != group[j]["layer_uid"]+1 for j in range(len(group)-1)):
            step["reason"] = "unexpected_non_MoE_layer_inside_forward"
            steps.append(step)
            continue
        # Last MoE layer must not include LM Head / sampling / next forward gap.
        tails, gaps = [], []
        for layer in group[:-1]:
            if layer["end"] is None:
                continue
            post = interval(data, layer["combine_end"], layer["end"])
            terminal = post[post.stage == "Add"]
            if not terminal.empty:
                t = float(terminal.end.max())
                gaps.append(max(0.0, layer["end"]-t))
                tails.append(t-layer["combine_end"])
        last = group[-1]
        post = interval(data, last["combine_end"], last["upper"])
        terminal = post[post.stage == "Add"]
        if terminal.empty or not tails:
            last["end"], last["end_source"] = None, "unresolved_terminal"
            step["reason"] = "missing_terminal_Add; no fabricated_last_layer_duration"
            steps.append(step)
            continue
        ref = float(np.median(tails))
        target = terminal.iloc[np.argmin(np.abs(terminal.end.to_numpy()-last["combine_end"]-ref))]
        terminal_end = float(target.end)
        gap = float(np.median(gaps)) if estimate_gap else 0.0
        last["end"] = min(terminal_end+gap, last["upper"]-EPS)
        last["end_estimated"] = bool(estimate_gap)
        last["end_source"] = "terminal_Add+median_gap" if estimate_gap else "terminal_Add_end"
        start, end = group[0]["start"], last["end"]
        rows = interval(data, start, end)
        phases = {x["phase"] for x in group}
        if phases == {"decode_dual_schedule"}:
            phase = "decode_dual_schedule"
        elif phases <= {"decode_candidate", "decode_dual_schedule"}:
            phase = "decode_candidate"
        elif "prefill_or_dp_padding" in phases:
            phase = "prefill_or_dp_padding"
        else:
            phase = "unknown"
        layouts = {x["layout"] for x in group}
        layout = next(iter(layouts)) if len(layouts)==1 else "mixed_layout"
        counts = rows[rows.stage != "CommunicationRuntime"].stage.value_counts()
        signature = sorted({tuple(x["q_tokens"]) for x in group}, key=str)
        step.update(complete=True, phase=phase, layout=layout, moe_start_us=start,
                    moe_end_us=end, moe_wall_us=end-start, kernel_count=int(counts.sum()),
                    shape_signature=str(signature), last_layer_end_estimated=last["end_estimated"],
                    last_layer_end_source=last["end_source"], terminal_gap_us=gap,
                    reason="complete_MoE_structure")
        for name in ("Gating","Dispatch","Expert_GEMM1","Expert_GEMM2","Combine"):
            step[name+"_count"] = int(counts.get(name,0))
        steps.append(step)
    return steps


def classify_stable_steps(steps, layers, min_steps=3):
    """Separate layout/shape regimes; retain long tails AFTER plateau onset."""
    runs = []
    for i, step in enumerate(steps):
        eligible = step["complete"] and step["phase"] in {"decode_candidate","decode_dual_schedule"}
        sig = (step.get("layout"), step.get("shape_signature")) if eligible else None
        if eligible:
            if runs and runs[-1]["last"] == i-1 and runs[-1]["sig"] == sig:
                runs[-1]["indices"].append(i)
                runs[-1]["last"] = i
            else:
                runs.append(dict(sig=sig, indices=[i], last=i))
    for regime, run in enumerate(runs):
        indices = run["indices"]
        records = [steps[i] for i in indices]
        ref_count = robust_mode_int([s["kernel_count"] for s in records[-8:]])
        tol = max(4, int(np.ceil(ref_count*0.001)))
        critical = ["Gating_count","Dispatch_count","Expert_GEMM1_count","Expert_GEMM2_count","Combine_count"]
        refs = {c: robust_mode_int([s[c] for s in records[-8:]]) for c in critical}
        structure = [abs(s["kernel_count"]-ref_count) <= tol and
                     all(s[c]==refs[c] for c in critical) for s in records]
        plateau = find_decode_plateau_start([s["moe_wall_us"] for s in records], structure,
                                           min_decode_steps=min_steps)
        for pos, step in enumerate(records):
            step["regime_id"] = regime
            step["kernel_structure_ok"] = structure[pos]
            step["plateau_ref_wall_us"] = plateau["ref_wall_us"]
            step["wall_plateau_inlier"] = bool(plateau["inlier_mask"][pos])
            step["plateau_confidence"] = plateau["confidence"]
            step["stable_decode"] = (plateau["start_idx"] is not None and
                                     pos >= plateau["start_idx"] and structure[pos])
    for i, step in enumerate(steps):
        if (i and step["stable_decode"] and steps[i-1]["stable_decode"]
                and step.get("regime_id")==steps[i-1].get("regime_id")):
            # Only adjacent observed complete forwards; never bridge a failed group.
            step["profiler_tpot_us"] = step["moe_start_us"]-steps[i-1]["moe_start_us"]
    by_id = {s["step_id"]:s for s in steps}
    for layer in layers:
        step = by_id.get(layer["step_id"], {})
        layer["stable_decode"] = step.get("stable_decode",False)
        layer["regime_id"] = step.get("regime_id")


LABELS = {
    "input_norm_metadata": "公共输入归一化 / 子组 metadata 准备",
    "prep_A_qkv_k": "A：QKV / q_c 归一化 / Indexer K 准备",
    "prep_A_cache_qw": "A：Indexer cache 写入与 Q/W 准备",
    "prep_B_qkv_k": "B：QKV / q_c 归一化 / Indexer K 准备",
    "prep_B_cache_qw": "B：Indexer cache 写入与 Q/W 准备",
    "prep_A_partial": "A：可见准备片段（缺失层首）",
    "prep_single_qkv_k": "完整 batch：QKV / q_c / Indexer K 准备",
    "prep_single_cache_qw": "完整 batch：Indexer cache 写入与 Q/W 准备",
    "prep_single_partial": "完整 batch：可见准备片段",
    "indexer_A": "Lightning Indexer A", "indexer_B": "Lightning Indexer B",
    "indexer_single": "Lightning Indexer（完整 batch）",
    "merge_preparation": "合并 A/B 的 q_c 与 KV 投影",
    "attention_kv_q": "完整 batch：attention KV 写入与 Q 准备",
    "sfa_q_layout": "SFA 前 Q 连续化 / 转置",
    "sfa_A": "SparseFlashAttention A", "sfa_B": "SparseFlashAttention B",
    "sfa_single": "SparseFlashAttention（完整 batch）",
    "merge_sfa_output": "合并 A/B SFA 输出",
    "v_up_projection": "V 上投影", "o_input_quant": "O 投影输入量化",
    "o_matmul": "O 投影 MatMul", "output_communication": "O 后通信",
    "moe": "MoE 与层末尾", "unresolved": "未归属算子",
}


def assign_modules(data, layer):
    rows = interval(data, layer["visible_start"], layer["upper"])
    # Exclude background collective runtime envelopes from module timing.
    rows["module"] = "unresolved"
    rows.loc[rows.stage == "CommunicationRuntime","module"] = "communication_runtime"
    li = rows[rows.stage=="LightningIndexer"]
    sfa = rows[rows.stage=="SparseFlashAttention"]
    is_dual = len(li)==2  # descriptive even if the fragment has no complete SFA
    stream = str(li.iloc[0].stream)
    compute = rows[(rows.stream == stream) & (rows.stage != "CommunicationRuntime")]
    def assign(ids, module):
        ids = rows.index.intersection(ids)
        ids = ids[rows.loc[ids,"module"] != "communication_runtime"]
        rows.loc[ids,"module"] = module
    previous_end = layer["visible_start"]
    for j,(idx,indexer) in enumerate(li.iterrows()):
        tag = (("A" if j==0 else "B") if is_dual else "single") if (
            layer["layout"]!="partial_or_unsupported") else f"observed_{j+1}"
        pre = compute[(compute.start>=previous_end-EPS) & (compute.start<indexer.start)]
        scatters = pre[pre.name.str.contains("scatterndupdate",case=False)]
        partial = j==0 and layer["start"] is None
        if j==0 and layer["start"] is not None:
            # First quantization before QKV starts group A; norm/Sub are shared.
            quant = pre[pre.name.str.contains("quant",case=False) & ~pre.stage.isin(["QuantMatmul"])]
            if not quant.empty:
                q_start = float(quant.iloc[0].start)
                assign(pre[pre.start<q_start].index, "input_norm_metadata")
                pre = pre[pre.start>=q_start]
        if partial:
            assign(pre.index, f"prep_{tag}_partial")
        elif len(scatters)==1:
            split=float(scatters.iloc[0].start)
            assign(pre[pre.start<split].index, f"prep_{tag}_qkv_k")
            assign(pre[pre.start>=split].index, f"prep_{tag}_cache_qw")
        else:
            assign(pre.index, "unresolved")
        assign([idx], f"indexer_{tag}")
        previous_end=float(indexer.end)
    # Shared preparation block: include immediate Cast/Slice input materialization,
    # but keep aclnnCat slices with the merge they belong to.
    before_sfa = compute[compute.start < sfa.start.min()] if not sfa.empty else compute.iloc[0:0]
    kv = before_sfa[before_sfa.name.str.contains("kvrmsnormropecache",case=False)]
    if len(kv)==1:
        kv_idx=kv.index[0]
        rope=before_sfa[(before_sfa.start>kv.iloc[0].start) &
                        before_sfa.name.str.contains("interleaverope",case=False)]
        if len(rope)==1:
            ids=before_sfa.index.tolist()
            pos=ids.index(kv_idx)
            while pos>0:
                n=before_sfa.loc[ids[pos-1],"name"].lower()
                if ("cast" in n or "slice" in n) and "aclnncat" not in n:
                    pos-=1
                else:
                    break
            start=float(before_sfa.loc[ids[pos],"start"])
            end=float(rope.iloc[0].end)
            block=compute[(compute.start>=start) & (compute.start<end)]
            if block.stage.isin(["Matmul","QuantMatmul"]).any():
                assign(block.index,"attention_kv_q")
                between=compute[(compute.start>=li.end.max()-EPS) & (compute.start<start)]
                assign(between[between.name.str.contains("aclnncat|concat",case=False)].index,"merge_preparation")
                layout=compute[(compute.start>=end-EPS) & (compute.start<sfa.start.min())]
                assign(layout[layout.name.str.contains("transpose|contiguous|asstrided",case=False)].index,"sfa_q_layout")
    for j,(idx,_) in enumerate(sfa.iterrows()):
        tag=("A" if j==0 else "B") if len(sfa)==2 else "single"
        assign([idx],f"sfa_{tag}")
    if not sfa.empty:
        post = rows[(rows.start>=sfa.end.max()-EPS) &
                    (rows.start<(layer["moe_start"] if layer["moe_start"] is not None else layer["upper"]))]
        v=post[post.stage=="BatchMatmulTranspose"]
        if len(v)==1:
            assign(v.index,"v_up_projection")
            merged=post[(post.start<v.iloc[0].start) & post.name.str.contains("aclnncat|concat",case=False)]
            assign(merged.index,"merge_sfa_output")
            o=post[(post.start>=v.iloc[0].end-EPS) & post.stage.isin(["QuantMatmul","Matmul"])]
            if len(o)==1:
                assign(o.index,"o_matmul")
                quant=post[(post.start>=v.iloc[0].end-EPS) & (post.start<o.iloc[0].start)
                           & post.name.str.contains("quant",case=False)]
                assign(quant.index,"o_input_quant")
                comm=post[(post.start>=o.iloc[0].end-EPS) & post.stage.isin(COMM)]
                assign(comm.index,"output_communication")
    if layer["moe_start"] is not None:
        end=layer["end"] if layer["end"] is not None else layer["upper"]
        assign(rows[(rows.start>=layer["moe_start"]) & (rows.start<end)].index,"moe")
    if layer["end"] is not None:
        rows=rows[rows.start<layer["end"]].copy()
    rows["layer_uid"]=layer["layer_uid"]
    rows["step_id"]=layer["step_id"]
    rows["model_layer"]=layer["model_layer"]
    rows["layout"]=layer["layout"]
    rows["stable_decode"]=layer["stable_decode"]
    rows["relative_start_us"]=rows.start-layer["visible_start"]
    rows["relative_end_us"]=rows.end-layer["visible_start"]
    rows["relative_origin"]="layer_start" if layer["start"] is not None else "fragment_start"
    return rows


def module_timings(rows, layer):
    result=[]
    measured=rows[rows.module!="communication_runtime"]
    if measured.empty:
        return result
    # Kernel spans are diagnostic; an independent start-to-start partition
    # includes launch gaps and must add up to the layer wall.
    for module,group in measured.groupby("module",sort=False):
        start,end=float(group.start.min()),float(group.end.max())
        coverage=union_duration(group,start,end)
        result.append(dict(layer_uid=layer["layer_uid"],step_id=layer["step_id"],
                           model_layer=layer["model_layer"],layout=layer["layout"],
                           stable_decode=layer["stable_decode"],regime_id=layer.get("regime_id"),
                           layer_start_missing=layer["start"] is None,
                           layer_end_estimated=layer["end_estimated"],
                           module=module,module_name=LABELS.get(module,module),
                           start_us=start,end_us=end,span_us=end-start,
                           kernel_sum_us=float(group.duration_us.sum()),kernel_union_us=coverage,
                           span_uncovered_us=max(0.0,end-start-coverage),kernel_count=len(group),
                           source_rows=";".join(map(str,group.source_row)),
                           kernel_names=" | ".join(group.name.drop_duplicates()),
                           status="partial" if "partial" in module or "observed_" in module or
                                  (module=="moe" and layer["end"] is None) else
                                  "unresolved" if module=="unresolved" else "measured",
                           relative_origin=rows.relative_origin.iloc[0],
                           relative_start_us=start-layer["visible_start"],
                           relative_end_us=end-layer["visible_start"]))
    return sorted(result,key=lambda r:r["start_us"])


def wall_partition(rows, layer):
    clean=rows[rows.module!="communication_runtime"].sort_values(["start","source_row"])
    if clean.empty:
        return []
    # Consecutive module runs, not min/max envelopes: no double-counting even
    # when an unresolved block appears between two assigned blocks.
    runs=[]
    for r in clean.itertuples():
        if not runs or runs[-1]["module"]!=r.module:
            runs.append(dict(module=r.module,start_us=float(r.start),last_end=float(r.end)))
        else:
            runs[-1]["last_end"]=max(runs[-1]["last_end"],float(r.end))
    out=[]
    for i,run in enumerate(runs):
        end=runs[i+1]["start_us"] if i+1<len(runs) else layer["end"]
        if end is None:
            end=run["last_end"]
        out.append(dict(layer_uid=layer["layer_uid"],step_id=layer["step_id"],
                        model_layer=layer["model_layer"],layout=layer["layout"],
                        stable_decode=layer["stable_decode"],segment=i+1,module=run["module"],
                        start_us=run["start_us"],end_us=end,wall_us=end-run["start_us"],
                        layer_start_missing=layer["start"] is None,
                        layer_end_source=layer["end_source"],end_estimated=layer["end_estimated"]))
    if layer["start"] is not None and layer["end"] is not None:
        total=sum(x["wall_us"] for x in out)
        if abs(total-(layer["end"]-layer["start"])) > EPS*10:
            raise ValueError(f"Layer {layer['layer_uid']} wall partition mismatch.")
    return out


def layer_record(layer, rows, modules):
    start, end = layer["start"], layer["end"]
    record = {k:v for k,v in layer.items() if k not in {"li_ids","sfa_ids"}}
    record["q_tokens"] = json.dumps(layer["q_tokens"])
    record["request_rows"] = json.dumps(layer["request_rows"])
    record["layer_wall_us"] = end-start if start is not None and end is not None else np.nan
    record["layer_complete"] = bool(start is not None and end is not None and layer["body_complete"])
    record["layer_end_estimated"] = layer["end_estimated"]
    record["moe_wall_us"] = end-layer["moe_start"] if end is not None and layer["moe_start"] is not None else np.nan
    for name in LABELS:
        item = next((m for m in modules if m["module"]==name), None)
        for metric in ("span_us","kernel_sum_us","kernel_union_us"):
            record[f"{name}_{metric}"] = item[metric] if item else np.nan
    record["layer_kernel_sum_us"] = float(rows.loc[rows.module!="communication_runtime","duration_us"].sum())
    li = rows[rows.stage=="LightningIndexer"]
    sfa = rows[rows.stage=="SparseFlashAttention"]
    for j,tag in enumerate(("A","B")):
        record[f"indexer_{tag}_end_to_sfa_{tag}_start_us"] = (
            float(sfa.iloc[j].start-li.iloc[j].end) if len(li)==len(sfa)==2 else np.nan)
    # These are dependency windows in the observed compute schedule, NOT a
    # measured amount of KV transfer overlap or guaranteed hidden latency.
    return record


def analyze(path, output_dir, device=None, sep="auto", first_moe=3, moe_count=58,
            min_steps=3, estimate_gap=True, target_layer=3, target_step=-1, debug=False):
    data, manifest = load_kernels(path, device, sep)
    shape_diagnostics = []
    for r in data[data.stage == "LightningIndexer"].itertuples():
        shapes = shape_list(r.input_shapes)
        q = shapes[0][0] if shapes and shapes[0] else None
        req = shapes[5][0] if len(shapes) > 5 and shapes[5] else None
        shape_diagnostics.append(dict(source_row=r.source_row, kernel_id=r.kernel_id,
                                      input_shapes_repr=repr(r.input_shapes),
                                      parsed_shapes=json.dumps(shapes), q_tokens=q,
                                      request_rows=req, parse_ok=q is not None and req is not None))
    shape_diagnostics = pd.DataFrame(shape_diagnostics, columns=[
        "source_row", "kernel_id", "input_shapes_repr", "parsed_shapes",
        "q_tokens", "request_rows", "parse_ok"])
    layers = reconstruct_layers(data)
    if not layers:
        raise ValueError("No LightningIndexer found.")
    steps = reconstruct_forwards(data,layers,first_moe,moe_count,estimate_gap)
    classify_stable_steps(steps,layers,min_steps)
    details, all_modules, partitions, kernels = [], [], [], []
    for layer in layers:
        rows=assign_modules(data,layer)
        module_rows=module_timings(rows,layer)
        details.append(layer_record(layer,rows,module_rows))
        all_modules.extend(module_rows)
        partitions.extend(wall_partition(rows,layer))
        kernels.append(rows)
    detail=pd.DataFrame(details)
    module_df=pd.DataFrame(all_modules)
    partition_df=pd.DataFrame(partitions)
    kernel_df=pd.concat(kernels,ignore_index=True)
    forward=pd.DataFrame(steps) if steps else pd.DataFrame(columns=[
        "step_id","complete","phase","layout","stable_decode","moe_wall_us","profiler_tpot_us"])
    stable=detail[detail.stable_decode & detail.layer_complete &
                  (detail.model_layer>=first_moe) & (detail.model_layer<first_moe+moe_count)].copy()
    stable_steps=forward[forward.stable_decode].copy()
    per_layer=pd.DataFrame(columns=["regime_id","layout","model_layer","samples"])
    metrics=[c for c in detail.columns if c.endswith("_us") and c not in {"start_us","end_us"}]
    if not stable.empty:
        aggregation={"samples":("layer_uid","count"),
                     "estimated_layer_end_samples":("layer_end_estimated","sum")}
        for metric in metrics:
            for stat in ("mean","median"):
                aggregation[metric+"_"+stat]=(metric,stat)
            aggregation[metric+"_p99"]=(metric,lambda x:x.quantile(.99))
        per_layer=stable.groupby(["regime_id","layout","model_layer"],as_index=False).agg(**aggregation)
    step_rows=[]
    for step in steps:
        if not step["stable_decode"]:
            continue
        group=stable[stable.step_id==step["step_id"]]
        if len(group)!=moe_count:
            continue
        total=float(group.layer_wall_us.sum())
        error=total-step["moe_wall_us"]
        if abs(error)>1e-5:
            raise ValueError("MoE layer wall sum does not match forward span.")
        step_rows.append(dict(step_id=step["step_id"],regime_id=step.get("regime_id"),
                              layout=step["layout"],shape_signature=step["shape_signature"],
                              layers=len(group),estimated_layer_ends=int(group.layer_end_estimated.sum()),
                              moe_layers_wall_us=total,profiler_tpot_us=step["profiler_tpot_us"],
                              layer_sum_error_us=error))
    per_step=pd.DataFrame(step_rows,columns=["step_id","regime_id","layout","shape_signature","layers",
                                           "estimated_layer_ends","moe_layers_wall_us","profiler_tpot_us",
                                           "layer_sum_error_us"])
    summaries=[]
    for regime,g in per_step.groupby("regime_id"):
        summary=dict(regime_id=regime,layout=g.layout.iloc[0],shape_signature=g.shape_signature.iloc[0],
                     stable_steps=len(g),moe_layers=moe_count,
                     moe_wall_us_mean=g.moe_layers_wall_us.mean(),
                     profiler_tpot_us_mean=g.profiler_tpot_us.mean(),
                     profiler_tpot_pairs=int(g.profiler_tpot_us.notna().sum()),
                     estimated_layer_ends=int(g.estimated_layer_ends.sum()))
        sg=stable[stable.regime_id==regime]
        for metric in metrics:
            summary[metric+"_mean"]=sg[metric].mean()
        summaries.append(summary)
    summary_df=pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=[
        "regime_id","layout","stable_steps","moe_wall_us_mean","profiler_tpot_us_mean"])
    # Choose an actual stable sample; for excerpts choose the best visible body.
    target=None
    ids=stable.step_id.drop_duplicates().sort_values().tolist()
    if ids:
        if target_step < -len(ids) or target_step >= len(ids):
            raise ValueError("--target-step is an index into stable observed steps.")
        match=stable[(stable.step_id==ids[target_step]) & (stable.model_layer==target_layer)]
        if match.empty:
            raise ValueError("--target-layer outside reconstructed MoE layers.")
        target=int(match.iloc[0].layer_uid)
    else:
        candidates=[x for x in layers if x["body_complete"]]
        target=(candidates[0] if candidates else layers[0])["layer_uid"]
    selected_kernels=kernel_df[kernel_df.layer_uid==target]
    selected_modules=module_df[module_df.layer_uid==target].copy()
    selected_modules.insert(0,"stage_order",np.arange(1,len(selected_modules)+1))
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    tables={
        "indexer_shape_diagnostics.csv":shape_diagnostics,
        "forward_classification.csv":forward,
        "all_layer_timing.csv":detail,
        "decode_summary.csv":summary_df,
        "decode_per_layer.csv":per_layer,
        "decode_per_step.csv":per_step,
        "all_module_timing.csv":module_df,
        "decode_original_order_timeline.csv":module_df[
            module_df.stable_decode & module_df.model_layer.ge(first_moe) &
            module_df.model_layer.lt(first_moe+moe_count)],
        "selected_layer_timeline.csv":selected_kernels,
        "selected_layer_original_order.csv":selected_modules,
        "layer_wall_partition.csv":partition_df,
    }
    if debug:
        tables["all_kernel_assignments.csv"]=kernel_df
    for name,frame in tables.items():
        frame.to_csv(out/name,index=False,float_format="%.6f",encoding="utf-8-sig")
    manifest.update(
        analyzer="dualbatch_v4.1", script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        indexer_shape_parse_failures=int((~shape_diagnostics.parse_ok).sum()),
        layers_detected=len(layers),
        forward_groups_detected=len(steps),complete_forwards=int(forward.complete.sum()),
        stable_steps=int(forward.stable_decode.sum()),selected_layer_uid=target,
        full_rank_trace_required_for_tpot=True,
        terminal_gap_policy="estimate_from_regular_layers" if estimate_gap else "terminal_kernel_end",
        limits=[
            "Single Indexer/SFA cannot uniquely prove V4 fallback versus original/prefill execution.",
            "Nonuniform token/request shapes => prefill_or_dp_padding, not definitive prefill.",
            "Static OP State is not proof of graph replay; no token generation ordinal is inferred.",
            "Layer numbers require a complete observed 58-MoE forward; excerpts keep IDs blank.",
            "Layer60 uses observed terminal Add; optional median terminal gap is explicitly estimated.",
            "kernel_sum may exceed wall span under concurrency; interval union avoids double-counting.",
            "span_uncovered_us is uncovered time, not a causal measurement of waiting.",
            "Wall partition attributes each inter-module gap to its preceding segment.",
        ])
    (out/"analysis_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Cleaned kernels: {len(data)}; layer groups: {len(layers)}")
    print(f"Complete MoE forwards: {int(forward.complete.sum())}; stable decode steps: {int(forward.stable_decode.sum())}")
    if not len(stable_steps):
        print("[WARN] No stable complete decode: selected timings are observational, not a stable sample.")
        print("Complete forward phases:", forward.loc[forward.complete, "phase"].value_counts().to_dict())
    if not shape_diagnostics.parse_ok.all():
        print("[WARN] Indexer shape parsing failed; inspect indexer_shape_diagnostics.csv.")
    print(f"Selected layer_uid={target}, origin={selected_kernels.relative_origin.iloc[0]}")
    for r in selected_modules.itertuples():
        print(f"{r.module_name:36s} span={r.span_us:10.3f} us; kernel_sum={r.kernel_sum_us:10.3f} us [{r.status}]")
    print(f"Outputs: {out.resolve()}")
    return dict(data=data,layers=layers,steps=forward,detail=detail,modules=module_df,
                partitions=partition_df,kernels=kernel_df,summary=summary_df)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv",default="kernel_details.csv")
    parser.add_argument("--output-dir",default="deepseek_dualbatch_analysis")
    parser.add_argument("--device",help="Required when input contains multiple Device_id values")
    parser.add_argument("--sep",choices=["auto","comma","tab"],default="auto")
    parser.add_argument("--first-moe-layer",type=int,default=3)
    parser.add_argument("--moe-layers",type=int,default=58)
    parser.add_argument("--min-decode-steps",type=int,default=3)
    parser.add_argument("--terminal-gap",choices=["estimate","none"],default="estimate")
    parser.add_argument("--target-layer",type=int,default=3)
    parser.add_argument("--target-step",type=int,default=-1,help="Index into stable steps; -1 selects the last")
    parser.add_argument("--debug",action="store_true")
    args=parser.parse_args()
    if args.min_decode_steps<3 or args.first_moe_layer<1 or args.moe_layers<2:
        parser.error("min-decode-steps >= 3, first-moe-layer >= 1, moe-layers >= 2 required")
    analyze(args.csv,args.output_dir,args.device,args.sep,args.first_moe_layer,args.moe_layers,
            args.min_decode_steps,args.terminal_gap=="estimate",args.target_layer,args.target_step,args.debug)


if __name__=="__main__":
    main()
