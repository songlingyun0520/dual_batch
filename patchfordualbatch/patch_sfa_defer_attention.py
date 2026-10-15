#!/usr/bin/env python3
"""Opt-in decode-only SFA preparation reorder for two audited source snapshots.

Default: print a diff. --apply: back up and update the installed Python source.
No torch/NPU imports are required. An unknown source is NEVER modified.
"""
import argparse
import difflib
import hashlib
import importlib.util
from pathlib import Path


KNOWN = {
    "63849e10e82f8bee4a9f0fb21ab0d606852cd6cf86a0a0fb983c8187980189a7": "profile_snapshot",
    "9f2a521f6eeca5b54f47de54129c6ce2c4ddf5d450e8ab10f111b86d0ba939d8": "local_v023_snapshot",
}
MARKER = "# SFA_DEFER_ATTN_PREP_PATCH_V1"


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one source anchor: {before[:100]!r}")
    return text.replace(before, after, 1)


def transform(source):
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest not in KNOWN:
        raise ValueError(f"Unsupported source SHA256={digest}; source left unchanged.")
    variant = KNOWN[digest]
    source = (
        MARKER + "\nimport os as _sfa_reorder_os\n"
        '_SFA_DEFER_ATTN_PREP = _sfa_reorder_os.environ.get("VLLM_ASCEND_SFA_DEFER_ATTN_PREP", "0") == "1"\n\n'
        + source
    )
    # This is a Python metadata/config decision, never a Tensor-value decision.
    guard = '''        defer_attention_prepare = (
            _SFA_DEFER_ATTN_PREP
            and attn_metadata.attn_state == AscendAttentionState.DecodeOnly
        )
        if defer_attention_prepare:
            unsupported = []
            if self.enable_mlapo:
                unsupported.append("enable_mlapo")
            if self.enable_dsa_cp:
                unsupported.append("DSA-CP")
            if getattr(attn_metadata, "dcp_context", None) is not None:
                unsupported.append("DCP")
            if self.vllm_config.kv_transfer_config is not None or self.is_kv_producer:
                unsupported.append("KV connector/producer")
            if getattr(self, "enable_sfa_prolog_v3", False):
                unsupported.append("SFA prolog v3")
            if not getattr(self, "has_indexer", True):
                unsupported.append("missing indexer")
            if getattr(self, "skip_topk", False) or getattr(self, "use_index_cache", False):
                unsupported.append("IndexCache/skip_topk")
            if any(getattr(self, name, False) for name in (
                "enable_sparse_sfa_c8", "enable_sparse_li_c8", "use_sparse_c8_indexer"
            )):
                unsupported.append("quantized sparse KV/indexer cache")
            if _sfa_reorder_os.environ.get("VLLM_ASCEND_SFA_DUAL_BATCH") == "1":
                unsupported.append("SFA dual batch patch")
            if unsupported:
                raise RuntimeError("SFA attention-prep reorder does not support: " + ", ".join(unsupported))

'''
    # Anchor in forward, not in any metadata builder/helper.
    source = replace_once(source, "        cos = attn_metadata.cos\n        sin = attn_metadata.sin\n", guard + "        cos = attn_metadata.cos\n        sin = attn_metadata.sin\n")
    if variant == "profile_snapshot":
        source = replace_once(source,
            "            else:\n                k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)\n",
            "            elif not defer_attention_prepare:\n                k_pe, k_nope = self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)\n")
        q_block = "            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)\n            q_pe = self.rope_single(q_pe, cos, sin)\n"
        delayed_kv = "            self.exec_kv(kv_no_split, cos, sin, kv_cache, slot_mapping, attn_metadata)\n"
        delayed_context = ""
    else:
        # Retain slot_mapping_sfa from this version (do not substitute slot_mapping).
        start = source.index("            kv_outputs = self.exec_kv(kv_no_split, cos, sin, kv_cache, kv_slots, attn_metadata)\n")
        end = source.index("            if self.enable_dsa_cp:\n", start)
        kv_block = source[start:end]
        source = replace_once(source, kv_block,
            "            if not defer_attention_prepare:\n" +
            "".join("    " + line if line.strip() else line for line in kv_block.splitlines(True)))
        q_block = "            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)\n            q_pe = self.rope_single(q_pe, cos, sin)\n            self._record_dcp_query_gather_context(ql_nope, q_pe, attn_metadata)\n"
        source = replace_once(source,
            "            if kv_cache is not None:\n                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)\n",
            "            if kv_cache is not None and not defer_attention_prepare:\n                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)\n")
        delayed_kv = "            self.exec_kv(kv_no_split, cos, sin, kv_cache, kv_slots, attn_metadata)\n"
        delayed_context = (
            "            self._record_dcp_query_gather_context(ql_nope, q_pe, attn_metadata)\n"
            "            if kv_cache is not None:\n"
            "                self._record_dcp_kv_gather_context(kv_cache, attn_metadata)\n"
        )
    source = replace_once(source, q_block,
        "            if not defer_attention_prepare:\n" +
        "".join("    " + line for line in q_block.splitlines(True)))
    delayed = (
        "        # Indexer has consumed its independent cache. Prepare SFA KV/Q now.\n"
        "        if defer_attention_prepare:\n" + delayed_kv +
        "            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)\n"
        "            q_pe = self.rope_single(q_pe, cos, sin)\n" + delayed_context + "\n"
    )
    anchor = "        attn_output = self._execute_sparse_flash_attention_process(\n"
    source = replace_once(source, anchor, delayed + anchor)
    compile(source, "sfa_v1.py", "exec")
    return source


def installed_source():
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise ValueError("Cannot locate vllm_ascend; use --source /actual/path/sfa_v1.py")
    return Path(next(iter(spec.submodule_search_locations))) / "attention" / "sfa_v1.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = (args.source or installed_source()).resolve()
    original = path.read_bytes().decode("utf-8")
    if MARKER in original:
        print(f"Already patched: {path}; no change.")
        return
    try:
        patched = transform(original)
    except ValueError as exc:
        raise SystemExit(str(exc))
    print(f"Source: {path}")
    print("".join(difflib.unified_diff(original.splitlines(True), patched.splitlines(True),
                                    fromfile="sfa_v1.py.before", tofile="sfa_v1.py.after")))
    if not args.apply:
        print("Dry run only. Use --apply to write the patch.")
        return
    backup = path.with_name(path.name + ".before_defer_attention")
    # Never replace a previous backup.
    with backup.open("xb") as out:
        out.write(original.encode("utf-8"))
    path.write_bytes(patched.encode("utf-8"))
    print(f"Patched. Backup: {backup}")
    print("Restart ALL workers with VLLM_ASCEND_SFA_DEFER_ATTN_PREP=1 and enable_mlapo=false.")


if __name__ == "__main__":
    main()
