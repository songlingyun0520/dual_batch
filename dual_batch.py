"""Decode-only request splitting for the existing fixed-prefix SFA demo.

One compute stream: prepare -> index0 -> index1 -> query prep -> SFA0 ->
SFA1 -> merged projections. KV is already resident; no transfer is simulated.
Opt-in through DSV32_DUAL_BATCH=1 after demo_patch.register().
"""
import copy
import json
import os
from pathlib import Path


def register():
    import torch
    import torch_npu
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    if getattr(AscendSFAImpl, '_demo_dual_batch', False):
        return
    original_forward = AscendSFAImpl.forward
    validate = os.environ.get('DSV32_DUAL_BATCH_VALIDATE') == '1'
    validated = set()
    scope = torch.profiler.record_function

    def slice_metadata(md, lo, hi):
        """Pure decode has one query token per request; preserve physical slots."""
        sub = copy.copy(md)
        sub.num_input_tokens = sub.num_actual_tokens = hi - lo
        sub.num_decodes = sub.num_decode_tokens = hi - lo
        sub.num_prefills = 0
        for field in ('slot_mapping', 'seq_lens', 'seq_lens_cpu', 'block_table',
                      'cos', 'sin', '_demo_horizons', '_demo_block_table',
                      '_demo_write_slots'):
            setattr(sub, field, getattr(md, field)[lo:hi])
        # vLLM SFA uses cumulative query END offsets (no leading zero).
        sub.cum_query_lens = md.cum_query_lens[lo:hi] if lo == 0 else md.cum_query_lens[lo:hi] - lo
        return sub

    def index(self, x, q_c, cache, md):
        k_li, k_scale = self.indexer_select_pre_process(x, md.cos, md.sin)
        assert k_scale is None, 'First version only supports the demo BF16 index cache'
        torch_npu.npu_scatter_nd_update_(
            cache[2].view(-1, k_li.shape[-1]),
            md.slot_mapping.view(-1, 1), k_li.view(-1, k_li.shape[-1]))
        return self.indexer_select_post_process(
            x, q_c, cache, md, md.cos, md.sin, md.cum_query_lens, md.seq_lens)

    def execute(self, layer_name, hidden_states, kv_cache, md, output):
        batch = md.num_input_tokens
        cut = (batch + 1) // 2
        groups = (slice_metadata(md, 0, cut), slice_metadata(md, cut, batch))

        with scope('dual_batch.common'):
            qkv = self.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv = qkv.split([self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            q_c = self.q_a_layernorm(q_c)
            # This writes current-token KV once; it is NOT historical KV loading.
            self.exec_kv(kv, md.cos, md.sin, kv_cache, md.slot_mapping, md)

        with scope('dual_batch.I0'):
            idx0 = index(self, hidden_states[:cut], q_c[:cut], kv_cache, groups[0])
        # HBM-hit completion point for B0. Future async loading hooks belong here.
        with scope('dual_batch.I1'):
            idx1 = index(self, hidden_states[cut:], q_c[cut:], kv_cache, groups[1])
        # HBM-hit completion point for B1. Do not add a global synchronization.

        with scope('dual_batch.P'):
            ql_nope, q_pe = self._q_proj_and_k_up_proj(q_c)
            q_pe = self.rope_single(q_pe, md.cos, md.sin)
            # Prepare contiguous group views before either SFA to keep the
            # measured windows free of group-specific layout work after A0.
            ql_nope = ql_nope.contiguous()
            q_pe = q_pe.contiguous()
            key_lens = md.seq_lens.clamp(max=2048)
            assert idx0.shape[1:] == idx1.shape[1:]
            ids = torch.arange(2048, device=idx0.device, dtype=idx0.dtype)
            shape = (batch,) + tuple(idx0.shape[1:])
            ids = ids.view(*([1] * (len(shape) - 1)), 2048).expand(shape)
            horizons = md._demo_horizons.reshape(batch, *([1] * (len(shape) - 1)))
            fixed_indices = torch.where(ids < horizons, ids, -1).contiguous()

        def attention(lo, hi, group):
            return torch.ops._C_ascend.npu_sparse_flash_attention(
                query=ql_nope[lo:hi], key=kv_cache[0], value=kv_cache[0],
                sparse_indices=fixed_indices[lo:hi], scale_value=self.scale,
                sparse_block_size=1, block_table=group._demo_block_table,
                actual_seq_lengths_query=group.cum_query_lens,
                actual_seq_lengths_kv=key_lens[lo:hi],
                query_rope=q_pe[lo:hi], key_rope=kv_cache[1],
                layout_query='TND', layout_kv='PA_BSND', sparse_mode=3)

        with scope('dual_batch.A0'):
            out0 = attention(0, cut, groups[0])
        with scope('dual_batch.A1'):
            out1 = attention(cut, batch, groups[1])
        with scope('dual_batch.merge_and_project'):
            merged = torch.cat((out0, out1), dim=0)
            output[...] = self.o_proj(self._v_up_proj(merged))[0]
        return output

    def forward(self, layer_name, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv=False, output=None):
        md = attn_metadata
        # Existing path handles profiling, B=1, prefill, and mixed steps.
        eligible = (md is not None and getattr(md, '_demo_decode', False)
                    and md.num_input_tokens >= 2
                    and md.num_actual_tokens == md.num_input_tokens
                    and md.seq_lens.shape[0] == md.num_input_tokens
                    and hidden_states.shape[0] == md.num_input_tokens)
        if not eligible:
            return original_forward(self, layer_name, hidden_states, kv_cache,
                                    md, need_gather_q_kv, output)
        assert output is not None
        assert not self.enable_mlapo and not self.enable_dsa_cp
        assert not self.use_sparse_c8_indexer and not need_gather_q_kv
        assert not self.is_kv_producer, 'KV connector is outside this demo implementation'

        # Validation runs only under the explicitly selected eager launcher.
        # The reference and split write the same current-token slots, so replay
        # is idempotent; also compare those cache rows after both executions.
        context_bucket = None
        key = None
        if validate:
            context_bucket = 'within_prefix' if int(md.seq_lens_cpu.max()) <= 2048 else 'beyond_prefix'
            key = (layer_name, md.num_input_tokens, context_bucket)
        do_check = validate and key not in validated
        if do_check:
            assert not torch.npu.is_current_stream_capturing(), 'Validation requires eager execution'
            reference = torch.empty_like(output)
            original_forward(self, layer_name, hidden_states, kv_cache, md, False, reference)
            index_rows = kv_cache[2].view(-1, 128)[md.slot_mapping.long()].clone()
            write_slots = md._demo_write_slots.long()
            valid_slots = write_slots[write_slots >= 0]
            kv_rows = [c.view(-1, c.shape[-1])[valid_slots].clone() for c in kv_cache[:2]]

        result = execute(self, layer_name, hidden_states, kv_cache, md, output)
        if do_check:
            delta = (reference.float() - result.float()).abs()
            max_abs = float(delta.max())
            rms_relative = float(delta.square().mean().sqrt() / reference.float().square().mean().sqrt().clamp(min=1e-12))
            close = bool(torch.allclose(reference, result, rtol=0.01, atol=0.001))
            cache_equal = bool(torch.equal(index_rows, kv_cache[2].view(-1, 128)[md.slot_mapping.long()]))
            cache_equal = cache_equal and all(torch.equal(old, c.view(-1, c.shape[-1])[valid_slots])
                                            for old, c in zip(kv_rows, kv_cache[:2]))
            record = dict(layer=layer_name, batch=md.num_input_tokens,
                          groups=[(md.num_input_tokens + 1)//2, md.num_input_tokens//2],
                          context_bucket=context_bucket, max_abs=max_abs,
                          rms_relative=rms_relative, output_close=close, cache_equal=cache_equal)
            path = Path(os.environ.get('DSV32_OUTPUT_DIR', os.environ['DSV32_DEMO_DIR'])) / 'dual_batch_validation.jsonl'
            with path.open('a') as f:
                f.write(json.dumps(record) + '\n')
            assert close and rms_relative <= 0.005 and cache_equal, record
            validated.add(key)
        return result

    AscendSFAImpl.forward = forward
    AscendSFAImpl._demo_dual_batch = True
    print('DUAL_BATCH enabled: request halves, I0 I1 P A0 A1, all KV resident; validation='+str(validate), flush=True)
