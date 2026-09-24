"""Opt-in patches for an inaccurate but complete single-NPU V3.2 demo.

Activated only by DSV32_SHAPE_DEMO=1 and this directory on PYTHONPATH.
Does not modify installed vLLM / vLLM-Ascend files.
"""
import os


def register():
    if os.environ.get('DSV32_SHAPE_DEMO') != '1':
        return
    import json
    import torch
    from vllm.model_executor.models import deepseek_v2 as ds
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl, AscendSFAMetadataBuilder
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    if getattr(ds, '_shape_demo_patched', False):
        return
    ds._shape_demo_patched = True
    keep = 2048
    request_slots = {}
    first_block_slots = {}
    allocated_caches = {}
    max_batch_seen = 0
    max_decode_batch_seen = 0
    max_prefill_batch_seen = 0
    metadata_steps = 0
    decode_buffers = {}

    # Keep DeepseekV2MoE.forward, AscendFusedMoE, native fused gating,
    # dispatch/GMM/activation/combine unchanged. Only storage and IDs change.
    from vllm_ascend.ops.fused_moe import fused_moe as fm
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import UnquantizedFusedMoEMethod
    physical_experts = 8
    original_factory = ds.SharedFusedMoE
    original_create = UnquantizedFusedMoEMethod.create_weights
    original_post = fm.AscendUnquantizedFusedMoEMethod.process_weights_after_loading
    original_select = fm.select_experts
    raw_weights = {}
    packed_weights = {}

    def shared_factory(*args, **kwargs):
        assert kwargs['num_experts'] == 256
        kwargs['num_experts'] = physical_experts
        return original_factory(*args, **kwargs)

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs):
        assert num_experts == physical_experts
        key = (hidden_size, intermediate_size_per_partition, params_dtype)
        if key not in raw_weights:
            original_create(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype, **attrs)
            raw_weights[key] = (layer.w13_weight, layer.w2_weight)
        else:
            layer.register_parameter('w13_weight', raw_weights[key][0])
            layer.register_parameter('w2_weight', raw_weights[key][1])

    def post_weights(self, layer):
        key = tuple(layer.w13_weight.shape)
        if key not in packed_weights:
            original_post(self, layer)
            packed_weights[key] = (layer.w13_weight, layer.w2_weight)
        else:
            layer.w13_weight, layer.w2_weight = packed_weights[key]

    def select_experts(*args, **kwargs):
        # Actual vLLM-Ascend fused top-k still sees 256 router scores.
        weights, ids = original_select(*args, **kwargs)
        return weights, ids.remainder(physical_experts)

    ds.SharedFusedMoE = shared_factory
    UnquantizedFusedMoEMethod.create_weights = create_weights
    fm.AscendUnquantizedFusedMoEMethod.process_weights_after_loading = post_weights
    fm.select_experts = select_experts

    original_allocate = NPUModelRunner._allocate_kv_cache_tensors
    original_reshape = NPUModelRunner._reshape_kv_cache_tensors

    def allocate(self, cache_config):
        if not self.use_sparse:
            return original_allocate(self, cache_config)
        capacity = self.scheduler_config.max_num_seqs
        assert not self.use_sparse_c8_indexer
        specs = self._get_layer_kv_cache_specs(cache_config)
        caches = {}
        for name, spec in specs.items():
            if name in self.runner_only_attn_layers:
                continue
            assert spec.block_size == 128
            # Attention KV is physically limited to the first 2048 positions.
            # Indexer keys retain full logical context so real top-k still runs.
            caches[name] = (
                torch.zeros((capacity * 16, 128, 1, 512), dtype=spec.dtype, device=self.device),
                torch.zeros((capacity * 16, 128, 1, 64), dtype=spec.dtype, device=self.device),
                torch.zeros((cache_config.num_blocks, 128, 1, 128), dtype=spec.dtype, device=self.device),
            )
        audit = {
            'layers': len(caches), 'max_num_seqs': capacity,
            'attention_cache_tokens_per_sequence_per_layer': keep,
            'logical_blocks': cache_config.num_blocks,
            'index_cache_tokens_per_layer': cache_config.num_blocks * 128,
            'attention_cache_bytes': sum(t.numel()*t.element_size() for c in caches.values() for t in c[:2]),
            'index_cache_bytes': sum(c[2].numel()*c[2].element_size() for c in caches.values()),
        }
        print('DEMO_CACHE '+json.dumps(audit), flush=True)
        from pathlib import Path
        Path(os.environ.get('DSV32_OUTPUT_DIR', os.environ['DSV32_DEMO_DIR']), 'cache_audit.json').write_text(json.dumps(audit, indent=2))
        allocated_caches.update(caches)
        return caches

    def reshape(self, config, raw):
        if self.use_sparse:
            return raw
        return original_reshape(self, config, raw)

    NPUModelRunner._allocate_kv_cache_tensors = allocate
    NPUModelRunner._reshape_kv_cache_tensors = reshape
    original_exec_kv = AscendSFAImpl.exec_kv
    original_indexer = AscendSFAImpl.indexer_select_post_process
    original_update = NPUModelRunner._update_states
    original_build = AscendSFAMetadataBuilder.build
    original_zero = NPUModelRunner._zero_block_ids

    def zero_blocks(self, block_ids):
        if not self.use_sparse:
            return original_zero(self, block_ids)
        # Scheduler block IDs address only the full indexer cache. Compact
        # attention storage is cleared separately when a request gets a slot.
        ids = torch.tensor(block_ids, device=self.device, dtype=torch.long)
        for cache in allocated_caches.values():
            cache[2].index_fill_(0, ids, 0)

    def update_states(self, scheduler_output):
        for req in scheduler_output.finished_req_ids:
            request_slots.pop(req, None)
        for req in getattr(scheduler_output, 'preempted_req_ids', None) or ():
            request_slots.pop(req, None)
        original_update(self, scheduler_output)
        for req in scheduler_output.num_scheduled_tokens:
            if req not in request_slots:
                free = set(range(self.scheduler_config.max_num_seqs)) - set(request_slots.values())
                assert free, 'No free compact KV slot'
                slot = request_slots[req] = min(free)
                for cache in allocated_caches.values():
                    cache[0][slot*16:(slot+1)*16].zero_()
                    cache[1][slot*16:(slot+1)*16].zero_()
        first_block_slots.clear()
        for req, slot in request_slots.items():
            state = self.requests.get(req)
            if state is not None and state.block_ids[0]:
                first_block_slots[state.block_ids[0][0]] = slot

    def build_metadata(self, common_prefix_len, common_attn_metadata, fast_build=False):
        nonlocal max_batch_seen, max_decode_batch_seen, max_prefill_batch_seen, metadata_steps
        md = original_build(self, common_prefix_len, common_attn_metadata, fast_build)
        num_reqs = common_attn_metadata.num_reqs
        first_blocks = md.block_table[:, 0].cpu().tolist()
        slots = [first_block_slots[b] for b in first_blocks] if first_block_slots else list(range(len(first_blocks)))
        starts = common_attn_metadata.query_start_loc_cpu[:num_reqs+1].tolist()
        lengths = md.seq_lens_cpu.tolist()
        rows, write_slots, horizons = [], [], []
        for i, (end, slot) in enumerate(zip(lengths, slots)):
            count = starts[i+1] - starts[i]
            begin = end - count
            for offset in range(count):
                position = begin + offset
                if position < keep:
                    rows.append(starts[i] + offset)
                    write_slots.append(slot * keep + position)
                horizons.append(min(end, keep) - count + offset + 1)
        device = md.slot_mapping.device
        md._demo_rows = torch.tensor(rows, dtype=torch.long, device=device)
        md._demo_write_slots = torch.tensor(write_slots, dtype=md.slot_mapping.dtype, device=device)
        md._demo_has_writes = bool(rows)
        md._demo_horizons = torch.tensor(horizons, dtype=torch.int32, device=device)
        base = torch.tensor(slots, dtype=md.block_table.dtype, device=device)[:, None] * 16
        md._demo_block_table = base + torch.arange(16, dtype=md.block_table.dtype, device=device)[None, :]
        query_sizes = [starts[i+1]-starts[i] for i in range(num_reqs)]
        md._demo_decode = all(q == 1 for q in query_sizes)
        if md._demo_decode:
            # Fixed-address inputs consumed by the captured native kernels.
            positions = [end-1 for end in lengths]
            writes = [slot*keep+pos if 0 <= pos < keep else -1
                      for slot,pos in zip(slots,positions)]
            md._demo_write_slots = torch.tensor(writes, dtype=md.slot_mapping.dtype, device=device)
            fields = ('_demo_write_slots', '_demo_horizons', '_demo_block_table',
                      'cos', 'sin', 'seq_lens', 'cum_query_lens', 'slot_mapping', 'block_table')
            for field in fields:
                value = getattr(md, field)
                key = (id(self), num_reqs, field, tuple(value.shape))
                if key not in decode_buffers:
                    decode_buffers[key] = value.clone()
                else:
                    decode_buffers[key].copy_(value)
                setattr(md, field, decode_buffers[key])

        decode_count = sum(q == 1 for q in query_sizes)
        prefill_count = sum(q > 1 for q in query_sizes)
        metadata_steps += 1
        changed = (num_reqs > max_batch_seen or decode_count > max_decode_batch_seen
                   or prefill_count > max_prefill_batch_seen)
        max_batch_seen = max(max_batch_seen, num_reqs)
        max_decode_batch_seen = max(max_decode_batch_seen, decode_count)
        max_prefill_batch_seen = max(max_prefill_batch_seen, prefill_count)
        if changed or metadata_steps % 32 == 0:
            from pathlib import Path
            audit = {'max_batch_seen': max_batch_seen, 'compact_slots': slots,
                     'query_lengths': query_sizes, 'seq_lengths': lengths,
                     'max_decode_batch_seen': max_decode_batch_seen,
                     'max_prefill_batch_seen': max_prefill_batch_seen,
                     'metadata_steps': metadata_steps}
            Path(os.environ.get('DSV32_OUTPUT_DIR', os.environ['DSV32_DEMO_DIR']), 'batch_audit.json').write_text(json.dumps(audit, indent=2))
            print('DEMO_BATCH '+json.dumps(audit), flush=True)
        return md

    NPUModelRunner._update_states = update_states
    NPUModelRunner._zero_block_ids = zero_blocks
    AscendSFAMetadataBuilder.build = build_metadata

    def exec_kv(self, kv_no_split, cos, sin, kv_cache, slots, metadata):
        if metadata._demo_decode:
            return original_exec_kv(self, kv_no_split, cos, sin, kv_cache,
                                    metadata._demo_write_slots, metadata)
        if metadata._demo_has_writes:
            rows = metadata._demo_rows
            return original_exec_kv(self, kv_no_split[rows], cos[rows], sin[rows], kv_cache,
                                    metadata._demo_write_slots, metadata)
        return None, None

    def indexer(self, *args, **kwargs):
        with torch.profiler.record_function('demo.real_lightning_indexer'):
            return original_indexer(self, *args, **kwargs)

    def sparse(self, ql_nope, q_pe, kv_cache, indices, metadata, query_lens, key_lens):
        retained = key_lens.clamp(max=keep)
        ids = torch.arange(keep, device=indices.device, dtype=indices.dtype)
        ids = ids.view(*([1]*(indices.ndim-1)), keep).expand_as(indices)
        # Keep causal indices valid for the kernel's capped KV length; padding
        # is -1. For later prefill chunks, values intentionally lose accuracy.
        qcount = ql_nope.shape[0]
        horizons = metadata._demo_horizons
        horizons = horizons.reshape(qcount, *([1]*(indices.ndim-1)))
        indices = torch.where(ids < horizons, ids, -1).contiguous()
        block_table = metadata._demo_block_table
        with torch.profiler.record_function('demo.real_sparse_flash_attention'):
            out = torch.ops._C_ascend.npu_sparse_flash_attention(
                query=ql_nope, key=kv_cache[0], value=kv_cache[0], sparse_indices=indices,
                scale_value=self.scale, sparse_block_size=1, block_table=block_table,
                actual_seq_lengths_query=query_lens, actual_seq_lengths_kv=retained,
                query_rope=q_pe, key_rope=kv_cache[1], layout_query='TND',
                layout_kv='PA_BSND', sparse_mode=3,
            )
        if not getattr(self, '_demo_logged_sfa', False):
            self._demo_logged_sfa = True
            print('DEMO_SFA '+json.dumps({'q_shape': list(ql_nope.shape), 'kv_shape': list(kv_cache[0].shape), 'indices_shape': list(indices.shape)}), flush=True)
        return out

    AscendSFAImpl.exec_kv = exec_kv
    AscendSFAImpl.indexer_select_post_process = indexer
    AscendSFAImpl._execute_sparse_flash_attention_process = sparse
    if os.environ.get('DSV32_DUAL_BATCH') == '1':
        from dual_batch import register as register_dual_batch
        register_dual_batch()
    print('DEMO_PATCH active: 61-layer config, shared MoE weights, compact prefix KV, real indexer + SFA', flush=True)
