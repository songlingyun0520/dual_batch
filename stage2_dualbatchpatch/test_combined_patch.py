"""CPU reference tests of the real forward control flow and request metadata.

Run with Python + NumPy. These tests do NOT emulate ACL Graph or CANN kernels.
"""
import ast
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parent


class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def view(self, *shape):
        return Tensor(np.asarray(self).reshape(*shape))

    def split(self, sizes, dim=-1):
        return tuple(Tensor(a) for a in np.split(np.asarray(self), np.cumsum(sizes)[:-1], axis=dim))

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self))

    def fill_(self, value):
        self.fill(value)
        return self


def load_functions(source, scope):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendSFAImpl")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name in {"_slice_decode_metadata", "_validate_dual_decode_layout"}]
    for node in [*helpers, fn]:
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
    exec(compile(ast.Module(body=[*helpers, fn], type_ignores=[]), "<real-forward-body>", "exec"), scope)
    return scope


def fixture(capacity, actual, cpu_lengths=True):
    rng = np.random.default_rng(456)
    hidden = Tensor(rng.normal(size=(capacity, 6)))
    # Every request has its own paged cache region; table entries are physical
    # page IDs and are intentionally shuffled to catch accidental rebasing.
    pages = Tensor(rng.permutation(capacity * 2).reshape(capacity, 2))
    lengths = Tensor((np.arange(capacity) % 5) + 2)
    lengths[actual:] = 0
    block_size = 4
    slots = Tensor(np.full(capacity, -1, dtype=np.int64))
    for row in range(actual):
        pos = int(lengths[row]) - 1
        slots[row] = pages[row, pos // block_size] * block_size + pos % block_size
    md = NS(num_input_tokens=capacity, num_actual_tokens=actual,
            num_decodes=actual, num_decode_tokens=actual, num_prefills=0,
            attn_state="decode", dcp_context=None, dsa_cp_context=None, sfa_cp_metadata=None,
            cos=Tensor(np.ones((capacity, 1))), sin=Tensor(np.zeros((capacity, 1))),
            slot_mapping=slots, block_table=pages, block_size=block_size,
            seq_lens=lengths, seq_lens_cpu=lengths.copy() if cpu_lengths else None,
            cum_query_lens=Tensor(np.arange(1, capacity + 1, dtype=np.int32)))
    caches = tuple(Tensor(rng.normal(size=(capacity * 8, dim))) for dim in (3, 1, 3))
    return hidden, md, caches


def execute(source, capacity=8, actual=8, dual=False, defer=False, state="decode", cpu_lengths=True,
            changes=None, md_changes=None, sp=False):
    hidden, md, caches = fixture(capacity, actual, cpu_lengths)
    md.attn_state = state
    if md_changes:
        md.__dict__.update(md_changes)
    rng = np.random.default_rng(123)
    qkv_w = rng.normal(size=(6, 8))
    index_k_w = rng.normal(size=(6, 3))
    index_q_w = rng.normal(size=(4, 3))
    query_w = rng.normal(size=(4, 4))
    output_w = rng.normal(size=(3, 6))
    calls = []
    call_sizes = []
    captures = []

    class Linear:
        def __init__(self, weight, name): self.weight, self.name = weight, name
        def __call__(self, x):
            calls.append(self.name)
            call_sizes.append((self.name, x.shape[0]))
            return (Tensor(x @ self.weight),)

    def scatter(cache, slots, values):
        calls.append("indexer_cache_write")
        for row, slot in enumerate(np.asarray(slots).reshape(-1)):
            if slot >= 0:
                cache[int(slot)] = values[row]
        return cache

    def kv_prepare(kv, cos, sin, kv_cache, slots, metadata):
        calls.append("attention_KV")
        call_sizes.append(("attention_KV", kv.shape[0]))
        for row, slot in enumerate(slots):
            if slot >= 0:
                kv_cache[0][int(slot)] = kv[row, :3]
                kv_cache[1][int(slot)] = kv[row, 3:]
        return None, None

    def q_prepare(q_c):
        calls.append("attention_Q")
        call_sizes.append(("attention_Q", q_c.shape[0]))
        q = Tensor(q_c @ query_w)
        return q[:, :3], q[:, 3:]

    def index_pre(x, **kwargs):
        calls.append("indexer_K")
        return Tensor(x @ index_k_w), None

    def positions(metadata, row, length):
        return np.array([int(metadata.block_table[row, pos // 4]) * 4 + pos % 4
                         for pos in range(length)], dtype=np.int64)

    def index_post(x, q_c, kv_cache, attn_metadata, actual_seq_lengths_query, actual_seq_lengths_key, **kwargs):
        calls.append("LI")
        n = x.shape[0]
        np.testing.assert_array_equal(actual_seq_lengths_query, np.arange(1, n + 1))
        assert kv_cache is caches  # Never copy/slice physical cache by group.
        captures.append((attn_metadata.block_table.copy(), actual_seq_lengths_key.copy(), caches[2].copy()))
        q = q_c @ index_q_w
        topk = np.full((n, 2), -1, dtype=np.int64)
        for row, length in enumerate(actual_seq_lengths_key):
            if length > 0:
                score = caches[2][positions(attn_metadata, row, int(length))] @ q[row]
                # Return logical sequence positions, not physical cache slots.
                chosen = np.argsort(score)[-min(2, int(length)):]
                topk[row, :len(chosen)] = chosen
        return Tensor(topk)

    def sfa(q, q_pe, kv_cache, topk, metadata, query_lens, key_lens):
        calls.append("SFA")
        n = q.shape[0]
        np.testing.assert_array_equal(query_lens, np.arange(1, n + 1))
        assert kv_cache is caches
        result = np.zeros((n, 3))
        for row in range(n):
            chosen = np.asarray(topk[row][topk[row] >= 0])
            if len(chosen):
                physical = positions(metadata, row, int(key_lens[row]))[chosen]
                kv = caches[0][physical]
                logits = kv @ q[row] + (caches[1][physical] @ q_pe[row])
                probs = np.exp(logits - logits.max())
                result[row] = probs @ kv / probs.sum()
        return Tensor(result)

    def identity(name):
        def f(x, *args):
            calls.append(name)
            return x
        return f

    def normalize(x):
        calls.append("q_c_norm")
        return Tensor(x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6))

    def concat(xs, dim):
        calls.append("concat")
        return Tensor(np.concatenate(xs, axis=dim))

    impl = NS(enable_mlapo=False, enable_dsa_cp=False, enable_sp=sp,
              enable_dsa_cp_with_layer_shard=False, enable_dsa_cp_with_o_proj_tp=False,
              enable_sfa_prolog_v3=False, enable_sparse_sfa_c8=False, enable_sparse_li_c8=False,
              has_indexer=True, skip_topk=False, use_index_cache=False, is_kv_producer=False,
              vllm_config=NS(kv_transfer_config=None, parallel_config=NS(), speculative_config=None,
                             compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY"))),
              fused_qkv_a_proj=Linear(qkv_w, "QKV"), q_lora_rank=4, kv_lora_rank=3, qk_rope_head_dim=1,
              q_a_layernorm=normalize, indexer_select_pre_process=index_pre,
              exec_kv=kv_prepare, _q_proj_and_k_up_proj=q_prepare, rope_single=identity("Q_rope"),
              _get_full_kv=identity("full_indexer_K"), indexer_select_post_process=index_post,
              _execute_sparse_flash_attention_process=sfa, _v_up_proj=identity("V"),
              o_proj=Linear(output_w, "O"),
              _record_dcp_query_gather_context=lambda *a: None,
              _record_dcp_kv_gather_context=lambda *a: None)
    if changes:
        impl.__dict__.update(changes)
    scope = dict(
        _call_sizes=call_sizes,
        copy=copy, _SFA_DUAL_BATCH=dual, _SFA_DEFER_ATTN_PREP=defer or dual,
        AscendAttentionState=NS(DecodeOnly="decode", SpecDecoding="spec"),
        AscendDeviceType=NS(A5="A5"), get_ascend_device_type=lambda: "910C",
        torch=NS(Tensor=Tensor, cat=concat, profiler=NS(record_function=lambda label: nullcontext()),
                 ops=NS(vllm=NS(maybe_all_gather_and_maybe_unpad=lambda x, flag: x))),
        torch_npu=NS(npu_scatter_nd_update_=scatter),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kw: None),
        wait_for_kv_layer_from_connector=lambda *a: None,
        maybe_save_kv_layer_to_connector=lambda *a: None,
        get_ascend_config=lambda: NS(c8_enable_reshape_optim=False),
        MLAPO_MAX_SUPPORTED_TOKENS=16, MAX_O_PROJ_PREFETCH_SIZE=1,
    )
    scope = load_functions(source, scope)
    output = Tensor(np.zeros((capacity, 6)))
    result = scope["forward"](impl, "layer", hidden, caches, md, output=output)
    return result, caches, calls, captures, scope


class CombinedPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (ROOT / "sfa_v1.upstream.py").read_text()
        cls.patched = (ROOT / "sfa_v1.py").read_text()

    def test_numerical_reference_and_order(self):
        cases = 0
        for capacity in (1, 2, 3, 4, 5, 7, 8, 12, 16):
            for actual in sorted({0, 1, max(1, capacity // 2), capacity - 1, capacity}):
                for cpu_lengths in (False, True):
                    kwargs = dict(capacity=capacity, actual=actual, cpu_lengths=cpu_lengths)
                    baseline = execute(self.original, **kwargs)
                    for dual, defer in ((False, False), (False, True), (True, False), (True, True)):
                        result = execute(self.patched, dual=dual, defer=defer, **kwargs)
                        np.testing.assert_allclose(result[0], baseline[0], rtol=1e-12, atol=1e-12)
                        for a, b in zip(result[1], baseline[1]):
                            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
                        calls = result[2]
                        groups = 2 if dual and capacity >= 2 else 1
                        self.assertEqual(calls.count("LI"), groups)
                        self.assertEqual(calls.count("SFA"), groups)
                        for name in ("attention_KV", "attention_Q", "V", "O"):
                            self.assertEqual(calls.count(name), 1)
                        for name in ("QKV", "q_c_norm", "indexer_K", "indexer_cache_write"):
                            self.assertEqual(calls.count(name), groups)
                        if defer or dual:
                            self.assertLess(max(i for i, x in enumerate(calls) if x == "LI"), calls.index("attention_KV"))
                            self.assertLess(calls.index("Q_rope"), calls.index("SFA"))
                        else:
                            self.assertEqual(calls, baseline[2])
                        if groups == 2:
                            prefix = ["QKV", "q_c_norm", "indexer_K", "full_indexer_K", "indexer_cache_write", "LI"]
                            self.assertEqual(calls, prefix * 2 + ["concat", "concat", "attention_KV",
                                "attention_Q", "Q_rope", "SFA", "SFA", "concat", "V", "O"])
                            cut = (capacity + 1) // 2
                            sizes = result[4]["_call_sizes"]
                            self.assertEqual([n for name, n in sizes if name == "QKV"], [cut, capacity - cut])
                            self.assertEqual([n for name, n in sizes if name == "attention_KV"], [capacity])
                            self.assertEqual([n for name, n in sizes if name == "attention_Q"], [capacity])
                            tables = np.concatenate([x[0] for x in result[3]])
                            np.testing.assert_array_equal(tables, baseline[3][0][0])
                        cases += 1
        print(f"Validated {cases} forward/reference cases (including wholly padded groups).")

    def test_a_indexer_does_not_wait_for_b_cache_write(self):
        result = execute(self.patched, capacity=8, actual=8, dual=True)
        _, md, initial_caches = fixture(8, 8)
        first_indexer_cache = result[3][0][2]
        # At LI A, A's cache must be current, while B's slots remain untouched.
        a_slots = np.asarray(md.slot_mapping[:4])
        b_slots = np.asarray(md.slot_mapping[4:])
        np.testing.assert_allclose(first_indexer_cache[a_slots], result[1][2][a_slots])
        np.testing.assert_array_equal(first_indexer_cache[b_slots], initial_caches[2][b_slots])
        self.assertFalse(np.array_equal(first_indexer_cache[b_slots], result[1][2][b_slots]))

    def test_prefill_and_spec_unchanged(self):
        for state in ("prefill", "spec"):
            a = execute(self.original, state=state)
            b = execute(self.patched, dual=True, state=state)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[2], b[2])

    def test_sp_gather_precedes_split(self):
        a = execute(self.original, sp=True)
        b = execute(self.patched, dual=True, sp=True)
        np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=1e-12)

    def test_guards(self):
        for bad in ({"enable_mlapo": True}, {"enable_dsa_cp": True},
                    {"enable_sparse_sfa_c8": True}, {"enable_sparse_li_c8": True},
                    {"is_kv_producer": True}, {"has_indexer": False},
                    {"skip_topk": True}, {"use_index_cache": True}, {"enable_sfa_prolog_v3": True}):
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                execute(self.patched, dual=True, changes=bad)
        for bad in ({"dcp_context": object()}, {"sfa_cp_metadata": object()}):
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                execute(self.patched, dual=True, md_changes=bad)
        for field, value in (("kv_transfer_config", object()), ("speculative_config", object()),
                             ("compilation_config", NS(cudagraph_mode=NS(name="FULL"))),
                             ("parallel_config", NS(decode_context_parallel_size=2))):
            cfg = NS(kv_transfer_config=None, speculative_config=None, parallel_config=NS(),
                     compilation_config=NS(cudagraph_mode=NS(name="FULL_DECODE_ONLY")))
            setattr(cfg, field, value)
            with self.assertRaisesRegex(RuntimeError, "does not support"):
                execute(self.patched, dual=True, changes={"vllm_config": cfg})
        with self.assertRaisesRegex(RuntimeError, "mismatched tensors"):
            execute(self.patched, dual=True, md_changes={"seq_lens": Tensor([1, 2, 3])})

    def test_rebase_uses_current_device_values(self):
        scope = execute(self.patched, dual=True)[4]
        _, md, _ = fixture(8, 8)
        helper = scope["_slice_decode_metadata"]
        a = helper(md, 4, 8)
        np.testing.assert_array_equal(a.cum_query_lens, [1, 2, 3, 4])
        # Demonstrate arithmetic is not hardcoded to lo or cached on CPU.
        md.cum_query_lens[:] = [2, 4, 6, 8, 10, 12, 14, 16]
        b = helper(md, 4, 8)
        np.testing.assert_array_equal(b.cum_query_lens, [2, 4, 6, 8])
        self.assertTrue(np.shares_memory(a.block_table, md.block_table))
        self.assertTrue(np.shares_memory(a.seq_lens, md.seq_lens))


if __name__ == "__main__":
    unittest.main(verbosity=2)
