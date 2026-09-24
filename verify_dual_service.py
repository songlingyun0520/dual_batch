"""HTTP regression cases; supports baseline and split-server comparison."""
import argparse
import json
import math
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--reference')
p.add_argument('--short', action='store_true')
p.add_argument('--base-url', default='http://127.0.0.1:8000')
p.add_argument('--logprob-atol', type=float, default=1e-3)
args = p.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
cases = [(f'b{b}-short', [64] * b) for b in [1, 2, 3, 4, 7, 8]]
if not args.short:
    cases += [('b4-boundary', [2047] * 4), ('b4-beyond', [2060] * 4),
              ('mixed-lengths', [31, 45, 61, 80]), ('reuse-b4', [64] * 4)]
reference = json.loads(Path(args.reference).read_text()) if args.reference else {}
results = {}
for name, lengths in cases:
    payload = dict(model='deepseek-v3.2-shape-demo',
                   prompt=[[100 + i] * n for i, n in enumerate(lengths)],
                   max_tokens=8, temperature=0, ignore_eos=True, logprobs=1)
    req = urllib.request.Request(args.base_url.rstrip('/') + '/v1/completions',
                                 data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.load(r)
    choices = sorted(data['choices'], key=lambda c: c['index'])
    assert len(choices) == len(lengths), data
    token_lists = []
    logprob_lists = []
    for choice in choices:
        lp = choice['logprobs']
        assert choice['finish_reason'] == 'length' and len(lp['tokens']) == 8, choice
        assert all(x is not None and math.isfinite(x) for x in lp['token_logprobs']), choice
        token_lists.append(lp['tokens'])
        logprob_lists.append(lp['token_logprobs'])
    record = dict(lengths=lengths, tokens=token_lists, logprobs=logprob_lists,
                  elapsed_s=time.monotonic()-start, finite=True, usage=data['usage'])
    if args.reference:
        assert name in reference, f'Missing baseline case: {name}'
        record['tokens_match_baseline'] = token_lists == reference[name]['tokens']
        record['logprob_max_abs_diff'] = max(abs(x-y) for a,b in zip(logprob_lists,reference[name]['logprobs']) for x,y in zip(a,b))
        assert record['tokens_match_baseline'], (name, record)
        assert record['logprob_max_abs_diff'] <= args.logprob_atol, (name, record)
    results[name] = record
    Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(dict(case=name, **{k:v for k,v in record.items() if k not in ['tokens','logprobs']})), flush=True)
