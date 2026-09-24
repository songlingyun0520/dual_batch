"""Capture a stable pure-decode window on the actual 61-layer service."""
import argparse
import json
import math
import os
import time
import urllib.request
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--batch', type=int, choices=[2, 4, 8], required=True)
p.add_argument('--context', type=int, default=32768)
p.add_argument('--output-tokens', type=int, default=64)
p.add_argument('--base-url', default='http://127.0.0.1:8000')
p.add_argument('--run-dir', type=Path, default=Path(os.environ.get('DSV32_OUTPUT_DIR', Path(__file__).resolve().parent / 'runtime')))
args = p.parse_args()
if args.output_tokens < 32:
    p.error('--output-tokens must be at least 32 to capture tokens 8..24')
if not 1 <= args.context or args.context + args.output_tokens > 32896:
    p.error('context must be positive and context + output-tokens <= 32896')
root = args.run_dir.resolve()
out = root/'dual_measurements'/f'b{args.batch}-ctx{args.context}-{time.time_ns()}'
out.mkdir(parents=True)
base = args.base_url.rstrip('/')

def call(path):
    req = urllib.request.Request(base+path, data=b'{}', headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return r.read()

def traces():
    return set(str(p) for p in (root/'profiles').glob('*'))

payload = dict(model='deepseek-v3.2-shape-demo',
               prompt=[[100+i]*args.context for i in range(args.batch)],
               max_tokens=args.output_tokens, temperature=0, ignore_eos=True,
               logprobs=1, stream=True, stream_options={'include_usage':True})
req = urllib.request.Request(base+'/v1/completions', data=json.dumps(payload).encode(),
                             headers={'Content-Type':'application/json'})
counts = [0]*args.batch
logprobs = []
active = False
captured = False
before = set()
manifest = dict(batch=args.batch, groups=[args.batch//2]*2, input_tokens=args.context,
                output_tokens=args.output_tokens, hbm_hit_rate=1.0,
                timing_mode='ACL graph decode; profiler device kernel timestamps')
start = time.monotonic()
print('Submitted '+str(out), flush=True)
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            if not raw.startswith(b'data: '):
                continue
            data = raw[6:].strip()
            if data == b'[DONE]':
                break
            chunk = json.loads(data)
            if chunk.get('error'):
                raise RuntimeError(chunk)
            if chunk.get('usage'):
                manifest['usage'] = chunk['usage']
            for choice in chunk.get('choices', []):
                lp = choice.get('logprobs') or {}
                counts[choice['index']] += len(lp.get('tokens', []))
                logprobs.extend(lp.get('token_logprobs', []))
            if min(counts) >= 8 and not active and not captured:
                before = traces()
                call('/start_profile')
                active = True
                manifest['profile_start_counts'] = counts.copy()
                print('Profile started '+str(counts), flush=True)
            if min(counts) >= 24 and active:
                call('/stop_profile')
                active = False
                captured = True
                manifest['profile_stop_counts'] = counts.copy()
                manifest['trace_dirs'] = sorted(traces()-before)
                print('Profile exported '+str(manifest['trace_dirs']), flush=True)
finally:
    if active:
        call('/stop_profile')
manifest.update(counts=counts, elapsed_s=time.monotonic()-start,
                finite_logprobs=all(x is not None and math.isfinite(x) for x in logprobs))
(out/'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)
assert captured and counts == [args.output_tokens]*args.batch and manifest['finite_logprobs'], manifest
