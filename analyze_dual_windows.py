"""Measure each layer directly between device Indexer/SFA boundaries.

W0 = end(I0) -> start(A0); W1 = end(I1) -> start(A1).
W1 is the budget available from I1 completion. A shared transfer path can
delay B1's actual load start: its remaining budget is W1-max(0,L0-C1).
Kernel busy time is reported separately from elapsed time; idle gaps are
not relabelled as effective compute. No division of whole-model latency.
"""
import argparse
import bisect
import collections
import csv
import json
import statistics
from decimal import Decimal
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('profile_output', type=Path)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--batch', type=int, choices=[2, 4, 8], required=True)
args = p.parse_args()
rows = list(csv.DictReader((args.profile_output/'kernel_details.csv').open()))
assert rows, 'kernel_details.csv is empty'
origin = min(Decimal(r['Start Time(us)'].strip()) for r in rows)
for r in rows:
    r['_start'] = float(Decimal(r['Start Time(us)'].strip())-origin)
    r['_end'] = r['_start']+float(r['Duration(us)'])
rows.sort(key=lambda r:r['_start'])
special = [r for r in rows if r['Type'] in ('LightningIndexerVllm','SparseFlashAttention')]
assert special, 'No Indexer/SFA kernels found; check profiler output and kernel names'
assert len(special) % 4 == 0, ('partial layer in capture', len(special))
assert len(special) % (61*4) == 0, ('partial step in capture', len(special))
steps = len(special)//(61*4)
intervals_by_model = collections.defaultdict(list)
for r in rows:
    merged = intervals_by_model[r['Model ID']]
    lo,hi = r['_start'],r['_end']
    if merged and lo <= merged[-1][1]:
        merged[-1] = (merged[-1][0],max(hi,merged[-1][1]))
    else:
        merged.append((lo,hi))
starts_by_model = {m:[a for a,b in xs] for m,xs in intervals_by_model.items()}

def busy(lo, hi, model_id):
    # Kernel intervals within a single replayed model; union avoids double counting.
    starts=starts_by_model[model_id]
    i = max(0,bisect.bisect_left(starts,lo)-1)
    j = bisect.bisect_left(starts,hi)
    return sum(max(0.0,min(hi,b)-max(lo,a)) for a,b in intervals_by_model[model_id][i:j])

samples=[]
for k in range(0,len(special),4):
    i0,i1,a0,a1=special[k:k+4]
    assert [r['Type'] for r in (i0,i1,a0,a1)] == ['LightningIndexerVllm']*2+['SparseFlashAttention']*2
    assert len({r['Model ID'] for r in (i0,i1,a0,a1)})==1
    assert i0['_end'] <= i1['_start'] and i1['_end'] <= a0['_start'] and a0['_end'] <= a1['_start']
    for r in (i0,i1,a0,a1):
        assert r['Input Shapes'].strip('"').startswith(str(args.batch//2)+','), r['Input Shapes']
    mid=i0['Model ID']
    sample=dict(step=k//(61*4), layer=(k//4)%61,
                I0_end_us=i0['_end'], I1_end_us=i1['_end'],
                A0_start_us=a0['_start'], A0_end_us=a0['_end'], A1_start_us=a1['_start'],
                C1_elapsed_us=i1['_end']-i0['_end'],
                P_elapsed_us=a0['_start']-i1['_end'],
                A0_kernel_us=a0['_end']-a0['_start'],
                A1_setup_gap_us=a1['_start']-a0['_end'],
                W0_elapsed_us=a0['_start']-i0['_end'],
                W1_elapsed_us=a1['_start']-i1['_end'],
                C1_busy_us=busy(i0['_end'],i1['_end'],mid),
                P_busy_us=busy(i1['_end'],a0['_start'],mid),
                W0_busy_us=busy(i0['_end'],a0['_start'],mid),
                W1_busy_us=busy(i1['_end'],a1['_start'],mid),
                I0_kernel_us=float(i0['Duration(us)']), I1_kernel_us=float(i1['Duration(us)']),
                A1_kernel_us=float(a1['Duration(us)']))
    samples.append(sample)

def summary(values):
    values=sorted(values)
    def pct(p):
        x=(len(values)-1)*p
        lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p05=pct(.05),p95=pct(.95),minimum=min(values),maximum=max(values))

fields=[k for k in samples[0] if k.endswith('_us') and k not in ['I0_end_us','I1_end_us','A0_start_us','A0_end_us','A1_start_us']]
stats={f:summary([s[f] for s in samples]) for f in fields}
per_layer={str(layer):{f:summary([s[f] for s in samples if s['layer']==layer]) for f in fields} for layer in range(61)}
result=dict(batch=args.batch,groups=[args.batch//2]*2,steps=steps,layers=61,
            layer_samples=len(samples),units='microseconds',
            source=str(args.profile_output),statistics=stats,per_layer=per_layer,
            definitions=dict(W0='end I0 -> start SFA0',W1='end I1 -> start SFA1',
                             busy='union of device kernel intervals; excludes idle gaps',
                             C1='end I0 -> end I1, includes all B1 index preparation',
                             P='end I1 -> start SFA0, includes Q preparation and fixed-index/layout preparation'),
            limitations='HBM-hit baseline only; no transfers, page mapping or copy contention measured. Graph profiler overhead is present. W1 assumes B0 has released the transfer path by end I1; otherwise subtract max(0,L0-C1).')
args.output.mkdir(parents=True,exist_ok=True)
(args.output/'single_layer_windows.json').write_text(json.dumps(result,indent=2))
with (args.output/'single_layer_samples.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
with (args.output/'per_layer_means.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=['layer']+fields);w.writeheader()
    for layer,values in per_layer.items():w.writerow(dict(layer=layer,**{k:v['mean'] for k,v in values.items()}))
print(json.dumps({k:v for k,v in result.items() if k!='per_layer'},indent=2))
