"""V5/V5.1 unsplit recent-2K profiler support. No NPU dependencies.

Module spans are first start -> last end, with kernel sums/union reported
separately. Pasted fragments never establish an absolute model layer or TPOT.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def shape_list(value):
    def unquote(s):
        s = s.strip()
        while len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
            s = s[1:-1].strip()
        return s
    result = []
    for part in unquote(str(value)).split(';'):
        part = unquote(part)
        if len(part) >= 2 and (part[0], part[-1]) in {('[', ']'), ('(', ')')}:
            part = part[1:-1].strip()
        result.append(tuple(map(int, part.split(',')))
                      if re.fullmatch(r'\d+(?:\s*,\s*\d+)*', part) else ())
    return result


def read_kernel_table(path, device_id=None):
    with open(path, encoding='utf-8-sig', newline='') as f:
        first = f.readline()
    sep = '\t' if '\t' in first else ','
    row = next(csv.reader([first], delimiter=sep))
    has_header = any(str(x).strip().lower() in {'name', 'kernel name', 'op name', 'op_name'} for x in row)
    if has_header:
        frame = pd.read_csv(path, sep=sep, dtype=str, encoding='utf-8-sig')
        frame.columns = [str(x).strip() for x in frame.columns]
        frame['_source_record'] = np.arange(len(frame)) + 2
    else:
        if len(row) < 21 or not re.fullmatch(r'\d+', row[0].strip()):
            raise ValueError('无法识别表头/分隔符；无表头输入须为 Ascend kernel_details 标准列顺序。')
        prefix = ['Device_id','Model ID','Task ID','Stream ID','Name','Type','OP State',
                  'Accelerator Core','Start Time(us)','Duration(us)','Wait Time(us)',
                  'Block Num','Mix Block Num','HF32 Eligible','Input Shapes','Input Data Types',
                  'Input Formats','Output Shapes','Output Data Types','Output Formats','Context ID']
        names = prefix + [f'profiler_extra_{i}' for i in range(21, len(row))]
        frame = pd.read_csv(path, sep=sep, dtype=str, header=None, names=names, encoding='utf-8-sig')
        if not isinstance(frame.index, pd.RangeIndex):
            raise ValueError('无表头记录列数不一致；拒绝自动移动列。')
        frame['_source_record'] = np.arange(len(frame)) + 1
    if device_id is not None:
        col = next((c for c in frame if re.sub(r'[\s_]', '', c.lower()) == 'deviceid'), None)
        if col is None:
            raise ValueError('--device-id 需要 Device_id 列')
        frame = frame[frame[col].astype(str).str.strip().eq(str(device_id))].copy()
    if frame.empty:
        raise ValueError('输入为空或没有匹配的 device')
    return frame


STAGES = [
    ('required_qkv_indexer_k', '输入归一化/QKV/Indexer K 准备'),
    ('ring_history_gather', '历史 attention KV 搬入 prefill 工作区'),
    ('movable_attention_kv_q', 'attention KV 写入与 attention Q 准备'),
    ('required_indexer_cache_qw', 'Indexer cache 写入与 Q/W 准备'),
    ('lightning_indexer', 'Lightning Indexer'),
    ('ring_index_remap', 'Indexer 输出映射到最近 2K'),
    ('sfa_q_transpose', 'SFA 前 Q 转置'),
    ('sparse_flash_attention', 'SparseFlashAttention'),
    ('ring_kv_commit', 'SFA 后最近 2K KV 提交'),
    ('v_up_projection', 'V 上投影'),
    ('o_input_quant', 'O 投影输入量化'),
    ('o_matmul', 'O 投影 MatMul'),
    ('output_communication', 'O 后通信'),
    ('moe_tail', 'MoE 与层末尾'),
]
COMM = {'CommunicationRuntime','AllReduce','AllGather','ReduceScatter','AllToAll'}


def union_time(rows):
    intervals = sorted(zip(rows.start, rows.end))
    total, end = 0., -np.inf
    for a, b in intervals:
        total += max(0., b-max(a,end))
        end = max(end,b)
    return total


def ring_signature(rows):
    names = rows.name.str.lower()
    return (names.str.contains('floormod').any()
            and names.str.contains('gescalar').any()
            and names.str.contains('swher|selectv2', regex=True).any())


def layer_compute_rows(ld, li, sf):
    """Follow a one-way compute-stream handoff within this bounded layer.

    LI and SFA anchor the two streams. Accept a handoff only if all compute
    starts on the old stream precede those on the new stream, with no material
    execution overlap. Communication runtime tasks never establish a handoff.
    This infers an observed order, not an event/dependency or graph boundary.
    """
    pool = ld[~ld.stage.isin(COMM) & ld.stream.notna()].sort_values('start')
    stream = li.iloc[0].stream if len(li) else (sf.iloc[0].stream if len(sf) else np.nan)
    if not np.isfinite(stream):
        counts = pool.stream.value_counts()
        if len(counts):
            stream = counts.index[0]
    old = pool[pool.stream.eq(stream)]
    if len(li) == 1 and len(sf) == 1 and np.isfinite(sf.iloc[0].stream):
        next_stream = sf.iloc[0].stream
        if next_stream != stream:
            new = pool[pool.stream.eq(next_stream)]
            # Tiny profiler timestamp overlap can occur at adjacent kernels.
            if (len(old) and len(new) and old.start.max() < new.start.min()
                    and old.end.max() <= new.start.min() + 0.5):
                return pd.concat([old, new]).sort_values('start'), 'sequential_handoff'
            return old, 'ambiguous_multistream'
    return old, 'single_stream'


def analyze_layer(rows, layer_start=np.nan, layer_end=np.nan, right_complete=False,
                  step_id=np.nan, model_layer=np.nan, end_estimated=False):
    """Classify a bounded unsplit layer/fragment using shapes and ordered anchors.

    Unknown streams/extra kernels are not silently assigned to ring work.
    A stage has explicit row membership, avoiding attribution by wall overlap.
    """
    ld = rows.sort_values('start').copy()
    li = ld[ld.stage.eq('LightningIndexer')]
    sf = ld[ld.stage.eq('SparseFlashAttention')]
    found, members = {}, {}
    prep_location = 'unknown'
    phase = 'unknown'
    mapping_path = 'unidentified'
    capacity = requests = np.nan
    compute, compute_stream_status = layer_compute_rows(ld, li, sf)
    compute_streams = json.dumps([int(x) for x in compute.stream.drop_duplicates()])
    # Do not manufacture single-batch module boundaries for two LI/SFA calls.
    unsupported = len(li)>1 or len(sf)>1
    if len(li)==1:
        shapes=shape_list(li.iloc[0].input_shapes)
        capacity=shapes[0][0] if shapes and shapes[0] else np.nan
        requests=shapes[5][0] if len(shapes)>5 and shapes[5] else np.nan
        if np.isfinite(capacity) and np.isfinite(requests):
            phase='decode_candidate' if capacity==requests else 'prefill_or_dp_padding'

    def add(sid, part, status='measured', first=None, last=None, evidence=''):
        if part.empty:return
        a=float(part.start.min()) if first is None else float(first)
        b=float(part.end.max()) if last is None else float(last)
        found[sid]=(a,b,status,evidence)
        members[sid]=set(part.index)

    def between(a,b):
        return compute[(compute.start>=a-1e-6)&(compute.start<b-1e-6)]

    li_start=float(li.iloc[0].start) if len(li)==1 else np.nan
    li_end=float(li.iloc[0].end) if len(li)==1 else np.nan
    sf_start=float(sf.iloc[0].start) if len(sf)==1 else np.nan
    sf_end=float(sf.iloc[0].end) if len(sf)==1 else np.nan
    # Mapping operates on the top-k matrix, not the 1-D ring write slots.
    mapping=between(li_end,sf_start)
    if not unsupported and ring_signature(mapping):
        begin=mapping[mapping.name.str.contains('GeScalar',case=False)].iloc[0]
        finish=mapping[mapping.name.str.contains('SWhere|SelectV2',case=False,regex=True)].iloc[-1]
        block=between(begin.start,finish.start+1e-5)
        sh=shape_list(begin.input_shapes)
        if sh and len(sh[0])==3 and sh[0][-1]==2048:
            add('ring_index_remap',block,evidence='top-k matrix: validity mask -> modulo -> SelectV2')
            if block.name.str.contains('RemainderTensorTensor',case=False).any():
                mapping_path='prefill_staging'
                phase='prefill_or_mixed'
            elif block.name.str.contains('RemainderTensorScalar',case=False).any():
                mapping_path='decode_ring'
                # A scalar modulo identifies the local implementation path,
                # not one-token-per-request scheduling. Do not override the
                # Q/request-row mismatch (e.g. 4096 token slots, 15 rows).
                if np.isfinite(capacity) and np.isfinite(requests) and capacity == requests:
                    phase='decode'
    # History copy/commit: shape + API names distinguish them from Indexer cache.
    def cache_op(row, token):
        s=shape_list(row.input_shapes)
        return (token in row['name'].lower() and len(s)>=2 and len(s[0])==2
                and s[0][-1] in (512,64) and len(s[1])==1 and 0<s[1][0]<=2048)

    def cache_block(token, sid, before=True):
        candidates=compute[compute.apply(lambda r:cache_op(r,token),axis=1)]
        if candidates.empty:return
        if before and np.isfinite(sf_start):candidates=candidates[candidates.start<sf_start]
        if not before and np.isfinite(sf_end):candidates=candidates[candidates.start>=sf_end-1e-6]
        if candidates.empty:return
        ids=list(compute.index)
        first=ids.index(candidates.index[0]);last=ids.index(candidates.index[-1])
        # Slot-index arithmetic emitted immediately before the first copy.
        allowed=('arange','remaindertensor','aclnnadds_','aclnnadd_')
        while first>0 and any(x in compute.iloc[first-1]['name'].lower() for x in allowed):first-=1
        if before:
            while last+1<len(compute) and 'tensormove' in compute.iloc[last+1]['name'].lower():last+=1
        block=compute.iloc[first:last+1]
        status='measured'
        if before and not compute.name.str.contains('KvRmsNormRopeCache',case=False).any():status='partial'
        if not before and (not len(sf) or last+1==len(compute)):status='partial'
        add(sid,block,status,evidence='cache shape (*,512/64), index length <=2048; '+token)

    if not unsupported:
        cache_block('aclnnindexselect', 'ring_history_gather')
        cache_block('aclnninplaceindexcopy', 'ring_kv_commit',False)
        if 'ring_kv_commit' in found or 'ring_history_gather' in found:
            phase='prefill_or_mixed'
        if len(li):add('lightning_indexer',li)
        if len(sf):add('sparse_flash_attention',sf)
        # A missing Cast is normal when slots already have int64 dtype.
        before_sf=compute[compute.start<sf_start] if np.isfinite(sf_start) else compute
        kv=before_sf[before_sf.name.str.contains('KvRmsNormRopeCache',case=False)]
        if len(kv)==1:
            k=kv.iloc[0];pos=list(compute.index).index(k.name);first=pos
            while first>0:
                previous=compute.iloc[first-1]
                if not any(x in previous['name'].lower() for x in ('cast','slice')):break
                if 'ring_history_gather' in members and previous.name in members['ring_history_gather']:break
                first-=1
            ropes=before_sf[(before_sf.start>k.start)&before_sf.name.str.contains('InterleaveRope',case=False)]
            if len(ropes)==1:
                block=between(compute.iloc[first].start,ropes.iloc[0].start+1e-5)
                if block.stage.isin(['Matmul','QuantMatmul']).any():
                    add('movable_attention_kv_q',block,evidence='optional Cast/Slice -> KV write -> Q InterleaveRope')
            if np.isfinite(li_start):prep_location='before_indexer' if k.start<li_start else 'after_indexer'
        pre=compute[compute.start<li_start]
        scatters=pre[pre.name.str.contains('scatterndupdate',case=False)]
        if len(scatters)==1:
            sc=float(scatters.iloc[0].start)
            # Exclude identified history/attention blocks, including their holes.
            used=set().union(*(members.get(x,set()) for x in ('movable_attention_kv_q','ring_history_gather')))
            prep=pre[(pre.start<sc)&~pre.index.isin(used)]
            if np.isfinite(layer_start) and len(prep):
                add('required_qkv_indexer_k',prep,first=layer_start)
            add('required_indexer_cache_qw',pre[pre.start>=sc])
        elif not len(li) and np.isfinite(layer_start) and 'ring_history_gather' in found:
            add('required_qkv_indexer_k',between(layer_start,found['ring_history_gather'][0]))
        if np.isfinite(sf_start):
            trans=compute[(compute.start<sf_start)&(compute.start>=li_end)&
                          compute.name.str.contains('transpose',case=False)&
                          ~compute.name.str.contains('matmul',case=False)]
            if len(trans):add('sfa_q_transpose',trans.iloc[[-1]])
        after=compute[compute.start>=sf_end-1e-6]
        norms=after[after.stage.eq('AddRmsNorm')]
        moe_start=float(norms.iloc[0].start) if len(norms) else np.nan
        outputs=after[after.start<moe_start] if np.isfinite(moe_start) else after
        vs=outputs[outputs.stage.eq('BatchMatmulTranspose')]
        if len(vs)==1:
            add('v_up_projection',vs)
            os=outputs[(outputs.start>vs.iloc[0].start)&outputs.stage.isin(['QuantMatmul','Matmul'])]
            if len(os):
                o=os.iloc[0];add('o_matmul',os.iloc[[0]])
                qs=outputs[(outputs.start>vs.iloc[0].start)&(outputs.start<o.start)&
                           outputs.name.str.contains('ascendquant|dynamicquant',case=False,regex=True)]
                if len(qs):add('o_input_quant',qs.iloc[[-1]])
                comm=ld[(ld.start>=o.start)&(ld.start<moe_start)&
                        ld.stage.isin(['AllReduce','ReduceScatter','AllGather'])]
                add('output_communication',comm)
        if np.isfinite(moe_start):
            tail=ld[(ld.start>=moe_start)&~ld.stage.eq('CommunicationRuntime')]
            add('moe_tail',tail,
                'estimated_layer_end' if end_estimated else 'measured' if right_complete else 'partial',
                first=moe_start,last=layer_end if right_complete else None,
                evidence='MoE input norm -> layer boundary; partial if right boundary missing')
    else:
        phase='unsupported_split'
    layout='attention_deferred' if prep_location=='after_indexer' else 'original' if prep_location=='before_indexer' else 'unresolved'
    reference=float(layer_start) if np.isfinite(layer_start) else float(ld.start.min())
    timeline=[]
    for sid,label in STAGES:
        a,b,status,evidence=found.get(sid,(np.nan,np.nan,'not_identified',''))
        if sid in ('ring_history_gather','ring_kv_commit') and phase=='decode' and sid not in found:
            status='not_present_decode'
        part=ld.loc[list(members.get(sid,set()))]
        span=b-a if np.isfinite(a) else np.nan
        timeline.append(dict(step_id=step_id,model_layer=model_layer,stage_id=sid,stage_name=label,
            category='最近2K KV' if sid.startswith('ring_') else label,status=status,
            phase=phase,input_capacity=capacity,request_rows=requests,
            mapping_path=mapping_path,
            execution_layout=layout,attention_prepare_location=prep_location,
            compute_stream_status=compute_stream_status,compute_streams=compute_streams,
            module_streams=json.dumps([int(x) for x in part.stream.dropna().drop_duplicates()]),
            layer_wall_us=layer_end-layer_start if np.isfinite(layer_start) and right_complete else np.nan,
            layer_end_estimated=end_estimated,relative_origin='layer_start' if np.isfinite(layer_start) else 'fragment_start',
            relative_start_us=a-reference,relative_end_us=b-reference,span_us=span,
            complete_module_span_us=span if status=='measured' else np.nan,
            trace_start_us=a,trace_end_us=b,kernel_duration_sum_us=float(part.duration_us.sum()) if len(part) else np.nan,
            kernel_union_us=union_time(part) if len(part) else np.nan,kernel_count=len(part),
            source_rows=json.dumps(sorted(part.source_row.astype(int).tolist())),
            boundary_evidence=evidence,boundary_delta_to_next_us=np.nan))
    timeline.sort(key=lambda r:r['trace_start_us'] if np.isfinite(r['trace_start_us']) else np.inf)
    present=[r for r in timeline if np.isfinite(r['trace_start_us'])]
    for a,b in zip(present,present[1:]):a['boundary_delta_to_next_us']=b['trace_start_us']-a['trace_end_us']
    for i,r in enumerate(timeline,1):r['stage_order']=i
    info=dict(phase=phase,input_capacity=capacity,request_rows=requests,
              mapping_path=mapping_path,
              compute_stream_status=compute_stream_status,compute_streams=compute_streams,
              indexer_start_us=li_start,indexer_end_us=li_end,sfa_start_us=sf_start,sfa_end_us=sf_end,
              attention_prepare_location=prep_location,execution_layout=layout,
              ring_mapping_observed='ring_index_remap' in found)
    return info,timeline,members


def observed_layers(data):
    """Find local attention starts; retain clipped first/last modules.

    Fragment-to-fragment starts are observations, NOT proven 61-layer step
    boundaries. Absolute model ids remain unknown until the full analyzer runs.
    """
    norms=data[data.stage.eq('AddRmsNorm')].start.to_numpy()
    lis=data[data.stage.eq('LightningIndexer')].sort_values('start')
    starts=set()
    previous=-np.inf
    for row in lis.itertuples():
        candidates=norms[(norms>previous)&(norms<row.start)]
        if len(candidates):starts.add(float(candidates[-1]))
        previous=row.start
    # A final clipped prefill layer can have QKV/history copy but no LI yet.
    for row in data[data.stage.eq('QuantMatmul')].itertuples():
        s=shape_list(row.input_shapes)
        if len(s)>1 and s[0] and s[0][-1]==7168 and any(2112 in x for x in s[1:]):
            candidates=norms[norms<row.start]
            if len(candidates):starts.add(float(candidates[-1]))
    bounds=sorted(starts)
    if not bounds or (len(lis) and lis.iloc[0].start<bounds[0]):bounds.insert(0,float(data.start.min()))
    results=[]
    times=data.start.to_numpy()
    for i,start in enumerate(bounds):
        stop=bounds[i+1] if i+1<len(bounds) else float(data.end.max())+1e-5
        rows=data.iloc[np.searchsorted(times,start):np.searchsorted(times,stop)]
        if rows.empty:continue
        known=start in starts
        # Never call the last visible kernel the layer's true end.
        info,stages,members=analyze_layer(rows,start if known else np.nan,stop,i+1<len(bounds))
        info.update(observed_layer=i,model_layer=np.nan,layer_start_us=start if known else np.nan,
                    observation_start_us=start,observation_end_us=float(rows.end.max()),
                    next_attention_start_us=stop if i+1<len(bounds) else np.nan,
                    layer_start_known=known,right_boundary_observed=i+1<len(bounds),
                    boundary_status='next_attention_start_unverified_transition' if i+1<len(bounds) else 'right_clipped',
                    adjacent_attention_span_us=stop-start if known and i+1<len(bounds) else np.nan)
        for stage in stages:
            stage['observed_layer']=i
            # Local fragments cannot prove that the next start isn't layer0
            # of another step. Keep this distinct from full-model layer wall.
            stage['layer_wall_us']=np.nan
            if stage['stage_id']=='moe_tail' and stage['status']=='measured':
                stage['status']='next_attention_boundary_unverified'
                stage['complete_module_span_us']=np.nan
        results.append((info,stages,members,rows))
    return results


def export_observed(data, output_dir, source, origin):
    results=observed_layers(data)
    layers=pd.DataFrame([x[0] for x in results])
    stages=pd.DataFrame([s for x in results for s in x[1]])
    details=[]
    for info,_,members,rows in results:
        part=rows.copy();part['observed_layer']=info['observed_layer'];part['phase']=info['phase']
        part['module']='unresolved';part['module_match_count']=0
        for sid,ids in members.items():
            part.loc[list(ids),'module']=sid
            part.loc[list(ids),'module_match_count']+=1
        part.loc[part.module_match_count.gt(1),'module']='ambiguous'
        part.loc[part.stage.eq('CommunicationRuntime'),'module']='communication_runtime'
        details.append(part)
    directory=Path(output_dir)
    layers.to_csv(directory/'ring_observed_layers.csv',index=False,encoding='utf-8-sig')
    stages.to_csv(directory/'ring_observed_modules.csv',index=False,float_format='%.6f',encoding='utf-8-sig')
    detail=pd.concat(details,ignore_index=True) if details else pd.DataFrame()
    detail.drop(columns=['_start_dec'],errors='ignore').to_csv(directory/'ring_observed_kernels.csv',index=False,encoding='utf-8-sig')
    measured=stages[stages.status.eq('measured')]
    summary=measured.groupby(['phase','input_capacity','stage_id','stage_name'],dropna=False,as_index=False).agg(
        samples=('span_us','count'),span_us_mean=('span_us','mean'),span_us_min=('span_us','min'),
        span_us_max=('span_us','max'),kernel_duration_sum_us_mean=('kernel_duration_sum_us','mean'))
    summary.to_csv(directory/'ring_observed_module_summary.csv',index=False,encoding='utf-8-sig')
    (directory/'ring_analysis_manifest.json').write_text(json.dumps(dict(
        source=str(Path(source).resolve()),timestamp_origin_us=str(origin),
        schema='ring2k-unsplit-v3',timing='first kernel start to last kernel end; sums reported separately',
        observed_scope='local fragments; no absolute model layer or TPOT; clipped modules excluded from means',
        phases=layers.phase.value_counts().to_dict() if len(layers) else {}),ensure_ascii=False,indent=2))
    print('Ring observed layers:',len(layers),'; phases:',layers.phase.value_counts().to_dict() if len(layers) else {})
    print('Ring parser v3; compute-stream layouts:',
          layers.compute_stream_status.value_counts().to_dict() if len(layers) else {})
    return results
