"""Replay identical real layer-7 FA inputs across query/KV geometries."""
import json
from pathlib import Path
import torch,torch_npu

torch.npu.set_device(0)
root=Path('/home/w00848261/verl/.zero_diff/runs/fa-shape-replay-002');root.mkdir(parents=True,exist_ok=True)
mask=torch.triu(torch.ones((2048,2048),device='npu',dtype=torch.bool),1)
records=[]
def fa(q,k,v,scale):
    return torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),q.shape[1],'TND',atten_mask=mask,scale=scale,pre_tockens=2147483647,next_tockens=0,keep_prob=1.0,inner_precise=0,sparse_mode=3,actual_seq_qlen=(q.shape[0],),actual_seq_kvlen=(k.shape[0],))[0]
def cmp(a,b):
    return {'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
with torch.no_grad():
    for rank in range(8):
        p=torch.load(Path('/home/w00848261/verl/.zero_diff/runs/proxy8-attention-004/trace')/f'rank{rank}/fa.003.pt',weights_only=False,map_location='cpu')
        start=0
        for end,length in zip(p['q_ends'],p['kv_lengths']):
            if length==1040 and end-start==1040:break
            start=end
        assert end-start==1040
        q,k,v=(p[n][start:end].to('npu') for n in ('query','key','value'));reference=p['output'][start+1024].to('npu')
        record={'rank':rank,'replay_full':cmp(fa(q,k,v,p['scale'])[1024],reference),'decode':cmp(fa(q[1024:1025],k[:1025],v[:1025],p['scale'])[0],reference),'history_queries':cmp(fa(q[:1025],k[:1025],v[:1025],p['scale'])[1024],reference),'blocks':{}}
        for block in (16,32,64,128,256,512):
            padded_q=torch.zeros((block,*q.shape[1:]),dtype=q.dtype,device=q.device);padded_q[-1]=q[1024]
            left=cmp(fa(padded_q,k[:1025],v[:1025],p['scale'])[-1],reference)
            end=((1025+block-1)//block)*block;row=1024%block
            padded_q.zero_();padded_q[row]=q[1024]
            padded_k=torch.zeros((end,*k.shape[1:]),dtype=k.dtype,device=k.device);padded_v=torch.zeros_like(padded_k)
            padded_k[:1025]=k[:1025];padded_v[:1025]=v[:1025]
            right=cmp(fa(padded_q,padded_k,padded_v,p['scale'])[row],reference)
            record['blocks'][str(block)]={'past_query_padding':left,'future_kv_query_padding':right}
        record['causal_prefix_1024']=cmp(fa(q[:1024],k[:1024],v[:1024],p['scale']),p['output'][start:start+1024].to('npu'))
        records.append(record);print(json.dumps(record),flush=True)
(root/'result.json').write_text(json.dumps(records,indent=2))
