from pathlib import Path
from types import SimpleNamespace
import torch,torch_npu,json
from vllm_ascend.ops.training_attention import training_causal_attention

torch.npu.set_device(0)
source=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-attention-004/trace')
root=Path('/home/w00848261/verl/.zero_diff/runs/fa-fix-actual-003');root.mkdir(exist_ok=True,parents=True)
mask=torch.ones(2048,2048,dtype=torch.bool,device='npu').triu(1)
records=[]
for rank in range(8):
    d=torch.load(source/f'rank{rank}/fa.002.pt',weights_only=False,map_location='cpu')
    p=torch.load(source/f'rank{rank}/fa.003.pt',weights_only=False,map_location='cpu')
    ds=0
    for di,(de,length) in enumerate(zip(d['q_ends'],d['kv_lengths'])):
        if length==1025 and de-ds==1:break
        ds=de
    ps=0
    for pe,length in zip(p['q_ends'],p['kv_lengths']):
        if length==1040 and pe-ps==1040:break
        ps=pe
    query=d['query'][ds:de].to('npu');k,v=(x.to('npu') for x in d['cache_kv'][di])
    kc=torch.zeros(33,32,1,256,device='npu',dtype=query.dtype);vc=torch.zeros_like(kc)
    kc.flatten(0,1)[:1025]=k;vc.flatten(0,1)[:1025]=v
    meta=SimpleNamespace(actual_seq_lengths_q=[1],seq_lens_list=[1025],attn_state=SimpleNamespace(name='DecodeOnly'),block_tables=torch.arange(33,device='npu')[None,:])
    actual=training_causal_attention(query,None,None,kc,vc,meta,d['scale'],mask,'LBNHC').cpu()
    expected=p['output'][ps+1024:ps+1025]
    record={'rank':rank,'count':actual.numel(),'mismatch':int((actual!=expected).sum()),'max_abs':float((actual.float()-expected.float()).abs().max())}
    records.append(record)
    assert torch.equal(actual,expected),record
(root/'result.json').write_text(json.dumps({'passed':True,'comparisons':records},indent=2));print(records,flush=True)
