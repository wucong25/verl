import json,traceback
from pathlib import Path
import torch,torch_npu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-chunk-001')
root.mkdir(parents=True,exist_ok=True)
d=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
x={n:d[n].npu() for n in ['q','k','v','g','beta']}
report={'scope':'FLA operator prefix/batch invariance only'}
def run(length,copies):
    inp={n:t[:,:length].repeat((1,copies)+(1,)*(t.ndim-2)).contiguous() for n,t in x.items()}
    meta=[j for i in range(copies) for c in range((length+63)//64) for j in [i,c]]
    out=chunk_gated_delta_rule_fwd(*(inp[n].transpose(1,2).contiguous() for n in ['q','k','v']),inp['g'],inp['beta'],initial_state=torch.zeros(copies,4,128,128,device='npu',dtype=torch.float32),output_final_state=True,cu_seqlens=[i*length for i in range(copies+1)],chunk_indices=meta,scale=128**-0.5,layout='BNSD')
    return out[0].cpu()
def compare(a,b):
    return {'mismatch':int((a!=b).sum()),'count':a.numel(),'max_abs':float((a.float()-b.float()).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
try:
    for length in [1,2,16,63,64,65]:
        a=run(length,1)
        report[f'prefix_{length}_vs_65']=compare(a,d['prefill'][:,:length])
        b=run(length,2)
        report[f'batch_{length}']=compare(a,b[:,:length])
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
