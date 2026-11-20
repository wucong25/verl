import json,traceback
from pathlib import Path
import torch,torch_npu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-replay-001')
root.mkdir(parents=True,exist_ok=True)
d=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
x={n:d[n].npu() for n in ['q','k','v','g','beta']}
report={'scope':'chunk-prefix replay feasibility diagnostic, not production decode'}
def run(start,end,state):
    z={n:t[:,start:end].contiguous() for n,t in x.items()}
    return chunk_gated_delta_rule_fwd(*(z[n].transpose(1,2).contiguous() for n in ['q','k','v']),z['g'],z['beta'],initial_state=state,output_final_state=True,cu_seqlens=[0,end-start],chunk_indices=[0,0],scale=128**-0.5,layout='BNSD')
try:
    state=torch.zeros(1,4,128,128,device='npu',dtype=torch.float32)
    rows=[]
    for end in range(1,66):
        start=((end-1)//64)*64
        o,s=run(start,end,state)
        rows.append(o[:,-1:])
        if end%64==0:
            state=s
    out=torch.cat(rows,dim=1).cpu()
    expected=d['prefill']
    report.update(count=out.numel(),mismatch=int((out!=expected).sum()),max_abs=float((out.float()-expected.float()).abs().max()),finite=bool(torch.isfinite(out).all()))
    torch.save(out,root/'output.pt')
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
