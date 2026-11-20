import json,traceback
from pathlib import Path
import megatron_adaptor
from fla_legacy_registration import register_missing_ops
lib,registered=register_missing_ops()
import torch
from mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule import flash_gated_delta_rule
root=Path('/home/w00848261/verl/.zero_diff/runs/native-replay-001')
root.mkdir(parents=True,exist_ok=True)
d=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
x=[d[n].npu() for n in ['q','k','v','g','beta']]
x[:2]=[t.repeat_interleave(2,dim=2) for t in x[:2]]
report={'scope':'native training chunk math prefix replay diagnostic'}
def compare(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    return {'mismatch':int((a!=b).sum()),'count':a.numel(),'max_abs':float((a.float()-b.float()).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
try:
    with torch.no_grad():
        expected,_=flash_gated_delta_rule(*x,scale=128**-0.5,output_final_state=True)
        rows=[]
        state=None
        for end in range(1,66):
            start=((end-1)//64)*64
            out,next_state=flash_gated_delta_rule(*(t[:,start:end].contiguous() for t in x),scale=128**-0.5,initial_state=state,output_final_state=True)
            rows.append(out[:,-1:])
            if end%64==0:
                state=next_state
        report['prefix_replay']=compare(torch.cat(rows,dim=1),expected)
        for length in [1,16,63,64,65]:
            args=[t[:,:length].repeat((1,2)+(1,)*(t.ndim-2)).contiguous() for t in x]
            out,_=flash_gated_delta_rule(*args,scale=128**-0.5,cu_seqlens=torch.tensor([0,length,2*length],dtype=torch.int64,device='npu'))
            single,_=flash_gated_delta_rule(*(t[:,:length].contiguous() for t in x),scale=128**-0.5)
            report[f'batch_{length}']=compare(single,out[:,:length])
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
