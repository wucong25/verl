"""Independent FP64 recurrence diagnostic; not same-rounding backward acceptance."""
import json,traceback
from pathlib import Path
import megatron_adaptor
from fla_legacy_registration import register_missing_ops
lib,registered=register_missing_ops()
import torch
from mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule import flash_gated_delta_rule
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-grad-001')
root.mkdir(parents=True,exist_ok=True)
report={'scope':'independent mathematical recurrence; internal chunk rounding not modeled','relative_l2_limit':0.03,'max_abs_limit':0.005}
(root/'contract.json').write_text(json.dumps(report,indent=2))
torch.set_num_threads(4)
d=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
x=[d[n] for n in ['q','k','v','g','beta']]
x[:2]=[t.repeat_interleave(2,dim=2) for t in x[:2]]
try:
    torch.manual_seed(123)
    upstream=(torch.randn_like(x[2].float())/x[2].numel()**0.5).bfloat16()
    actual=[t.npu().detach().requires_grad_(True) for t in x]
    out,_=flash_gated_delta_rule(*actual,scale=128**-0.5)
    out.backward(upstream.npu())
    ref=[t.double().requires_grad_(True) for t in x]
    q,k,v,g,beta=ref
    state=torch.zeros(1,4,128,128,dtype=torch.float64)
    rows=[]
    for t in range(q.shape[1]):
        state=state*g[:,t].exp()[...,None,None]
        delta=(v[:,t]-(state*k[:,t,...,None]).sum(-2))*beta[:,t,...,None]
        state=state+k[:,t,...,None]*delta[...,None,:]
        rows.append(((state*q[:,t,...,None]).sum(-2)*(128**-0.5)).unsqueeze(1))
    expected=torch.cat(rows,dim=1)
    expected.backward(upstream.double())
    for name,a,b in zip(['q','k','v','g','beta'],actual,ref):
        aa=a.grad.cpu().double(); bb=b.grad
        diff=aa-bb
        rel=float(diff.norm()/bb.norm().clamp_min(1e-30))
        maximum=float(diff.abs().max())
        report[name]={'relative_l2':rel,'max_abs':maximum,'finite':bool(torch.isfinite(aa).all()),'within_diagnostic_limits':rel<=0.03 and maximum<=0.005}
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
