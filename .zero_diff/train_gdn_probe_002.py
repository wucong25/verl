import json, traceback
from pathlib import Path
import megatron_adaptor
import fla_npu
fla_npu.load_legacy_torch_ops()
import torch
from mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule import flash_gated_delta_rule
root=Path('/home/w00848261/verl/.zero_diff/runs/train-gdn-002')
root.mkdir(parents=True,exist_ok=True)
data=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
report={'scope':'native training GDN op diagnostic, not actor acceptance'}
def compare(a,b):
    a,b=a.detach().cpu(),b.detach().cpu()
    assert a.shape==b.shape
    return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
try:
    args=[data[n].npu() for n in ['q','k','v','g','beta']]
    args[0]=args[0].repeat_interleave(2,dim=2)
    args[1]=args[1].repeat_interleave(2,dim=2)
    with torch.no_grad():
        out,_=flash_gated_delta_rule(*args,scale=128**-0.5,output_final_state=True,use_qk_l2norm_in_kernel=False)
    report['training_vs_prefill']=compare(out,data['prefill'])
    report['training_vs_decode']=compare(out,data['decode'])
    args=[x.detach().requires_grad_(True) for x in args]
    out_grad,_=flash_gated_delta_rule(*args,scale=128**-0.5,use_qk_l2norm_in_kernel=False)
    report['no_grad_vs_grad']=compare(out,out_grad)
    out_grad.float().square().mean().backward()
    report['gradients']=[{'finite':bool(torch.isfinite(x.grad).all()),'nonzero':int(torch.count_nonzero(x.grad))} for x in args]
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
