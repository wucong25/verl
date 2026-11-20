import json,traceback
from pathlib import Path
import megatron_adaptor
import torch
from verl.utils.kernel.npu.fla_compat import register_fla_legacy_ops
lib,names=register_fla_legacy_ops()
from mindspeed_bridge.features.megatron.common.fla_npu_causal_conv1d import fla_npu_causal_conv1d
from fla_npu.ops.ascendc import causal_conv1d_fn,causal_conv1d_update
root=Path('/home/w00848261/verl/.zero_diff/runs/conv-001')
root.mkdir(parents=True,exist_ok=True)
report={'scope':'native training conv vs inference prefill/decode','device':8}
def compare(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
try:
    torch.npu.set_device(8)
    torch.manual_seed(42)
    x=torch.randn(1,65,1024).bfloat16().npu().requires_grad_(True)
    w=torch.randn(1024,4).bfloat16().npu().requires_grad_(True)
    y,_=fla_npu_causal_conv1d(x,w,activation='silu')
    state=torch.zeros(2,3,1024,device='npu',dtype=torch.bfloat16)
    with torch.no_grad():
        pre=causal_conv1d_fn(x[0].contiguous(),w.T.contiguous(),None,conv_states=state,query_start_loc_cpu=[0,65],cache_indices_cpu=[1],has_initial_state_cpu=[False],activation='silu')
        report['train_prefill']=compare(y[0],pre)
        state.zero_(); rows=[]
        for t in range(65):
            out=causal_conv1d_update(x[0,t:t+1].contiguous(),state,w.T.contiguous(),activation='silu',conv_state_indices_cpu=[1],query_start_loc_cpu=[0,1],max_query_len=1)
            rows.append(out)
        report['prefill_decode']=compare(pre,torch.cat(rows,dim=0))
    y.float().square().mean().backward()
    report['gradients_finite']=bool(torch.isfinite(x.grad).all() and torch.isfinite(w.grad).all())
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
