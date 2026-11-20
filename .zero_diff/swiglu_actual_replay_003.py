import json,traceback
from pathlib import Path
import torch,torch_npu
root=Path('/home/w00848261/verl/.zero_diff/runs/swiglu-actual-003');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(9)
 tr=Path('/home/w00848261/verl/.zero_diff/runs/proxy-megatron-011/trace/rank0');pre='chunk0.module.module.language_model.decoder.layers.0.mlp.'
 for module in ('shared_experts','experts'):
  a=torch.load(tr/(pre+module+'.linear_fc1.000.pt'),weights_only=True)
  b=torch.load(tr/(pre+module+'.linear_fc2.000.pt'),weights_only=True)
  x=a['output'][0].npu();expected=b['inputs'][0]
  gate,up=x.chunk(2,-1)
  if module=='experts':
   call=torch.load(tr/(pre+module+'.000.pt'),weights_only=True)
   probs=call['inputs'][2].reshape(-1,1).npu()
   report['weighted']={
    'split_bf16_then_prob':cmp(((torch.nn.functional.silu(gate)*up)*probs).to(x.dtype),expected),
    'fused_bf16_then_prob':cmp((torch_npu.npu_swiglu(x)*probs).to(x.dtype),expected),
    'fp32_all':cmp((torch.nn.functional.silu(gate.float())*up.float()*probs).to(x.dtype),expected),
   }
  report[module]={'shape':list(x.shape),'split_bf16':cmp(torch.nn.functional.silu(gate)*up,expected),'fused':cmp(torch_npu.npu_swiglu(x),expected),'fp32':cmp((torch.nn.functional.silu(gate.float())*up.float()).to(x.dtype),expected)}
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
