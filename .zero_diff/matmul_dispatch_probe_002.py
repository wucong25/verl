import json,traceback
from pathlib import Path
import torch,torch_npu
from verl.utils.kernel.npu.batch_invariant import enable_batch_invariant_matmul
root=Path('/home/w00848261/verl/.zero_diff/runs/matmul-dispatch-002');root.mkdir(exist_ok=True)
report={}
try:
 torch.npu.set_device(8);torch.manual_seed(42)
 a=torch.randn(65,256,dtype=torch.bfloat16,device='npu',requires_grad=True)
 b=torch.randn(256,128,dtype=torch.bfloat16,device='npu',requires_grad=True)
 dy=torch.randn(65,128,dtype=torch.bfloat16,device='npu')
 dx_ref=(dy.cpu().double()@b.detach().cpu().double().t()).float()
 dw_ref=(a.detach().cpu().double().t()@dy.cpu().double()).float()
 lib=enable_batch_invariant_matmul()
 o=torch.matmul(a,b);o.backward(dy)
 for name,actual,ref in [('dx',a.grad,dx_ref),('dw',b.grad,dw_ref)]:
  actual=actual.cpu().float();rel=float(torch.linalg.vector_norm(actual-ref)/torch.linalg.vector_norm(ref));report[name]={'relative_l2':rel,'finite':bool(actual.isfinite().all())};assert rel<0.01
 with torch.no_grad():
  from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
  expected=linear_persistent(a,b.t())
  report['forward_equal_rollout']=bool(torch.equal(o,expected));assert report['forward_equal_rollout']
  aa=torch.randn(3,1,16,1,device='npu');bb=torch.arange(144,device='npu',dtype=torch.float32).reshape(1,1,1,144)
  report['mrope_outer']=bool(torch.equal(torch.matmul(aa,bb),aa*bb));assert report['mrope_outer']
  report['batch_invariant']=bool(torch.equal(o,torch.cat([torch.matmul(t,b) for t in a.split([1,63,1])])));assert report['batch_invariant']
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
