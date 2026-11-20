import json,traceback
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
root=Path('/home/w00848261/verl/.zero_diff/runs/norm-actual-001');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(8)
 infer=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-008/trace')
 d=torch.load(infer/'language_model.model.layers.0.linear_attn.norm.016.pt',weights_only=True)
 x,z=[a.npu() for a in d['inputs']]
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');index=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
 name='model.language_model.layers.0.linear_attn.norm.weight'
 with safe_open(str(ckpt/index[name]),framework='pt',device='cpu') as f:w=f.get_tensor(name).npu()
 with torch.no_grad():
  for dtype in (torch.bfloat16,torch.float32):
   y,_=torch_npu.npu_rms_norm(x,w.to(dtype),1e-6)
   y=(y*torch.nn.functional.silu(z.float())).to(x.dtype)
   report[str(dtype)]=cmp(y,d['output'])
   torch.save(y.cpu(),root/('fp32_weight.pt' if dtype==torch.float32 else 'bf16_weight.pt'))
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
