import json,traceback
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
root=Path('/home/w00848261/verl/.zero_diff/runs/residual-actual-001');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(9)
 rec=torch.load('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-011/trace/language_model.model.layers.0.post_attention_layernorm.016.pt',weights_only=True)
 x,residual=(t.npu() for t in rec['inputs'])
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map'];key='model.language_model.layers.0.post_attention_layernorm.weight'
 with safe_open(str(ckpt/idx[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key).npu()
 for label,gamma in [('native_fp32',1+w.float()),('bf16',1+w)]:
  norm=torch_npu.npu_rms_norm(x+residual,gamma,1e-6)[0]
  report[label]=cmp(norm,rec['output'][0])
 report['residual']=cmp(x+residual,rec['output'][1])
 report['gamma_cast']=cmp(1+w.float(),(1+w).float())
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
