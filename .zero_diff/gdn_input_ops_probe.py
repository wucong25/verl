import json,traceback
from pathlib import Path
import megatron_adaptor
import torch
from safetensors import safe_open
from fla.modules.l2norm import l2norm
from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-input-001')
root.mkdir(parents=True,exist_ok=True)
report={'scope':'GDN normalization/gating diagnostic, precomputed random projection outputs'}
def compare(a,b):
    return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'dtype':str(a.dtype)}
try:
    torch.manual_seed(42)
    x=torch.randn(1,65,4,128).bfloat16().npu()
    with torch.no_grad():
        ref=l2norm(x)
        report['l2norm']=compare(ref,l2norm_fwd(x))
    base=Path('/mnt/weight/Qwen3.5-35B-A3B')
    index=json.loads((base/'model.safetensors.index.json').read_text())['weight_map']
    weights={}
    for suffix in ['A_log','dt_bias']:
        key=next(k for k in index if '.layers.0.linear_attn.' in k and k.endswith('.'+suffix))
        with safe_open(str(base/index[key]),framework='pt',device='cpu') as f:
            weights[suffix]=f.get_tensor(key)[:4].npu()
        report[suffix+'_dtype']=str(weights[suffix].dtype)
    a=torch.randn(65,4).bfloat16().npu()
    b=torch.randn(65,4).bfloat16().npu()
    g=-weights['A_log'].exp()*torch.nn.functional.softplus(a.float()+weights['dt_bias'])
    beta=b.sigmoid()
    ig,ib=fused_gdn_gating_patch(weights['A_log'],a,b,weights['dt_bias'])
    report['gating_g']=compare(g,ig.squeeze(0))
    report['gating_beta']=compare(beta,ib.squeeze(0))
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
