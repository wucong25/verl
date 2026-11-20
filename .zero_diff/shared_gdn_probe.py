import json,traceback
from pathlib import Path
import megatron_adaptor
from fla_legacy_registration import register_missing_ops
lib,registered=register_missing_ops()
import torch
from mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule import flash_gated_delta_rule
from vllm_ascend.ops.gdn_batch_invariant import chunk_gdn_training_math
root=Path('/home/w00848261/verl/.zero_diff/runs/shared-gdn-001')
root.mkdir(parents=True,exist_ok=True)
d=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-002/replay.pt',map_location='cpu',weights_only=True)
x=[d[n].npu() for n in ['q','k','v','g','beta']]
report={'scope':'shared inference primitive vs native training GDN; no model integration'}
try:
    with torch.no_grad():
        native=[x[0].repeat_interleave(2,dim=2),x[1].repeat_interleave(2,dim=2),*x[2:]]
        for length in [1,16,63,64,65]:
            for packed in [False,True]:
                meta=torch.tensor([0,length,2*length],device='npu',dtype=torch.int64) if packed else None
                args=[t[:,:length].repeat((1,2 if packed else 1)+(1,)*(t.ndim-2)).contiguous() for t in x]
                refs=[t[:,:length].repeat((1,2 if packed else 1)+(1,)*(t.ndim-2)).contiguous() for t in native]
                out,st=chunk_gdn_training_math(*args,scale=128**-0.5,cu_seqlens=meta)
                ref,sr=flash_gated_delta_rule(*refs,scale=128**-0.5,cu_seqlens=meta,output_final_state=True)
                report[f'{length}_{packed}']={'count':out.numel(),'mismatch':int((out!=ref).sum()),'max_abs':float((out.float()-ref.float()).abs().max()),'state_equal':bool(torch.equal(st,sr)),'finite':bool(torch.isfinite(out).all())}
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
