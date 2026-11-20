import json, time, traceback
from pathlib import Path
import torch
import torch_npu
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd, recurrent_gated_delta_rule
root = Path('/home/w00848261/verl/.zero_diff/runs/gdn-002')
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.npu.set_device(0)
T, HK, HV, D = 65, 2, 4, 128
q = torch.nn.functional.normalize(torch.randn(1,T,HK,D), dim=-1).bfloat16().npu()
k = torch.nn.functional.normalize(torch.randn(1,T,HK,D), dim=-1).bfloat16().npu()
v = torch.randn(1,T,HV,D).bfloat16().npu()
g = (-torch.rand(1,T,HV)).npu()
beta = torch.rand(1,T,HV).bfloat16().npu()
state = torch.zeros(1,HV,D,D,device='npu',dtype=torch.float32)
report = {'seed':42,'shape':[T,HK,HV,D],'torch':torch.__version__,'scope':'operator diagnostic only; externally normalized q/k; not model acceptance'}
def compare(a,b):
    a,b=a.cpu(),b.cpu()
    assert a.shape==b.shape,(a.shape,b.shape)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'mismatch':int((a!=b).sum()),'count':a.numel(),'max_abs':float((a.float()-b.float()).abs().max())}
def prefill():
    return chunk_gated_delta_rule_fwd(q.transpose(1,2).contiguous(),k.transpose(1,2).contiguous(),v.transpose(1,2).contiguous(),g,beta,initial_state=state.clone(),output_final_state=True,chunk_size=64,cu_seqlens=[0,T],chunk_indices=[0,0,0,1],scale=D**-0.5,layout='BNSD',state_v_first=False)
try:
    started=time.time()
    a=prefill(); torch.npu.synchronize()
    report['prefill_seconds']=time.time()-started
    b=prefill()
    report['prefill_repeat']=compare(a[0],b[0])
    ss=state.transpose(-1,-2).contiguous().clone()
    outputs=[]
    for t in range(T):
        out=recurrent_gated_delta_rule(q[0,t:t+1].contiguous(),k[0,t:t+1].contiguous(),v[0,t:t+1].contiguous(),ss,beta=beta[0,t:t+1].contiguous(),g=g[0,t:t+1].contiguous(),scale=D**-0.5,actual_seq_lengths=torch.tensor([0,1],device='npu',dtype=torch.int32),ssm_state_indices=torch.tensor([0],device='npu',dtype=torch.int32))
        outputs.append(out)
    rec=torch.cat(outputs,dim=0).unsqueeze(0)
    report['prefill_decode']=compare(a[0],rec)
    report['state']=compare(a[1],ss.transpose(-1,-2).contiguous())
    torch.save({'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),'g':g.cpu(),'beta':beta.cpu(),'prefill':a[0].cpu(),'decode':rec.cpu()},root/'replay.pt')
except Exception:
    report['error']=traceback.format_exc()
finally:
    (root/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
