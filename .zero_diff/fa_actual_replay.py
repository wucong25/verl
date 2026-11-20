from pathlib import Path
import torch,torch_npu,json,traceback
torch.npu.set_device(9)
T=Path('.zero_diff/runs/proxy-megatron-011/trace');p='chunk0.module.module.language_model.decoder.layers.1.self_attention.core_attention.000.pt'
mask=torch.ones(2048,2048,dtype=torch.bool,device='npu').triu(1)
def cmp(a,b):
 a,b=a.cpu(),b.cpu();return {'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max()),'n':a.numel()}
def fa(q,k,v,precise=0):
 return torch_npu.npu_fusion_attention(q.contiguous(),k.contiguous(),v.contiguous(),q.shape[1],'TND',atten_mask=mask,scale=0.0625,pre_tockens=2147483647,next_tockens=0,keep_prob=1.0,inner_precise=precise,sparse_mode=3,actual_seq_qlen=(q.shape[0],),actual_seq_kvlen=(k.shape[0],))[0]
report={}
try:
 for r in (8,9):
  d=torch.load(T/f'rank{r}'/p,weights_only=False);q,k,v=[x.npu() for x in d['inputs'][:3]];expected=d['output']
  for precise in (0,1,2):
   a=fa(q,k,v,precise);report[f'{r}_fa{precise}_native']=cmp(a,expected)
  a=fa(q,k,v)
  report[f'{r}_headsplit']=cmp(torch.cat([fa(q[:,i:i+2],k,v) for i in range(0,8,2)],1),a)
  report[f'{r}_prefix128']=cmp(fa(q[:128],k[:128],v[:128]),a[:128])
  for pos in (0,63,127,128,143):report[f'{r}_decode{pos}']=cmp(fa(q[pos:pos+1],k[:pos+1],v[:pos+1]),a[pos:pos+1])
except Exception:report['error']=traceback.format_exc()
print(json.dumps(report,indent=2),flush=True)
