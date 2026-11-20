from pathlib import Path
import torch,torch_npu,json
T=Path('.zero_diff/runs/proxy-megatron-011/trace');p='chunk0.module.module.language_model.output_layer.000.pt'
torch.npu.set_device(9)
shards=[torch.load(T/f'rank{r}'/p,weights_only=False)['output'][0].squeeze(1).float().npu() for r in (8,9)]
logits=torch.cat(shards,-1);sample=json.load(open('.zero_diff/runs/proxy-vllm-013/result.json'));labels=torch.tensor(sample['response'],dtype=torch.long,device='npu');rows=torch.arange(127,143,device='npu')
native=torch.load('.zero_diff/runs/proxy-megatron-011/recompute.pt',weights_only=False)['native_recompute'][0]
infer=torch.tensor(json.load(open('.zero_diff/runs/proxy-vllm-024/result.json'))['fixed_reference_logprobs'])
print('shape',logits.shape,flush=True)
shifted=logits-logits.max(-1,keepdim=True).values
lse=(shifted[:,:logits.shape[-1]//2].exp().sum(-1)+shifted[:,logits.shape[-1]//2:].exp().sum(-1)).log()
canonical=-(lse[:,None]-shifted)
def cmp(a,b):
 a,b=a.cpu(),b.cpu();return {'mismatch':int((a!=b).sum()),'max':float((a-b).abs().max())}
print(json.dumps({'canonical_native':cmp(canonical[rows,labels],native),'torch_logsoftmax_infer':cmp(logits.log_softmax(-1)[rows,labels],infer),'torch_logsoftmax_native':cmp(logits.log_softmax(-1)[rows,labels],native),'canonical_batch':cmp(canonical[rows,labels],torch.cat([-( (v[:,:v.shape[-1]//2].exp().sum(-1)+v[:,v.shape[-1]//2:].exp().sum(-1)).log()[:,None]-v) for v in shifted.split(1)])[rows,labels])}),flush=True)
