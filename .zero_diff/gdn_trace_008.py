def install_trace(model):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import vllm_ascend.ops.gdn as gdn
    import vllm_ascend.ops.gdn_batch_invariant as math
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-gdn-008/trace')/f'rank{dist.get_rank()}'
    root.mkdir(parents=True,exist_ok=True)
    active=[False];counts={};handles=[];names=[]
    def tree(v):
        if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
        if isinstance(v,(tuple,list)):return tuple(tree(x) for x in v)
        if isinstance(v,dict):return {k:tree(x) for k,x in v.items()}
        return v if isinstance(v,(type(None),int,float,str,bool)) else str(type(v))
    def save(name,record):
        step=counts.get(name,0);counts[name]=step+1
        torch.save(record,root/f'{name}.{step:03d}.pt')
    def hook(name):
        def fn(module,args,kwargs,out):save(name,{'args':tree(args),'kwargs':tree(kwargs),'output':tree(out)})
        return fn
    for name,module in model.named_modules():
        if name.endswith('.layers.0'):
            handles.append(module.register_forward_pre_hook(lambda *args:active.__setitem__(0,True)))
            handles.append(module.register_forward_hook(lambda *args:active.__setitem__(0,False)))
        if '.layers.0.' in name and ('linear_attn' in name or name.endswith(('input_layernorm','post_attention_layernorm','mlp'))):
            names.append(name);handles.append(module.register_forward_hook(hook(name),with_kwargs=True))
    def wrap(mod,name,conv=False):
        original=getattr(mod,name)
        def run(*args,**kwargs):
            record=None
            if active[0]:
                record={'args':tree(args),'kwargs':tree({k:v for k,v in kwargs.items() if k!='conv_states'})}
            out=original(*args,**kwargs)
            if record is not None:record['output']=tree(out);save(name,record)
            return out
        setattr(mod,name,run)
    wrap(gdn,'causal_conv1d_fn',True)
    wrap(math,'chunk_gdn_training_math')
    model._zero_diff_trace_handles=handles
    return names
