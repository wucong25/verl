"""Capture layer 7 module boundaries and actual FA arguments, all eight ranks."""
def install_trace(model):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    import vllm_ascend.ops.training_attention as attention
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-attention-003/trace')/f'rank{dist.get_rank()}'
    root.mkdir(parents=True,exist_ok=True)
    active=[False];handles=[];names=[]
    def tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,(tuple,list)):return tuple(tree(v) for v in value)
        if isinstance(value,dict):return {k:tree(v) for k,v in value.items()}
        if value is None or isinstance(value,(str,int,float,bool)):return value
        return str(type(value))
    def make_hook(name):
        counter=[0]
        def hook(module,args,kwargs,output):
            step=counter[0];counter[0]+=1
            torch.save({'inputs':tree(args),'kwargs':tree(kwargs),'output':tree(output)},root/f'{name}.{step:03d}.pt')
        return hook
    for name,module in model.named_modules():
        if '.layers.7.' in name and ('.self_attn' in name or name.endswith(('input_layernorm','post_attention_layernorm','mlp'))):
            handles.append(module.register_forward_hook(make_hook(name),with_kwargs=True));names.append(name)
        if name.endswith('.layers.7'):
            handles.append(module.register_forward_pre_hook(lambda *args:active.__setitem__(0,True)))
            handles.append(module.register_forward_hook(lambda *args:active.__setitem__(0,False)))
    original=attention.training_causal_attention
    counter=[0]
    def capture(query,key,value,key_cache,value_cache,metadata,scale,mask,layout):
        result=original(query,key,value,key_cache,value_cache,metadata,scale,mask,layout)
        if active[0]:
            step=counter[0];counter[0]+=1
            record={'query':tree(query),'key':tree(key),'value':tree(value),'output':tree(result),'scale':scale,'q_ends':metadata.actual_seq_lengths_q,'kv_lengths':metadata.seq_lens_list,'state':metadata.attn_state.name,'cache_layout':layout,'cache_kv':[]}
            if metadata.attn_state.name!='PrefillNoCache':
                hnd=layout in ('LBHNC','HND');bs=key_cache.shape[2 if hnd else 1]
                for request,length in enumerate(metadata.seq_lens_list):
                    indices=metadata.block_tables[request,:(length+bs-1)//bs].long()
                    k=key_cache.index_select(0,indices);v=value_cache.index_select(0,indices)
                    if hnd:k,v=k.transpose(1,2),v.transpose(1,2)
                    k=k.reshape(-1,k.shape[-2],k.shape[-1])[:length];v=v.reshape(-1,v.shape[-2],v.shape[-1])[:length]
                    record['cache_kv'].append((tree(k),tree(v)))
            torch.save(record,root/f'fa.{step:03d}.pt')
        return result
    attention.training_causal_attention=capture
    model._zero_diff_trace_handles=handles
    return names
