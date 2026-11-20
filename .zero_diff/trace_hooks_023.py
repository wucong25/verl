"""Temporary eager forward hooks for first-divergence localization."""
def install_trace(model):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-023/trace')/f'rank{dist.get_rank()}'
    root.mkdir(parents=True,exist_ok=True)
    selected={'embed_tokens','input_layernorm','post_attention_layernorm','linear_attn','self_attn','mlp','in_proj_qkvz','in_proj_qkv','in_proj_z','in_proj_ba','out_proj','norm','gate','experts','qkv_proj','o_proj','gate_up_proj','down_proj','act_fn','expert_gate','q_norm','k_norm','rotary_emb','attn'}
    names=[]; handles=[]
    def cpu_tree(value):
        if isinstance(value,torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value,(tuple,list)):
            return tuple(cpu_tree(t) for t in value)
        if value is None or isinstance(value,(int,float,str,bool)):
            return value
        return str(type(value))
    def make_hook(name):
        counter=[0]
        def hook(module,args,output):
            step=counter[0]; counter[0]+=1
            if step == 0:
                for n,p in module.named_parameters(recurse=False):
                    if p.numel()<=4096:torch.save(cpu_tree(p),root/f'{name}.{n}.pt')
            torch.save({'inputs':cpu_tree(args),'output':cpu_tree(output)},root/f'{name}.{step:03d}.pt')
        return hook
    for name,module in model.named_modules():
        if '.layers.' in name and name.rsplit('.',1)[-1] in selected:
            names.append(name)
            handles.append(module.register_forward_hook(make_hook(name)))
    def route_hook(original, name):
        counter=[0]
        def call(*args, **kwargs):
            step=counter[0];counter[0]+=1
            result=original(*args,**kwargs)
            torch.save({'output':cpu_tree(result),'logits':cpu_tree(kwargs.get('router_logits'))},root/f'{name}.routing.{step:03d}.pt')
            return result
        return call
    for name,module in model.named_modules():
        if '.layers.0.' not in name:continue
        router=getattr(module,'router',None)
        if router is not None and hasattr(router,'_select_experts'):
            router._select_experts=route_hook(router._select_experts,name)
            names.append(name+'.routing')
    seen=set()
    def capture_mlp(original, method, name, quant):
        counter=[0]
        def call(mlp_input,*args,**kwargs):
            step=counter[0];counter[0]+=1
            record={key:cpu_tree(getattr(mlp_input,key,None)) for key in ('hidden_states','group_list','group_list_type','need_trans','topk_scales')}
            record['args']=cpu_tree(args)
            if method=='apply_gmm1' and step==0:
                torch.save(cpu_tree(quant.get_mlp_weights(mlp_input.layer)),root/f'{name}.weights.pt')
            result=original(mlp_input,*args,**kwargs)
            record['output']=cpu_tree(result)
            torch.save(record,root/f'{name}.{method}.{step:03d}.pt')
            return result
        return call
    for name,module in model.named_modules():
        if '.layers.0.' not in name:continue
        quant=getattr(module,'quant_method',None)
        if quant is not None and hasattr(quant,'apply_gmm1') and id(quant) not in seen:
            seen.add(id(quant))
            for method in ('apply_gmm1','apply_act_quant','apply_gmm2'):
                setattr(quant,method,capture_mlp(getattr(quant,method),method,name,quant))
            names.append(name+'.gmm_trace')
    model._zero_diff_trace_handles=handles
    return names
