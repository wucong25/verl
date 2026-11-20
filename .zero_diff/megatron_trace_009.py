"""Diagnostic-only native engine hooks; no replacement forward implementation."""
from pathlib import Path
import json
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from verl.single_controller.base.decorator import register, Dispatch
from verl.workers.engine_workers_tinker import TinkerTrainingWorker


class DiagnosticWorker(TinkerTrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def install_trace(self, root):
        rank=dist.get_rank()
        if mpu.get_data_parallel_rank()!=0:
            return {'rank':rank,'traced':False}
        dest=Path(root)/f'rank{rank}'
        dest.mkdir(parents=True,exist_ok=True)
        def cpu(x):
            if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
            if isinstance(x,(tuple,list)):return tuple(cpu(t) for t in x)
            if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
            if x is None or isinstance(x,(bool,str,int,float)):return x
            return str(type(x))
        def make_hooks(name):
            calls=[0]; saved=[]
            def pre(module,args,kwargs):
                if calls[0] == 0:
                    for n,p in module.named_parameters(recurse=False):
                        if p.numel()<=4096 or n == 'layer_norm_weight':torch.save(p.detach().cpu().clone(),dest/f'{name}.{n}.pt')
                saved.append((calls[0],cpu(args),cpu(kwargs)));calls[0]+=1
            def post(module,args,kwargs,out):
                step,inputs,kw=saved.pop()
                torch.save({'inputs':inputs,'kwargs':kw,'output':cpu(out),'grad_enabled':torch.is_grad_enabled()},dest/f'{name}.{step:03d}.pt')
            return pre,post
        metadata=[]; handles=[]
        for i,model in enumerate(self.engine.module):
            for name,module in model.named_modules():
                if 'vision' in name or 'local_experts.' in name:continue
                full=f'chunk{i}.{name}'
                metadata.append({'name':full,'type':str(type(module)),'parameters':{n:list(p.shape) for n,p in module.named_parameters(recurse=False)}})
                if hasattr(module, 'gated_delta_rule'):
                    import inspect
                    operation=module.gated_delta_rule
                    f=getattr(module,'_gdn_l2norm',module._prepare_qkv_for_gated_delta_rule)
                    metadata[-1]['l2norm_or_prepare']={'module':getattr(f,'__module__',None),'name':getattr(f,'__qualname__',None),'source':inspect.getsourcefile(f) if f is not None else None}
                    metadata[-1]['gated_delta_rule']={'module':getattr(operation,'__module__',None),'name':getattr(operation,'__qualname__',None),'source':inspect.getsourcefile(operation)}
                    def capture_gdn(original, name):
                        counter=[0]
                        def call(*args,**kwargs):
                            step=counter[0];counter[0]+=1
                            inputs,kw=cpu(args),cpu(kwargs)
                            result=original(*args,**kwargs)
                            torch.save({'inputs':inputs,'kwargs':kw,'output':cpu(result)},dest/f'{name}.kernel.{step:03d}.pt')
                            return result
                        return call
                    module.gated_delta_rule=capture_gdn(operation,full)
                    module._prepare_qkv_for_gated_delta_rule=capture_gdn(module._prepare_qkv_for_gated_delta_rule,full+'.prepare')
                    from megatron.core.ssm import gated_delta_net
                    for attr in ('l2norm','causal_conv1d'):
                        f=getattr(gated_delta_net,attr,None)
                        metadata[-1][attr]={'module':getattr(f,'__module__',None),'name':getattr(f,'__qualname__',None),'source':inspect.getsourcefile(f) if f is not None else None}
                if not name or not any(t in name for t in ('embedding','decoder.layers','final_layernorm','output_layer')):continue
                if name.count('.')>7:continue
                pre,post=make_hooks(full)
                handles.extend((module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post,with_kwargs=True)))
        original_forward = self.engine.forward_backward_batch
        def checked_forward(*args, **kwargs):
            try:
                return original_forward(*args, **kwargs)
            except Exception:
                import traceback
                (dest / 'forward_error.txt').write_text(traceback.format_exc())
                raise
        self.engine.forward_backward_batch = checked_forward
        self._diagnostic_handles=handles
        (dest/'modules.json').write_text(json.dumps(metadata,indent=2))
        return {'rank':rank,'traced':True,'pp':mpu.get_pipeline_model_parallel_rank(),'modules':len(metadata)}
