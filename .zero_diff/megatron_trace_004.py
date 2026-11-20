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
        if mpu.get_tensor_model_parallel_rank()!=0 or mpu.get_data_parallel_rank()!=0:
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
                if not name or not any(t in name for t in ('embedding','decoder.layers','final_layernorm','output_layer')):continue
                if name.count('.')>7:continue
                pre,post=make_hooks(full)
                handles.extend((module.register_forward_pre_hook(pre,with_kwargs=True),module.register_forward_hook(post,with_kwargs=True)))
        self._diagnostic_handles=handles
        (dest/'modules.json').write_text(json.dumps(metadata,indent=2))
        return {'rank':rank,'traced':True,'pp':mpu.get_pipeline_model_parallel_rank(),'modules':len(metadata)}
