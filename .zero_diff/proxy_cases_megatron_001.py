"""Native no-grad and grad-enabled Megatron forwards, with no optimizer update."""
import json,os,traceback,time
from pathlib import Path

def worker_setup():
    import megatron_adaptor

def main():
    import megatron_adaptor
    import torch,ray
    from functools import partial
    from verl import DataProto
    from verl.single_controller.ray import RayClassWithInitArgs,RayResourcePool,RayWorkerGroup
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig,McoreEngineConfig,McoreOptimizerConfig,ActorConfig
    from verl.workers.engine_workers import TrainingWorkerConfig
    from verl.workers.engine_workers_tinker import TinkerTrainingWorker
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import left_right_2_no_padding,no_padding_2_padding
    from verl.utils import tensordict_utils as tu
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy-cases-megatron-001')
    root.mkdir(parents=True,exist_ok=True)
    report={'scope':'proxy native infer_batch and grad-enabled engine forward_backward_batch; no optimizer update','topology':{'TP':2,'PP':2,'CP':1,'EP':8,'ETP':1,'world':16}}
    try:
        sample=json.loads(Path('/home/w00848261/verl/.zero_diff/runs/proxy-cases-inference-001/result.json').read_text())
        assert sample.get('passed'), sample.get('error')
        cases=sample['cases']*2
        cfg=TrainingWorkerConfig(model_type='language_model',model_config=HFModelConfig(path='/home/w00848261/verl/.zero_diff/proxy-4layer',use_remove_padding=True,trust_remote_code=True),engine_config=McoreEngineConfig(forward_only=False,use_mbridge=True,tensor_model_parallel_size=2,pipeline_model_parallel_size=2,context_parallel_size=1,expert_model_parallel_size=8,expert_tensor_parallel_size=1,param_offload=True,optimizer_offload=True,use_dynamic_bsz=True,use_remove_padding=True,max_token_len_per_gpu=4096,infer_max_token_len_per_gpu=4096,dtype='bfloat16',override_mcore_model_config={'preserve_gdn_a_log_fp32':True,'npu_batch_invariant_matmul':True,'npu_batch_invariant_reduction':True},override_transformer_config={'attention_backend':'auto','use_flash_attn':True,'use_triton_gdn':False,'use_ascend_gdn':True,'moe_grouped_gemm':True,'moe_token_dispatcher_type':'alltoall','moe_permute_fusion':True,'hidden_dropout':0.0,'attention_dropout':0.0}),optimizer_config=McoreOptimizerConfig(lr=1e-6,lr_decay_steps=2),checkpoint_config=CheckpointConfig())
        (root/'config.txt').write_text(repr(cfg))
        env={k:v for k,v in os.environ.items() if k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','HF_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','RAY_AUTH_MODE','RAY_AUTH_TOKEN_PATH']}
        env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES']='1'
        ray.init(num_cpus=32,_temp_dir='/home/w00848261/verl/.zero_diff/ray',runtime_env={'env_vars':env,'worker_process_setup_hook':worker_setup},include_dashboard=False)
        wg=RayWorkerGroup(device_name="npu",resource_pool=RayResourcePool(process_on_nodes=[16],max_colocate_count=1),ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TinkerTrainingWorker),config=cfg))
        wg.reset()
        width=max(len(c['tokens']) for c in cases)
        prompt=torch.zeros((len(cases),width),dtype=torch.long)
        mask=torch.zeros_like(prompt)
        for i,c in enumerate(cases):
            prompt[i,-len(c['tokens']):]=torch.tensor(c['tokens'])
            mask[i,-len(c['tokens']):]=1
        response=torch.tensor([c['response'] for c in cases],dtype=torch.long)
        ids=torch.cat((prompt,response),dim=1)
        attention_mask=torch.cat((mask,torch.ones_like(response)),dim=1)
        data=DataProto.from_single_dict({'input_ids':ids,'prompts':prompt,'responses':response,'attention_mask':attention_mask,'position_ids':(attention_mask.cumsum(-1)-1).clamp(min=0),'response_mask':torch.ones_like(response)},meta_info={'temperature':1.0,'global_token_num':attention_mask.sum(-1).tolist(),'compute_loss':False})
        td=left_right_2_no_padding(data.to_tensordict())
        wg._collect_info['train']=wg._query_collect_info('train')
        started=time.time()
        infer=wg.infer_batch(td).get()
        report['recompute_seconds']=time.time()-started
        old=no_padding_2_padding(tu.get(infer,'log_probs').cpu(),td)
        torch.save({'native_recompute':old,'rollout':[c['decode_logprobs'] for c in cases]},root/'recompute.pt')
        def compare(a,b):
            assert a.shape==b.shape,(a.shape,b.shape)
            return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a-b).abs().max()),'dtype':str(a.dtype),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all())}
        rollout=torch.tensor([c['decode_logprobs'] for c in cases],dtype=old.dtype)
        report['recompute_vs_rollout']=compare(old,rollout)
        data.batch['old_log_probs']=old
        data.batch['advantages']=torch.ones_like(old)
        data.batch['ref_log_prob']=old.clone()
        actor=ActorConfig(strategy='megatron',rollout_n=1,ppo_micro_batch_size_per_gpu=1,use_kl_loss=False)
        wg.set_loss_fn(partial(ppo_loss,config=actor))
        td=left_right_2_no_padding(data.to_tensordict())
        tu.assign_non_tensor(td,global_batch_size=len(cases))
        wg.optimizer_zero_grad()
        started=time.time()
        grad=wg.forward_backward(td).get()
        report['forward_backward_seconds']=time.time()-started
        actual=no_padding_2_padding(tu.get(grad,'log_probs').cpu(),td)
        report['grad_forward_vs_rollout']=compare(actual,rollout)
        report['grad_forward_vs_recompute']=compare(actual,old)
        report['passed']=all(report[k]['count']==128 and report[k]['finite'] and report[k]['mismatch']==0 for k in ('recompute_vs_rollout','grad_forward_vs_rollout','grad_forward_vs_recompute'))
        torch.save({'native_recompute':old,'grad_forward':actual,'rollout':rollout},root/'dual_forward.pt')
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        (root/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)
        if ray.is_initialized():
            ray.shutdown()

if __name__=='__main__':
    main()
