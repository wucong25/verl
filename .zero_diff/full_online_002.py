"""Three resident-weight versions, two native updates, real device IPC sync."""
import json, os, time, traceback
from pathlib import Path


def worker_setup():
    import megatron_adaptor


def resident_identity(worker):
    return {'pid': os.getpid(), 'rank': worker.rank, 'model_id': id(worker.model_runner.model)}


def main():
    import megatron_adaptor
    import torch, ray
    from functools import partial
    from verl import DataProto
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig, ActorConfig
    from verl.workers.engine_workers import TrainingWorkerConfig
    from online_worker_008 import OnlineWorker
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
    from verl.utils import tensordict_utils as tu
    root=Path('/home/w00848261/verl/.zero_diff/runs/full-online-002');root.mkdir(parents=True,exist_ok=True)
    ipc=Path('/home/w00848261/verl/.zero_diff/ipc');ipc.mkdir(exist_ok=True)
    os.environ['VERL_WEIGHT_SYNC_IPC_DIR']=str(ipc)
    os.environ['VERL_RAY_JOB_ID']='zdf002'
    os.environ['VERL_REPLICA_RANK']='0'
    report={'scope':'40-layer full model; native dual forward, two optimizer steps, resident IPC sync','phases':[],'optimizer_steps':0}
    llm=None
    def serial(value):
        if isinstance(value,torch.Tensor):return value.tolist()
        return str(value)
    def save():
        (root/'result.json').write_text(json.dumps(report,indent=2,default=serial))
    def cmp(a,b):
        assert a.shape==b.shape and a.numel()>0
        return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a-b).abs().max()),'finite':bool(torch.isfinite(a).all() and torch.isfinite(b).all()),'dtype':str(a.dtype)}
    try:
        cases=json.loads(Path('/home/w00848261/verl/.zero_diff/case_manifest_full_dataset.json').read_text())['cases']
        cfg=TrainingWorkerConfig(model_type='language_model',model_config=HFModelConfig(path='/mnt/weight/Qwen3.5-35B-A3B',use_remove_padding=True,trust_remote_code=True),engine_config=McoreEngineConfig(forward_only=False,use_mbridge=True,tensor_model_parallel_size=2,pipeline_model_parallel_size=2,context_parallel_size=1,expert_model_parallel_size=8,expert_tensor_parallel_size=1,param_offload=True,optimizer_offload=True,use_dynamic_bsz=True,use_remove_padding=True,max_token_len_per_gpu=4096,infer_max_token_len_per_gpu=4096,dtype='bfloat16',override_mcore_model_config={'preserve_gdn_a_log_fp32':True,'preserve_gdn_norm_export_fp32':True,'npu_batch_invariant_matmul':True,'npu_batch_invariant_reduction':True},override_transformer_config={'attention_backend':'auto','use_flash_attn':True,'use_triton_gdn':False,'use_ascend_gdn':True,'moe_grouped_gemm':True,'moe_token_dispatcher_type':'alltoall','moe_permute_fusion':True,'hidden_dropout':0.0,'attention_dropout':0.0}),optimizer_config=McoreOptimizerConfig(lr=1e-5,lr_decay_steps=2),checkpoint_config=CheckpointConfig())
        from dataclasses import asdict
        (root/'resolved-training-config.json').write_text(json.dumps(asdict(cfg),indent=2,default=serial))
        env={k:v for k,v in os.environ.items() if k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','HF_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','RAY_AUTH_MODE','RAY_AUTH_TOKEN_PATH','VERL_WEIGHT_SYNC_IPC_DIR']}
        env['VLLM_BATCH_INVARIANT']='0'
        env['HCCL_NPU_SOCKET_PORT_RANGE']='auto'
        env['HCCL_HOST_SOCKET_PORT_RANGE']='auto'
        env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES']='1'
        ray.init(num_cpus=32,_temp_dir='/home/w00848261/verl/.zero_diff/ray',runtime_env={'env_vars':env,'worker_process_setup_hook':worker_setup},include_dashboard=False)
        wg=RayWorkerGroup(device_name='npu',resource_pool=RayResourcePool(process_on_nodes=[16],max_colocate_count=1),ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(OnlineWorker),config=cfg))
        wg.reset()
        wg._collect_info['train']=wg._query_collect_info('train')
        actor=ActorConfig(strategy='megatron',rollout_n=1,ppo_micro_batch_size_per_gpu=1,use_kl_loss=False)
        wg.set_loss_fn(partial(ppo_loss,config=actor))
        before=wg.parameter_audit();(root/'parameters-theta0.json').write_text(json.dumps(before))
        os.environ['VLLM_BATCH_INVARIANT']='1'
        from vllm import LLM, SamplingParams
        llm=LLM(model='/mnt/weight/Qwen3.5-35B-A3B',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,dtype='bfloat16',enforce_eager=True,max_model_len=1152,max_num_seqs=4,max_num_batched_tokens=2304,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=False,enable_sleep_mode=True,worker_extension_cls='verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension',limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True,'training_tensor_parallel_size':2,'training_router_math':True,'training_expert_math':True,'training_attention_math':True,'training_logprob_math':True}},seed=42)
        (root/'resolved-inference-config.txt').write_text(str(llm.llm_engine.vllm_config))
        identity=llm.collective_rpc(resident_identity);report['resident_initial']=identity
        from weight_audit_007 import capture_weight_state
        pre=llm.collective_rpc(capture_weight_state,args=(str(root/'weight-audit'),'before-sync'))
        (root/'rollout-weights-before.json').write_text(json.dumps(pre))
        llm.sleep(level=1);llm.wake_up(tags=['weights'])
        pending=wg.export_and_send('zdf002',str(ipc))
        llm.collective_rpc('update_weights_from_ipc',timeout=900,kwargs={'peft_config':None,'base_sync_done':True,'use_shm':False})
        report['initial_weight_sync']=ray.get(pending)
        post=llm.collective_rpc(capture_weight_state,args=(str(root/'weight-audit'),'after-sync'))
        (root/'rollout-weights-after.json').write_text(json.dumps(post))
        report['initial_weight_changes']=[{'rank':a['rank'],'changed':[name for name,value in a['parameters'].items() if value!=b['parameters'].get(name)]} for a,b in zip(pre,post)]
        save();assert all(not item['changed'] for item in report['initial_weight_changes']), report['initial_weight_changes']
        llm.wake_up(tags=['kv_cache','scheduling']);save()
        for version in range(3):
            phase={'version':version};report['phases'].append(phase);save()
            started=time.time()
            params=SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=16,ignore_eos=True,logprobs=0,seed=42+version)
            outputs=llm.generate([{'prompt_token_ids':c['tokens']} for c in cases],params,use_tqdm=False)
            responses=[list(out.outputs[0].token_ids) for out in outputs]
            rollout=torch.tensor([[p[t].logprob for p,t in zip(out.outputs[0].logprobs,response)] for out,response in zip(outputs,responses)],dtype=torch.float32)
            forced=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0,ignore_eos=True)
            histories=[{'prompt_token_ids':c['tokens']+response} for c,response in zip(cases,responses)]
            prefills=llm.generate(histories,forced,use_tqdm=False)
            def prefill_lp(out,c,response):
                return [out.prompt_logprobs[len(c['tokens'])+i][token].logprob for i,token in enumerate(response)]
            batched=torch.tensor([prefill_lp(out,c,response) for out,c,response in zip(prefills,cases,responses)],dtype=torch.float32)
            singles=[llm.generate([history],forced,use_tqdm=False)[0] for history in histories]
            single=torch.tensor([prefill_lp(out,c,response) for out,c,response in zip(singles,cases,responses)],dtype=torch.float32)
            phase['pd']=cmp(rollout,batched)
            phase['mixed_batch']=cmp(batched,single)
            phase['rollout_seconds']=time.time()-started
            torch.save({'rollout':rollout,'batched_prefill':batched,'single_prefill':single},root/f'inference-theta{version}.pt')
            phase['responses']=responses;phase['rollout_logprobs']=rollout.tolist()
            phase['resident']=llm.collective_rpc(resident_identity)
            assert phase['resident']==identity,'Inference process/model identity changed'
            save()
            assert all(phase[k]['mismatch']==0 and phase[k]['finite'] and phase[k]['count']==64 for k in ('pd','mixed_batch'))
            llm.sleep(level=1)
            train_cases=cases*2
            width=max(len(c['tokens']) for c in train_cases)
            prompt=torch.zeros((len(train_cases),width),dtype=torch.long)
            mask=torch.zeros_like(prompt)
            for i,c in enumerate(train_cases):
                prompt[i,-len(c['tokens']):]=torch.tensor(c['tokens'])
                mask[i,-len(c['tokens']):]=1
            response_tensor=torch.tensor(responses*2,dtype=torch.long)
            ids=torch.cat((prompt,response_tensor),dim=1)
            attention_mask=torch.cat((mask,torch.ones_like(response_tensor)),dim=1)
            data=DataProto.from_single_dict({'input_ids':ids,'prompts':prompt,'responses':response_tensor,'attention_mask':attention_mask,'position_ids':(attention_mask.cumsum(-1)-1).clamp(min=0),'response_mask':torch.ones_like(response_tensor)},meta_info={'temperature':1.0,'global_token_num':attention_mask.sum(-1).tolist(),'compute_loss':False})
            td=left_right_2_no_padding(data.to_tensordict());started=time.time()
            infer=wg.infer_batch(td).get();old=no_padding_2_padding(tu.get(infer,'log_probs').cpu(),td)
            target=rollout.repeat(2,1);phase['recompute']=cmp(old,target)
            save();assert phase['recompute']['mismatch']==0, phase['recompute']
            data.batch['old_log_probs']=old;data.batch['advantages']=torch.ones_like(old);data.batch['ref_log_prob']=old.clone()
            td=left_right_2_no_padding(data.to_tensordict());tu.assign_non_tensor(td,global_batch_size=8)
            if version < 2:
                grad=wg.train_and_step(td).get()
                report['optimizer_steps']+=1
                phase['optimizer']={'api':'native engine.optimizer_step inside the same train_mode as forward_backward','grad_norm':tu.get(grad,'native_optimizer_grad_norm')}
            else:
                wg.optimizer_zero_grad();grad=wg.forward_backward(td).get()
            actual=no_padding_2_padding(tu.get(grad,'log_probs').cpu(),td)
            phase['grad_forward']=cmp(actual,target);phase['grad_vs_recompute']=cmp(actual,old)
            phase['native_seconds']=time.time()-started
            torch.save({'rollout':target,'recompute':old,'grad_forward':actual},root/f'dual-theta{version}.pt');save()
            assert all(phase[k]['mismatch']==0 and phase[k]['finite'] for k in ('recompute','grad_forward','grad_vs_recompute'))
            if version==2:
                break
            after=wg.parameter_audit();(root/f'parameters-theta{version+1}.json').write_text(json.dumps(after))
            phase['changed_parameters']=[sum(item['sha256']!=b['parameters'][name]['sha256'] for name,item in a['parameters'].items()) for a,b in zip(after,before)]
            assert sum(phase['changed_parameters'])>0,'Optimizer did not change model parameters'
            before=after;save()
            llm.wake_up(tags=['weights'])
            started=time.time();pending=wg.export_and_send('zdf002',str(ipc))
            llm.collective_rpc('update_weights_from_ipc',timeout=900,kwargs={'peft_config':None,'base_sync_done':True,'use_shm':False})
            phase['weight_sync']=ray.get(pending)
            phase['weight_sync_seconds']=time.time()-started
            llm.wake_up(tags=['kv_cache','scheduling']);save()
        report['passed']=report['optimizer_steps']==2 and len(report['phases'])==3
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        save();print(json.dumps(report,indent=2,default=serial),flush=True)
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()
        if ray.is_initialized():ray.shutdown()

if __name__=='__main__':main()
