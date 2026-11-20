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
    from online_worker_003 import OnlineWorker
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
    from verl.utils import tensordict_utils as tu
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy-online-003');root.mkdir(parents=True,exist_ok=True)
    ipc=Path('/home/w00848261/verl/.zero_diff/ipc');ipc.mkdir(exist_ok=True)
    os.environ['VERL_WEIGHT_SYNC_IPC_DIR']=str(ipc)
    os.environ['VERL_RAY_JOB_ID']='zd003'
    os.environ['VERL_REPLICA_RANK']='0'
    report={'scope':'4-layer proxy; native dual forward, two optimizer steps, resident IPC sync','phases':[],'optimizer_steps':0}
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
        sample=json.load(open('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-028/result.json'))
        tokens=sample['tokens']
        cfg=TrainingWorkerConfig(model_type='language_model',model_config=HFModelConfig(path='/home/w00848261/verl/.zero_diff/proxy-4layer',use_remove_padding=True,trust_remote_code=True),engine_config=McoreEngineConfig(forward_only=False,use_mbridge=True,tensor_model_parallel_size=2,pipeline_model_parallel_size=2,context_parallel_size=1,expert_model_parallel_size=8,expert_tensor_parallel_size=1,param_offload=True,optimizer_offload=True,use_dynamic_bsz=True,use_remove_padding=True,max_token_len_per_gpu=1024,infer_max_token_len_per_gpu=1024,dtype='bfloat16',override_mcore_model_config={'preserve_gdn_a_log_fp32':True,'npu_batch_invariant_matmul':True,'npu_batch_invariant_reduction':True},override_transformer_config={'attention_backend':'auto','use_flash_attn':True,'use_triton_gdn':False,'use_ascend_gdn':True,'moe_grouped_gemm':True,'moe_token_dispatcher_type':'alltoall','moe_permute_fusion':True,'hidden_dropout':0.0,'attention_dropout':0.0}),optimizer_config=McoreOptimizerConfig(lr=1e-6,lr_decay_steps=2),checkpoint_config=CheckpointConfig())
        env={k:v for k,v in os.environ.items() if k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','TMPDIR','XDG_CACHE_HOME','ASCEND_WORK_PATH','ASCEND_CACHE_PATH','HF_HOME','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','RAY_AUTH_MODE','RAY_AUTH_TOKEN_PATH','VERL_WEIGHT_SYNC_IPC_DIR']}
        env['VLLM_BATCH_INVARIANT']='0'
        env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES']='1'
        ray.init(num_cpus=32,_temp_dir='/home/w00848261/verl/.zero_diff/ray',runtime_env={'env_vars':env,'worker_process_setup_hook':worker_setup},include_dashboard=False)
        wg=RayWorkerGroup(device_name='npu',resource_pool=RayResourcePool(process_on_nodes=[16],max_colocate_count=1),ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(OnlineWorker),config=cfg))
        wg.reset()
        wg._collect_info['train']=wg._query_collect_info('train')
        actor=ActorConfig(strategy='megatron',rollout_n=1,ppo_micro_batch_size_per_gpu=1,use_kl_loss=False)
        wg.set_loss_fn(partial(ppo_loss,config=actor))
        os.environ['VLLM_BATCH_INVARIANT']='1'
        from vllm import LLM, SamplingParams
        llm=LLM(model='/home/w00848261/verl/.zero_diff/proxy-4layer',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,dtype='bfloat16',enforce_eager=True,max_model_len=256,max_num_seqs=2,max_num_batched_tokens=512,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=False,enable_sleep_mode=True,worker_extension_cls='verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension',limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True,'training_tensor_parallel_size':2,'training_router_math':True,'training_expert_math':True,'training_attention_math':True,'training_logprob_math':True}},seed=42)
        identity=llm.collective_rpc(resident_identity);report['resident_initial']=identity
        before=wg.parameter_audit();(root/'parameters-theta0.json').write_text(json.dumps(before))
        for version in range(3):
            phase={'version':version};report['phases'].append(phase);save()
            started=time.time()
            params=SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=16,ignore_eos=True,logprobs=0,seed=42+version)
            first=llm.generate([{'prompt_token_ids':tokens}],params,use_tqdm=False)[0].outputs[0]
            response=list(first.token_ids);rollout=torch.tensor([p[t].logprob for p,t in zip(first.logprobs,response)],dtype=torch.float32)
            forced=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0,ignore_eos=True)
            full=tokens+response
            prefill=llm.generate([{'prompt_token_ids':full}],forced,use_tqdm=False)[0]
            lp=torch.tensor([prefill.prompt_logprobs[len(tokens)+i][token].logprob for i,token in enumerate(response)])
            phase['pd']=cmp(rollout,lp)
            mixed=llm.generate([{'prompt_token_ids':full},{'prompt_token_ids':tokens[:97]+response}],forced,use_tqdm=False)[0]
            mixed_lp=torch.tensor([mixed.prompt_logprobs[len(tokens)+i][token].logprob for i,token in enumerate(response)])
            phase['mixed_batch']=cmp(rollout,mixed_lp)
            phase['rollout_seconds']=time.time()-started
            phase['response']=response;phase['rollout_logprobs']=rollout.tolist()
            phase['resident']=llm.collective_rpc(resident_identity)
            assert phase['resident']==identity,'Inference process/model identity changed'
            save()
            assert phase['pd']['mismatch']==0 and phase['mixed_batch']['mismatch']==0
            llm.sleep(level=1)
            prompt=torch.tensor(tokens).unsqueeze(0).repeat(4,1);responses=torch.tensor(response).unsqueeze(0).repeat(4,1);ids=torch.cat((prompt,responses),1)
            data=DataProto.from_single_dict({'input_ids':ids,'prompts':prompt,'responses':responses,'attention_mask':torch.ones_like(ids),'position_ids':torch.arange(ids.shape[1]).unsqueeze(0).repeat(4,1),'response_mask':torch.ones_like(responses)},meta_info={'temperature':1.0,'global_token_num':[ids.shape[1]]*4,'compute_loss':False})
            td=left_right_2_no_padding(data.to_tensordict());started=time.time()
            infer=wg.infer_batch(td).get();old=no_padding_2_padding(tu.get(infer,'log_probs').cpu(),td)
            target=rollout.unsqueeze(0).repeat(4,1);phase['recompute']=cmp(old,target)
            data.batch['old_log_probs']=old;data.batch['advantages']=torch.ones_like(old);data.batch['ref_log_prob']=old.clone()
            td=left_right_2_no_padding(data.to_tensordict());tu.assign_non_tensor(td,global_batch_size=4)
            wg.optimizer_zero_grad();grad=wg.forward_backward(td).get();actual=no_padding_2_padding(tu.get(grad,'log_probs').cpu(),td)
            phase['grad_forward']=cmp(actual,target);phase['grad_vs_recompute']=cmp(actual,old)
            phase['native_seconds']=time.time()-started
            torch.save({'rollout':target,'recompute':old,'grad_forward':actual},root/f'dual-theta{version}.pt');save()
            assert all(phase[k]['mismatch']==0 and phase[k]['finite'] for k in ('recompute','grad_forward','grad_vs_recompute'))
            if version==2:
                break
            phase['optimizer']=wg.optimizer_step({'lr':1e-5});report['optimizer_steps']+=1
            after=wg.parameter_audit();(root/f'parameters-theta{version+1}.json').write_text(json.dumps(after))
            phase['changed_parameters']=[sum(item['sha256']!=b['parameters'][name]['sha256'] for name,item in a['parameters'].items()) for a,b in zip(after,before)]
            assert sum(phase['changed_parameters'])>0,'Optimizer did not change model parameters'
            before=after;save()
            llm.wake_up(tags=['weights'])
            started=time.time();pending=wg.export_and_send('zd003',str(ipc))
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
