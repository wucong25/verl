"""Broader inference-only PD, batching, chunking and cache-slot reuse diagnostic."""
import json,sys,time,traceback
from pathlib import Path


def main():
    from vllm import LLM,SamplingParams
    chunked=bool(int(sys.argv[1]))
    root=Path('/home/w00848261/verl/.zero_diff/runs')/f'proxy-bi-matrix-{int(chunked):03d}'
    root.mkdir(parents=True,exist_ok=True)
    report={'scope':'4-layer inference only','chunked_prefill':chunked,'cases':[]}
    source=json.loads(Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-008/result.json').read_text())
    base=source['tokens']+source['response']
    def save():
        (root/'result.json').write_text(json.dumps(report,indent=2))
    def cmp(a,b):
        assert len(a)==len(b)
        return {'count':len(a),'mismatch':sum(x!=y for x,y in zip(a,b)),'max_abs':max((abs(x-y) for x,y in zip(a,b)),default=0)}
    def prompt_probs(out):
        return [d[t].logprob for t,d in zip(out.prompt_token_ids[1:],out.prompt_logprobs[1:])]
    try:
        llm=LLM(model='/home/w00848261/verl/.zero_diff/proxy-4layer',trust_remote_code=True,tensor_parallel_size=8,dtype='bfloat16',enforce_eager=True,max_model_len=320,max_num_seqs=4,max_num_batched_tokens=64 if chunked else 320,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=chunked,limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True}},seed=42)
        sample=SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=8,ignore_eos=True,logprobs=0,seed=42)
        force=SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0)
        started=time.time()
        for length in (1,63,64,65,129):
            tokens=base[:length]
            generated=llm.generate([{'prompt_token_ids':tokens}],sample,use_tqdm=False)[0].outputs[0]
            response=list(generated.token_ids)
            decode=[d[t].logprob for t,d in zip(response,generated.logprobs)]
            forced=llm.generate([{'prompt_token_ids':tokens+response}],force,use_tqdm=False)[0]
            prefill=[forced.prompt_logprobs[length+i][t].logprob for i,t in enumerate(response)]
            report['cases'].append({'kind':'pd','length':length,'tokens':tokens,'response':response,'decode':decode,'prefill':prefill,**cmp(decode,prefill)})
            save()
        reference=llm.generate([{'prompt_token_ids':base}],force,use_tqdm=False)[0]
        for companions in ([base[:63]],[base[:65],base[:129]],[base+base[:100],base[:1]]):
            prompts=[base]+companions
            outputs=llm.generate([{'prompt_token_ids':p} for p in prompts],force,use_tqdm=False)
            a=prompt_probs(reference);b=prompt_probs(outputs[0])
            report['cases'].append({'kind':'batch_forced','lengths':[len(p) for p in prompts],'reference':a,'actual':b,**cmp(a,b)})
            save()
        target=base[:63]
        alone=llm.generate([{'prompt_token_ids':target}],sample,use_tqdm=False)[0].outputs[0]
        together=llm.generate([{'prompt_token_ids':target},{'prompt_token_ids':base[:129]}],[sample,sample],use_tqdm=False)[0].outputs[0]
        a=[d[t].logprob for t,d in zip(alone.token_ids,alone.logprobs)]
        b=[d[t].logprob for t,d in zip(together.token_ids,together.logprobs)]
        report['cases'].append({'kind':'mixed_decode','tokens_equal':list(alone.token_ids)==list(together.token_ids),'alone_tokens':list(alone.token_ids),'mixed_tokens':list(together.token_ids),**cmp(a,b)})
        report['elapsed']=time.time()-started
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        save();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
