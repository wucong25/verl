"""Materialize a eight-text-layer diagnostic checkpoint without editing source weights."""
import collections,json,re,shutil,hashlib
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
source=Path('/mnt/weight/Qwen3.5-35B-A3B')
root=Path('/home/w00848261/verl/.zero_diff')
out=root/'proxy-8layer'
out.mkdir(exist_ok=True)
if (out/'manifest.json').exists():
    raise SystemExit('Proxy already exists; inspect manifest before rebuilding')
config=json.loads((source/'config.json').read_text())
config['text_config']['num_hidden_layers']=8
config['text_config']['layer_types']=config['text_config']['layer_types'][:8]
config['text_config']['mtp_num_hidden_layers']=0
index=json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
groups=collections.defaultdict(list)
for key,shard in index.items():
    m=re.search(r'model.language_model.layers\.(\d+)\.',key)
    if m and int(m.group(1))>=8:
        continue
    if key.startswith('mtp.'):
        continue
    groups[shard].append(key)
newmap={}; total=0; receipts=[]
for i,(shard,keys) in enumerate(sorted(groups.items()),1):
    with safe_open(str(source/shard),framework='pt',device='cpu') as reader:
        tensors={key:reader.get_tensor(key) for key in keys}
    name=f'model-{i:05d}-of-{len(groups):05d}.safetensors'
    save_file(tensors,str(out/name),metadata={'format':'pt'})
    total+=sum(t.numel()*t.element_size() for t in tensors.values())
    newmap.update({key:name for key in keys})
    receipts.append({'source_shard':shard,'source_size':(source/shard).stat().st_size,'output_shard':name,'keys':keys,'output_size':(out/name).stat().st_size})
    del tensors
    print(name,len(keys),flush=True)
(out/'config.json').write_text(json.dumps(config,indent=2))
(out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':newmap},indent=2))
for p in source.iterdir():
    if p.name!='config.json' and p.name!='model.safetensors.index.json' and p.suffix in {'.json','.jinja','.txt','.model'}:
        if p.is_file():
            shutil.copyfile(p,out/p.name)
(out/'manifest.json').write_text(json.dumps({'source':str(source),'layers':list(range(8)),'text_only_test_scope':True,'vision_weights_retained':True,'mtp':False,'total_size':total,'source_config_sha256':hashlib.sha256((source/'config.json').read_bytes()).hexdigest(),'shards':receipts},indent=2))
print('PROXY_COMPLETE',total,len(newmap),flush=True)
