"""Snapshot current allowed repository changes into reproducible review patches."""
import argparse,datetime,hashlib,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--status',required=True);args=parser.parse_args()
root=Path('/home/w00848261/verl/.zero_diff/delivery');root.mkdir(exist_ok=True)
records=[]
for repo in (Path('/home/w00848261/verl'),Path('/vllm'),Path('/vllm-ascend')):
 def git(*cmd,check=True):return subprocess.run(['git','--no-pager','-C',str(repo),*cmd],check=check,capture_output=True)
 patch=git('diff','--no-ext-diff','--binary','HEAD','--').stdout
 untracked=git('ls-files','--others','--exclude-standard','-z').stdout.decode().split('\0')
 for name in sorted(filter(None,untracked)):
  p=Path(name)
  allowed=p.parts[0] in (('verl','tests') if repo.name=='verl' else ('vllm_ascend','tests') if repo.name=='vllm-ascend' else ('vllm','tests'))
  if not allowed or p.suffix not in ('.py','.sh','.yaml','.toml'):continue
  result=git('diff','--no-ext-diff','--no-index','--binary','--','/dev/null',name,check=False)
  assert result.returncode in (0,1),result.stderr.decode()
  patch+=result.stdout
 out=root/f'{repo.name}.patch';out.write_bytes(patch)
 if patch:git('apply','--reverse','--check',str(out))
 records.append({'repo':str(repo),'base':git('rev-parse','HEAD').stdout.decode().strip(),'patch':out.name,'bytes':len(patch),'sha256':hashlib.sha256(patch).hexdigest()})
manifest={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':args.status,'patches':records}
(root/'patch-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
