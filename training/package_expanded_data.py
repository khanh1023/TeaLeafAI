"""Package the exact curated train/validation/test dataset, without original archives or caches."""
from pathlib import Path
import argparse,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
DATA=ROOT/'training/dataset';records=json.loads((DATA/'manifest.json').read_text());args.output.parent.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(args.output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
 for record in records:
  path=DATA/record['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
  archive.write(path,'TeaLeafAI-training/dataset/'+record['path'])
 for name in ['manifest.json','classes.json','split_summary.json','exclusions.json']:
  archive.write(DATA/name,'TeaLeafAI-training/dataset/'+name)
 archive.write(ROOT/'docs/TRAINING_DATA.md','TeaLeafAI-training/README.md')
with zipfile.ZipFile(args.output) as archive:assert archive.testzip() is None
hash_=hashlib.sha256(args.output.read_bytes()).hexdigest()
args.output.with_name('TRAINING_SHA256SUMS.txt').write_text(f'{hash_}  {args.output.name}\n')
print(args.output,args.output.stat().st_size,'bytes;',hash_)
