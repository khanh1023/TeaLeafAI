"""Download attributed tea data, curate without changing originals, group and split 70/15/15."""
from pathlib import Path
import argparse,collections,hashlib,io,json,random,re,shutil,zipfile
import numpy as np
from PIL import Image,ImageOps
import imagehash
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from prepare_dataset import CLASSES,CLASS_MAP,normalize_name,EXTENSIONS

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--cache-dir',type=Path,required=True)
p.add_argument('--healthy-extra',type=int,default=400)
p.add_argument('--replace',action='store_true')
args=p.parse_args()
OUT=ROOT/'training/dataset'
if OUT.exists():
 if not args.replace:raise SystemExit('Generated dataset exists; pass --replace to rebuild.')
 if not ((OUT/'manifest.json').exists() or ((OUT/'classes.json').exists() and (OUT/'split_summary.json').exists())):raise SystemExit('Refusing to replace an unrecognized folder.')
 shutil.rmtree(OUT)
OUT.mkdir(parents=True)
rng=random.Random(42)
records=[];excluded=[];raw_seen={};pixel_seen={};conflicts=set();fingerprints=[]

def sha(data):return hashlib.sha256(data).hexdigest()
def canonical_hashes(image):
 image.thumbnail((256,256))
 pixels=np.asarray(image)
 mask=(pixels[:,:,1]>pixels[:,:,0]*1.03)&(pixels[:,:,1]>pixels[:,:,2]*1.07)
 ys,xs=np.where(mask)
 if len(xs)>256:
  box=(max(0,int(xs.min())-5),max(0,int(ys.min())-5),min(image.width,int(xs.max())+6),min(image.height,int(ys.max())+6))
  image=image.crop(box)
 variants=[]
 for angle in [0,90,180,270]:
  rotated=image.rotate(angle,expand=True)
  variants.extend([int(str(imagehash.phash(rotated)),16),int(str(imagehash.phash(ImageOps.mirror(rotated))),16)])
 return min(variants)

def add(raw,label,source,name,healthy_filter=False):
 digest=sha(raw)
 if digest in raw_seen:
  old=records[raw_seen[digest]]
  if old['label']!=label:conflicts.update([old['id'],digest[:16]])
  excluded.append({'source':source,'original':name,'reason':'exact_duplicate_or_conflicting_label','matches':old['id'],'label':label})
  return False
 try:
  with Image.open(io.BytesIO(raw)) as original:
   image=ImageOps.exif_transpose(original).convert('RGB');image.load()
   width,height=image.size
   if min(width,height)<80:raise ValueError('image_too_small')
   image.thumbnail((512,512),Image.Resampling.LANCZOS)
   pixel_hash=sha(np.asarray(image).tobytes())
   if pixel_hash in pixel_seen:
    old=records[pixel_seen[pixel_hash]]
    if old['label']!=label:conflicts.add(old['id'])
    excluded.append({'source':source,'original':name,'reason':'decoded_pixel_duplicate','matches':old['id'],'label':label})
    return False
   perceptual=canonical_hashes(image.copy())
   if healthy_filter and any((perceptual^other).bit_count()<=6 for other in fingerprints):
    excluded.append({'source':source,'original':name,'reason':'healthy_near_duplicate_not_added','label':label})
    return False
   normalized=io.BytesIO();image.save(normalized,'JPEG',quality=92,subsampling=0)
 except Exception as e:
  excluded.append({'source':source,'original':name,'label':label,'reason':str(e)})
  return False
 index=len(records);identifier=digest[:16]
 raw_seen[digest]=index;pixel_seen[pixel_hash]=index;fingerprints.append(perceptual)
 temporary=OUT/'staging'/label/f'{source}-{identifier}.jpg';temporary.parent.mkdir(parents=True,exist_ok=True);temporary.write_bytes(normalized.getvalue())
 records.append({'id':identifier,'label':label,'source':source,'original':name,'raw_sha256':digest,'pixel_sha256':pixel_hash,'phash':f'{perceptual:016x}','original_size':[width,height],'sha256':sha(normalized.getvalue()),'temporary':str(temporary)})
 return True

# Keep the original files untouched. Exact contradictory labels are quarantined from this run.
for folder in sorted((ROOT/'DuLieuGoc').iterdir()):
 if not folder.is_dir():continue
 label=CLASS_MAP.get(normalize_name(folder.name))
 if not label:continue
 for path in sorted(folder.rglob('*')):
  if path.is_file() and path.suffix.lower() in EXTENSIONS:add(path.read_bytes(),label,'original',str(path.relative_to(ROOT)))
print('Original curated:',collections.Counter(r['label'] for r in records),flush=True)
label_map={'Tea Algal Spot':'algal_leaf','Tea Brown Blight':'brown_blight','Tea Grey Blight':'gray_light','Tea Healthy':'healthy','Tea Red Spot':'red_leaf_spot'}
with zipfile.ZipFile(args.cache_dir/'tea-compressed.zip') as archive:
 for name in sorted(archive.namelist()):
  parts=Path(name).parts
  if len(parts)>=2 and Path(name).suffix.lower() in EXTENSIONS and parts[-2] in label_map:add(archive.read(name),label_map[parts[-2]],'mendeley_rrff_v2',name)
print('After Mendeley:',collections.Counter(r['label'] for r in records),flush=True)
# Use only explicitly healthy tea labels from the second dataset; do not map its combined blight class.
table=pq.read_table(args.cache_dir/'tea-bangladesh.parquet')
healthy=[r for r in table.to_pylist() if r['label']==0];rng.shuffle(healthy)
added=0
for row in healthy:
 if added>=args.healthy_extra:break
 image=row['image']
 if add(image['bytes'],'healthy','tea_bangladesh_v4',image['path'],healthy_filter=True):added+=1
print('New independent-looking healthy views accepted:',added,flush=True)
kept=[]
for record in records:
 if record['id'] in conflicts:
  excluded.append({'source':record['source'],'original':record['original'],'label':record['label'],'reason':'conflicting_exact_label_quarantine'});Path(record['temporary']).unlink()
 else:kept.append(record)
records=kept
# Put near-identical image families into a single split, even across different labels.
parent=list(range(len(records)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
hashes=[int(r['phash'],16) for r in records]
for i in range(len(records)):
 for j in range(i):
  if (hashes[i]^hashes[j]).bit_count()<=4:
   a,b=find(i),find(j)
   if a!=b:parent[a]=b
for i,r in enumerate(records):r['group']=records[find(i)]['id']
groups=[r['group'] for r in records]
strata=[r['label']+'|'+r['source'] for r in records]
folds=np.zeros(len(records),dtype=int)
for fold,(_,indices) in enumerate(StratifiedGroupKFold(n_splits=20,shuffle=True,random_state=42).split(np.zeros(len(records)),strata,groups)):
 folds[indices]=fold
summary={split:collections.Counter() for split in ['train','val','test']}
for r,fold in zip(records,folds):
 split='train' if fold<14 else 'val' if fold<17 else 'test'
 r['split']=split;r['path']=f"{split}/{r['label']}/{Path(r['temporary']).name}"
 target=OUT/r['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.move(r.pop('temporary'),target)
 summary[split][r['label']]+=1
shutil.rmtree(OUT/'staging',ignore_errors=True)
counts=collections.Counter(r['label'] for r in records)
assert counts['healthy']>max(counts[k] for k in CLASSES if k!='healthy')
assert all(summary[s][k]>0 for s in summary for k in CLASSES)
for split in summary:
 others={r['group'] for r in records if r['split']!=split}
 assert not ({r['group'] for r in records if r['split']==split}&others)
report={'seed':42,'target_ratios':{'train':0.7,'val':0.15,'test':0.15},'split_method':'20-fold stratified group assignment: 14 train / 3 validation / 3 test folds','group_rule':'rotation/mirror-canonical green-region pHash Hamming <=4; same group stays in one split','healthy_extra_requested':args.healthy_extra,'healthy_extra_accepted':added,'total_images':len(records),'counts':dict(counts),'split_counts':{s:dict(v) for s,v in summary.items()},'actual_ratios':{s:sum(v.values())/len(records) for s,v in summary.items()},'groups':len(set(groups)),'largest_group':max(collections.Counter(groups).values()),'excluded_count':len(excluded),'exact_conflict_ids':sorted(conflicts),'sources':json.loads((args.cache_dir/'sources.json').read_text())}
(OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
(OUT/'split_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
(OUT/'exclusions.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2))
(OUT/'classes.json').write_text(json.dumps(CLASSES,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
