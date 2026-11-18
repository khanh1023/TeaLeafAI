"""Download pinned, CC BY 4.0 tea image sources and verify SHA-256."""
from pathlib import Path
import argparse,hashlib,json
import requests

SOURCES=[
 {'id':'rrff4vb35m-v2','file':'tea-compressed.zip','url':'https://data.mendeley.com/public-files/datasets/rrff4vb35m/files/db2e8fe7-0f9f-419e-b1c7-6eaf3564a518/file_downloaded','sha256':'90e694b5eb3d0ad22ea018d578f39775a94ea9ab919172e8ea627bc12d49215b','dataset_url':'https://data.mendeley.com/datasets/rrff4vb35m/2','citation':'Billah, Sagor, Sultana & Uddin (2025), DOI 10.17632/rrff4vb35m.2','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/'},
 {'id':'tea-bangladesh-v4','file':'tea-bangladesh.parquet','url':'https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification_bangladesh/resolve/eb53d418d10a4bdf8cbab95928aad905aa62a368/data/train-00000-of-00001.parquet','sha256':'67fe2cc3aa6ed34e804ee9c7da9a6e39327eef2c33275905aeeb20abf5690613','dataset_url':'https://data.mendeley.com/datasets/tt2smzrzrs/4','mirror':'https://huggingface.co/datasets/Project-AgML/tea_leaf_disease_classification_bangladesh','citation':'Ahmad (2024), DOI 10.17632/tt2smzrzrs.4; Project-AgML mirror','license':'CC BY 4.0','license_url':'https://creativecommons.org/licenses/by/4.0/'}
]

def digest_file(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cache-dir',type=Path,required=True);args=parser.parse_args();args.cache_dir.mkdir(parents=True,exist_ok=True)
 for item in SOURCES:
  target=args.cache_dir/item['file']
  if not target.exists():
   partial=target.with_suffix(target.suffix+'.part')
   with requests.get(item['url'],stream=True,timeout=120) as response:
    response.raise_for_status()
    with partial.open('wb') as f:
     for chunk in response.iter_content(1024*1024):f.write(chunk)
   if digest_file(partial)!=item['sha256']:partial.unlink();raise RuntimeError('Downloaded source does not match pinned checksum.')
   partial.replace(target)
  if digest_file(target)!=item['sha256']:raise RuntimeError(f'Cached source checksum mismatch: {target}')
  print(item['id'],target.stat().st_size,'bytes; SHA-256 verified.',flush=True)
 (args.cache_dir/'sources.json').write_text(json.dumps(SOURCES,indent=2))
if __name__=='__main__':main()
