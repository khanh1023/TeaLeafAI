"""Fetch explicitly matching tea classes from pinned CC BY 4.0 dataset revisions.

The public dataset viewer serves individual images, avoiding multi-GB archives.
Only raw configurations are selected. Downloaded images are EXIF-corrected,
converted to RGB and bounded to 512 px; original byte hashes remain in receipts.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps
import requests

SOURCES = [
    dict(id='tealeafbd_v4', repo='Project-AgML/teaLeafBD_disease_classification',
         revision='068914a550e65791dcb488e4ef9a828d710bff3c', config='default',
         dataset_url='https://data.mendeley.com/datasets/744vznw5k2/4',
         citation='Ahammed et al. (2025), teaLeafBD, DOI 10.17632/744vznw5k2.4',
         labels={0: 'brown_blight', 1: 'gray_light', 3: 'healthy', 6: 'algal_leaf'},
         expected={0: 508, 1: 1013, 3: 935, 6: 418}),
    dict(id='habiganj_v1', repo='Project-AgML/tea_leaf_disease_classification_bangladesh_2',
         revision='cb677887ee7b52ee0b084e7a2e3dc8cd3ec989fd', config='default',
         dataset_url='https://data.mendeley.com/datasets/3x42rbj8yv/1',
         citation='Chowdhury & Arman (2025), DOI 10.17632/3x42rbj8yv.1',
         labels={1: 'healthy'}, expected={1: 650}, viewer_mode='rows',
         row_ranges=[(1850, 2500)]),
    dict(id='hibiscus_tea_raw_v4', repo='Project-AgML/Hibiscus_Tea_disease_classification',
         revision='995934b58fe3530bfdf8e9ec87689b12d6426165', config='raw',
         dataset_url='https://data.mendeley.com/datasets/5bzy89brkv/4',
         citation='Billah, Sagor, Uddin & Hossain (2025), DOI 10.17632/5bzy89brkv.4',
         labels={0: 'algal_leaf', 1: 'brown_blight', 5: 'gray_light', 8: 'red_leaf_spot'},
         expected={0: 54, 1: 48, 5: 53, 8: 44},
         parquet='raw/train-00000-of-00001.parquet',
         parquet_sha256='18582f2734821b7e0ee3f1e4a89dc8aa481b0cb425e4d3c153107f008b06d291'),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(url, **kwargs):
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=(15, 90), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def collect(source, cache, workers, refresh):
    if source.get('parquet'):
        return collect_parquet(source, cache)
    folder = cache / source['id']
    folder.mkdir(parents=True, exist_ok=True)
    mode = source.get('viewer_mode', 'filter')
    page_folder = folder / ('pages' if mode == 'filter' else 'rows-pages')
    page_folder.mkdir(exist_ok=True)
    params = dict(dataset=source['repo'], config=source['config'], split='train', length=100)
    if mode == 'filter':
        params['where'] = ' OR '.join(f'"label" = {k}' for k in source['labels'])
    total = sum(source['expected'].values())
    ranges = source.get('row_ranges', [(0, source.get('total_rows', total))])
    offsets = [offset for start, end in ranges for offset in range(start, end, 100)]

    def page(offset):
        path = page_folder / f'{offset:05}.json'
        if not path.exists() or refresh:
            data = get(f'https://datasets-server.huggingface.co/{mode}',
                       params={**params, 'offset': offset}).json()
            path.write_text(json.dumps(data))
        return json.loads(path.read_text())['rows']

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        pages = list(pool.map(page, offsets))
    rows = [r for block in pages for r in block if r['row']['label'] in source['labels']]
    actual = collections.Counter(r['row']['label'] for r in rows)
    if actual != source['expected']:
        raise RuntimeError(f"Source row counts changed: {source['id']}: {actual}")
    if len({r['row_idx'] for r in rows}) != len(rows):
        raise RuntimeError('Source viewer repeated row identities.')
    print(source['id'], 'eligible rows', len(rows), flush=True)

    def download(entry):
        row = entry['row']
        idx = entry['row_idx']
        image = row['image']
        if source['revision'] not in image['src']:
            raise RuntimeError('Viewer revision differs from pinned revision; do not silently update.')
        if source['config'] == 'raw' and 'tea' not in row.get('crop_type', '').lower():
            raise RuntimeError(f'Non-tea crop in selected class: {row.get("crop_type")}')
        receipt_path = folder / f'{idx:05}.json'
        output = folder / f'{idx:05}.jpg'
        if receipt_path.exists() and output.exists():
            receipt = json.loads(receipt_path.read_text())
            if sha(output.read_bytes()) == receipt['sha256']:
                return receipt
        raw = get(image['src']).content
        with Image.open(io.BytesIO(raw)) as opened:
            decoded = ImageOps.exif_transpose(opened).convert('RGB')
            decoded.load()
            size = list(decoded.size)
            if min(size) < 80:
                raise RuntimeError(f'Source image too small: {source["id"]}/{idx}')
            decoded.thumbnail((512, 512), Image.Resampling.LANCZOS)
            pixel_sha = sha(np.asarray(decoded).tobytes())
            buffer = io.BytesIO()
            decoded.save(buffer, 'JPEG', quality=92, subsampling=0)
        normalized = buffer.getvalue()
        output.write_bytes(normalized)
        receipt = dict(source=source['id'], source_row=idx, source_label=row['label'],
                       label=source['labels'][row['label']],
                       original=f"{source['repo']}@{source['revision']}:{source['config']}/train/row={idx}",
                       raw_sha256=sha(raw), pixel_sha256=pixel_sha,
                       sha256=sha(normalized), original_size=size, file=output.name)
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt

    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for receipt in pool.map(download, rows):
            receipts.append(receipt)
            if len(receipts) % 100 == 0:
                print(source['id'], len(receipts), '/', total, flush=True)
    (folder / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
    return dict(**source, license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
                eligible_downloaded=len(receipts), mirror=f"https://huggingface.co/datasets/{source['repo']}")


def collect_parquet(source, cache):
    """Pinned direct parquet fallback when the public viewer cannot decode rows."""
    import pyarrow.parquet as pq
    folder = cache / source['id']
    folder.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{source['parquet']}"
    downloaded = folder / 'source.parquet'
    if not downloaded.exists():
        partial = downloaded.with_suffix('.parquet.part')
        h = hashlib.sha256()
        with get(url, stream=True) as response, partial.open('wb') as file:
            for chunk in response.iter_content(1024 * 1024):
                file.write(chunk); h.update(chunk)
        if h.hexdigest() != source['parquet_sha256']:
            partial.unlink(); raise RuntimeError('Direct source parquet checksum mismatch.')
        partial.rename(downloaded)
    h = hashlib.sha256()
    with downloaded.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    if h.hexdigest() != source['parquet_sha256']:
        raise RuntimeError('Cached source parquet checksum mismatch.')
    receipts = []; excluded = []; seen = collections.Counter(); row_index = 0
    for batch in pq.ParquetFile(downloaded).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            idx = row_index; row_index += 1
            if row['label'] not in source['labels']:
                continue
            if 'tea' not in row.get('crop_type', '').lower():
                raise RuntimeError('Non-tea crop in selected raw tea class.')
            seen[row['label']] += 1
            raw = row['image']['bytes']
            try:
                with Image.open(io.BytesIO(raw)) as opened:
                    image = ImageOps.exif_transpose(opened).convert('RGB'); image.load()
                    size = list(image.size)
                    if min(size) < 80:
                        raise ValueError('image_too_small')
                    image.thumbnail((512, 512), Image.Resampling.LANCZOS)
                    pixel_sha = sha(np.asarray(image).tobytes())
                    buffer = io.BytesIO(); image.save(buffer, 'JPEG', quality=92, subsampling=0)
                content = buffer.getvalue()
            except (OSError, ValueError) as error:
                excluded.append(dict(source=source['id'], original=f'raw/train/row={idx}',
                                     label=source['labels'][row['label']], reason=f'cannot_decode_source_image:{error}'))
                continue
            output = folder / f'{idx:05}.jpg'; output.write_bytes(content)
            receipt = dict(source=source['id'], source_row=idx, source_label=row['label'],
                           label=source['labels'][row['label']],
                           original=f"{source['repo']}@{source['revision']}:{source['config']}/train/row={idx}",
                           source_original_path=row['image'].get('path'), raw_sha256=sha(raw),
                           pixel_sha256=pixel_sha, sha256=sha(content), original_size=size, file=output.name)
            (folder / f'{idx:05}.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
            receipts.append(receipt)
    if seen != source['expected']:
        raise RuntimeError(f'Source class counts changed: {seen}')
    (folder / 'receipts.json').write_text(json.dumps(receipts, ensure_ascii=False, indent=2))
    print(source['id'], len(receipts), 'valid raw tea images;', len(excluded), 'invalid', flush=True)
    downloaded.unlink()  # The normalized images and original-byte hashes remain; save workspace space.
    return dict(**source, license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
                eligible_downloaded=len(receipts), invalid_source_images=excluded, source_parquet_url=url,
                mirror=f"https://huggingface.co/datasets/{source['repo']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--workers', default=10, type=int)
    parser.add_argument('--refresh-pages', action='store_true')
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for source in SOURCES:
        metadata.append(collect(source, args.cache_dir, args.workers, args.refresh_pages))
        (args.cache_dir / 'sources-v2.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print('All selected source rows downloaded:', sum(s['eligible_downloaded'] for s in metadata), flush=True)


if __name__ == '__main__':
    main()
