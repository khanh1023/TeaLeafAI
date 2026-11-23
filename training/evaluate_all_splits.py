"""Audit a frozen checkpoint on train/val/test/all and predefined sensitivity probes."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'training/.cache/matplotlib'))
import argparse,collections,csv,datetime,hashlib,io,json,platform,time
import numpy as np
from PIL import Image,ImageEnhance,ImageFilter,__version__ as pillow_version
import tensorflow as tf
import onnxruntime as ort
from reevaluate_expanded import metrics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--robustness',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    mpath=ROOT/'training/reports/expanded-dataset-manifest.json'
    records=json.loads(mpath.read_text());classes=json.loads((ROOT/'backend/models/classes.json').read_text())
    web=json.loads((ROOT/'frontend/public/model/manifest.json').read_text())
    assert classes==web['classes'] and sha(mpath)==web['dataset_manifest_sha256']
    for field in ['group','raw_sha256','pixel_sha256','sha256']:
        index=collections.defaultdict(set)
        for r in records:index[r[field]].add(r['split'])
        assert all(len(s)==1 for s in index.values()),field
    assert len({r['path'] for r in records})==len(records)
    data=ROOT/'training/dataset'
    for r in records:
        path=(data/r['path']).resolve()
        assert path.is_relative_to(data.resolve()) and sha(path)==r['sha256'],r['path']
    kp=ROOT/'backend/models/tea_leaf_mobilenetv2.keras';op=ROOT/'frontend/public/model/tea-leaf.onnx'
    ksha,osha=sha(kp),sha(op);assert osha==web['sha256']
    print(f'Verified all {len(records)} images and frozen split/model integrity.',flush=True)
    model=tf.keras.models.load_model(kp,compile=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(op),opts,providers=['CPUExecutionProvider'])
    def load(r):
        with Image.open(data/r['path']) as im:return im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
    def predict_onnx(image):
        return session.run(['probabilities'],{'images':np.asarray(image,dtype=np.float32)[None]})[0][0]
    results={};all_entries=[];all_k=[];all_o=[];all_y=[]
    for split in ['train','val','test']:
        entries=[r for r in records if r['split']==split]
        kb=[];ob=[]
        for offset in range(0,len(entries),16):
            images=[load(r) for r in entries[offset:offset+16]]
            arr=np.stack([np.asarray(im,dtype=np.float32) for im in images])
            kb.append(model(arr,training=False).numpy())
            ob.extend(predict_onnx(im) for im in images)
        k=np.concatenate(kb);o=np.stack(ob);y=np.array([classes.index(r['label']) for r in entries])
        results[split]={'keras':metrics(y,k,classes),'onnx':metrics(y,o,classes),
                        'top1_disagreements':int((k.argmax(1)!=o.argmax(1)).sum()),
                        'max_probability_error':float(np.max(np.abs(k-o)))}
        all_entries.extend(entries);all_k.append(k);all_o.append(o);all_y.append(y)
        print(split,len(entries),'Keras Accuracy',results[split]['keras']['accuracy'],'ONNX',results[split]['onnx']['accuracy'],flush=True)
    k,o,y=np.concatenate(all_k),np.concatenate(all_o),np.concatenate(all_y)
    results['all']={'keras':metrics(y,k,classes),'onnx':metrics(y,o,classes),
                    'includes_training_images':True,'interpretation':'Descriptive pooled score, not held-out generalization.'}
    np.savez_compressed(args.output/'native-predictions.npz',truth=y,keras=k,onnx=o,
                        ids=np.array([r['id'] for r in all_entries]),splits=np.array([r['split'] for r in all_entries]))
    with (args.output/'native-predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['id','split','path','source','actual','keras','onnx','keras_correct','onnx_correct','top_score'])
        for i,r in enumerate(all_entries):
            a,b=int(k[i].argmax()),int(o[i].argmax())
            writer.writerow([r['id'],r['split'],r['path'],r['source'],r['label'],classes[a],classes[b],a==y[i],b==y[i],float(o[i,b])])
    train_counts=collections.Counter(r['label'] for r in records if r['split']=='train')
    majority=max(classes,key=lambda c:train_counts[c])
    test_entries=[r for r in records if r['split']=='test'];ty=np.array([classes.index(r['label']) for r in test_entries])
    baseline=np.zeros((len(ty),len(classes)),dtype=np.float32);baseline[:,classes.index(majority)]=1
    baseline_result=metrics(ty,baseline,classes)
    # Define all probes before measuring any transformed accuracy; do not select the best.
    probe_defs=[('brightness_065','Độ sáng ×0,65'),('brightness_135','Độ sáng ×1,35'),
                ('blur_2','Mờ Gaussian, bán kính 2 px'),('resolution_32','Giảm về 32 × 32 rồi phóng lại'),
                ('jpeg_35','Nén JPEG quality 35'),('rotation_15','Xoay +15°')]
    robust=[];robust_rows=[]
    if args.robustness:
        images=[load(r) for r in test_entries]
        for key,label in probe_defs:
            probabilities=[]
            for r,im in zip(test_entries,images):
                if key=='brightness_065':altered=ImageEnhance.Brightness(im).enhance(.65)
                elif key=='brightness_135':altered=ImageEnhance.Brightness(im).enhance(1.35)
                elif key=='blur_2':altered=im.filter(ImageFilter.GaussianBlur(2))
                elif key=='resolution_32':altered=im.resize((32,32),Image.Resampling.BILINEAR).resize((224,224),Image.Resampling.BILINEAR)
                elif key=='jpeg_35':
                    buffer=io.BytesIO();im.save(buffer,'JPEG',quality=35)
                    with Image.open(io.BytesIO(buffer.getvalue())) as dec:altered=dec.convert('RGB')
                else:altered=im.rotate(15,resample=Image.Resampling.BILINEAR,fillcolor=im.getpixel((0,0)))
                pr=predict_onnx(altered);probabilities.append(pr)
                robust_rows.append([key,r['id'],r['label'],classes[int(pr.argmax())],classes[int(pr.argmax())]==r['label']])
            data_result=metrics(ty,np.stack(probabilities),classes)
            robust.append({'condition':key,'label':label,'metrics':data_result,
                           'delta_accuracy_percentage_points':(data_result['accuracy']-results['test']['onnx']['accuracy'])*100})
            print('probe',key,data_result['accuracy'],flush=True)
        with (args.output/'robustness-predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f,lineterminator='\n');writer.writerow(['condition','id','actual','predicted','correct']);writer.writerows(robust_rows)
    assert sha(kp)==ksha and sha(op)==osha and sha(mpath)==web['dataset_manifest_sha256']
    result={'evaluated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'model_version':web['model_version'],'classes':classes,'model_hashes':{'keras':ksha,'onnx':osha},
            'model_bytes':{'keras':kp.stat().st_size,'onnx':op.stat().st_size},
            'dataset_manifest_sha256':sha(mpath),'all_image_sha256_verified':True,
            'groups_and_exact_hashes_disjoint':True,'retrained':False,'split_changed':False,
            'preprocessing':'Pillow RGB bilinear 224x224; float32 0–255; Rescaling inside model; no augmentation for split scores.',
            'software':{'python':platform.python_version(),'tensorflow':tf.__version__,'onnxruntime':ort.__version__,
                        'numpy':np.__version__,'pillow':pillow_version},
            'splits':results,'majority_baseline':{'selected_from':'train label counts','class':majority,'test_metrics':baseline_result},
            'robustness':{'engine':'ONNX CPU','fixed_test_images':len(ty),'baseline':results['test']['onnx'],
                          'predefined_conditions':probe_defs,'conditions':robust,
                          'scope':'Single deterministic artificial perturbation at a time; not an independent field trial or causal explanation of original errors.'},
            'elapsed_seconds':time.monotonic()-started,
            'limitations':['All includes train images; test score is the main held-out measure.',
                           'Repeated internal test used in development; not externally field validated.',
                           'Source disease labels inherited; incomplete plant/session identities.',
                           'Robustness probes are diagnostic only; no checkpoint selected or tuned using them.']}
    (args.output/'native-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished. Test=',results['test']['keras']['accuracy'],'All descriptive=',results['all']['keras']['accuracy'],flush=True)


if __name__=='__main__':main()
