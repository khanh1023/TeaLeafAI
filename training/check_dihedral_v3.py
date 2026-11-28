"""Check full eight-view symmetry on validation; never reads test images.

Keep the existing four-view model unless validation Accuracy improves, macro F1
does not decrease, and cross-entropy improves. This is an exploratory internal
validation experiment after preceding development-test results were reported.
"""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import argparse,hashlib,json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Use a new output directory.')
    args.output.mkdir(parents=True)
    report=json.loads((args.initial/'evaluation.json').read_text())
    classes=json.loads((args.data/'classes.json').read_text())
    records=json.loads((args.data/'manifest.json').read_text())
    assert hashlib.sha256((args.data/'manifest.json').read_bytes()).hexdigest()==report['dataset_manifest_sha256']
    assert hashlib.sha256((args.initial/'tea_leaf_mobilenetv2.keras').read_bytes()).hexdigest()==report['model_sha256']
    val=[r for r in records if r['split']=='val']
    cached=np.load(args.initial/'validation-candidates.npz')
    assert list(cached['image_ids'])==[r['id'] for r in val]
    assert report['inference_protocol']['selected']=='four_flip_mean'
    truth=cached['truth'];four=cached['four_flip_mean']
    plan=dict(dataset_manifest_sha256=report['dataset_manifest_sha256'],base_model_sha256=report['model_sha256'],
              alternatives=['four_flip_mean','eight_dihedral_mean'],
              selection='Adopt eight views only if validation Accuracy strictly improves, macro-F1 does not decrease and cross-entropy decreases by more than 1e-4. Otherwise retain four views.',
              previous_development_test_already_reported=True,test_images_read=0)
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    model=tf.keras.models.load_model(args.initial/'tea_leaf_mobilenetv2.keras',compile=False)
    forward=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec([None,224,224,3],tf.float32)])
    blocks=[]
    for offset in range(0,len(val),32):
        arrays=[]
        for r in val[offset:offset+32]:
            with Image.open(args.data/r['path']) as im:arrays.append(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32))
        x=np.stack(arrays).transpose(0,2,1,3).copy()
        blocks.append(forward(x).numpy())
    eight=(four+np.concatenate(blocks))*.5
    def metrics(probs):
        return dict(loss=float(-np.log(np.maximum(probs[np.arange(len(truth)),truth],1e-7)).mean()),accuracy=float(accuracy_score(truth,probs.argmax(1))),macro_f1=float(f1_score(truth,probs.argmax(1),average='macro')))
    a,b=metrics(four),metrics(eight)
    adopt=b['accuracy']>a['accuracy'] and b['macro_f1']>=a['macro_f1'] and b['loss']<a['loss']-1e-4
    result=dict(plan,four_flip_mean=a,eight_dihedral_mean=b,adopt_eight=adopt)
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output/'validation-predictions.npz',truth=truth,four_flip_mean=four,eight_dihedral_mean=eight,image_ids=cached['image_ids'])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
