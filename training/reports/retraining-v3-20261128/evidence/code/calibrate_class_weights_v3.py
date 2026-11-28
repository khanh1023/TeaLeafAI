"""Select a small training-weight correction on validation, then evaluate test.

Beta grid is fixed to 0,.25,.5,.75,1. Use the smallest validation cross-entropy
subject to nondecreasing validation Accuracy and macro F1. This does not change
dataset partitions or fit CNN weights. Earlier test reports remain archived.
"""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import argparse,datetime,hashlib,json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Use a new output directory.')
    report=json.loads((args.initial/'evaluation.json').read_text())
    classes=json.loads((args.data/'classes.json').read_text())
    records=json.loads((args.data/'manifest.json').read_text())
    assert report['classes']==classes and report['dataset_manifest_sha256']==sha(args.data/'manifest.json')
    assert report['model_sha256']==sha(args.initial/'tea_leaf_mobilenetv2.keras')
    val=[r for r in records if r['split']=='val'];cached=np.load(args.initial/'validation-candidates.npz')
    assert list(cached['image_ids'])==[r['id'] for r in val]
    assert report['inference_protocol']['selected']=='four_flip_mean'
    probs=cached['four_flip_mean'];truth=cached['truth']
    weights=np.array([report['training']['class_weights'][c] for c in classes],dtype=np.float32)
    args.output.mkdir(parents=True)
    plan=dict(dataset_manifest_sha256=report['dataset_manifest_sha256'],base_model_sha256=report['model_sha256'],
              betas=[0,.25,.5,.75,1],weights=weights.tolist(),
              selection='Minimum validation cross-entropy among fixed beta grid, requiring Accuracy and macro-F1 not below beta=0. No test used for this selection.',
              formula='q[c] = (p[c] / training_class_weight[c]**beta) / sum_j(p[j] / training_class_weight[j]**beta)',
              earlier_internal_test_already_reported=True,script_sha256=sha(Path(__file__)))
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    def adjust(p,beta):
        q=p/np.power(weights,beta);return q/q.sum(axis=1,keepdims=True)
    def metrics(p):
        return dict(loss=float(-np.log(np.maximum(p[np.arange(len(truth)),truth],1e-7)).mean()),accuracy=float(accuracy_score(truth,p.argmax(1))),macro_f1=float(f1_score(truth,p.argmax(1),average='macro')))
    candidates=[dict(beta=b,**metrics(adjust(probs,b))) for b in plan['betas']]
    baseline=candidates[0]
    feasible=[r for r in candidates if r['accuracy']>=baseline['accuracy'] and r['macro_f1']>=baseline['macro_f1']]
    chosen=min(feasible,key=lambda r:(r['loss'],r['beta']))
    if chosen['loss']>=baseline['loss']-1e-4:chosen=baseline
    beta=chosen['beta']
    protocol=dict(plan,selected_beta=beta,candidates=candidates,selected_validation=chosen,
                  selected_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    # Freeze the validation-only choice before reading test predictions/labels.
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    model=tf.keras.models.load_model(args.initial/'tea_leaf_mobilenetv2.keras',compile=False)
    if beta:
        raw=tf.keras.Input((224,224,3),name='images')
        scaled=tf.keras.layers.Rescaling(scale=np.power(weights,-beta).tolist(),name='training_weight_correction')(model(raw,training=False))
        output=tf.keras.ops.divide(scaled,tf.keras.ops.sum(scaled,axis=-1,keepdims=True))
        model=tf.keras.Model(raw,output,name='tea_leaf_mobilenetv2_weight_corrected')
    saved=args.output/'tea_leaf_mobilenetv2.keras';model.save(saved)
    model=tf.keras.models.load_model(saved,compile=False)
    # Verify the serialized graph against the selected probability transformation
    # using validation images only. Full independent native/ONNX parity follows.
    adjusted_val=adjust(probs,beta)
    for i in [next(i for i,r in enumerate(val) if r['label']==c) for c in classes]:
        with Image.open(args.data/val[i]['path']) as im:x=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)[None]
        actual=model(x,training=False).numpy()[0]
        assert np.max(np.abs(actual-adjusted_val[i]))<1e-5
    test=[r for r in records if r['split']=='test'];base_test=np.load(args.initial/'test_predictions.npz')
    assert list(base_test['image_ids'])==[r['id'] for r in test]
    probabilities=adjust(base_test['probabilities'],beta);actual_truth=base_test['truth'];pred=probabilities.argmax(1)
    prior_metrics={k:report[k] for k in ['accuracy','balanced_accuracy','macro_f1','test_images']}
    report=dict(report);report['calibration_protocol']=protocol;report['pre_calibration_test_metrics']=prior_metrics
    report.update(model_sha256=sha(saved),accuracy=float(accuracy_score(actual_truth,pred)),balanced_accuracy=float(balanced_accuracy_score(actual_truth,pred)),
                  macro_f1=float(f1_score(actual_truth,pred,average='macro')),weighted_f1=float(f1_score(actual_truth,pred,average='weighted')),
                  classification_report=classification_report(actual_truth,pred,target_names=classes,labels=list(range(8)),output_dict=True,zero_division=0),
                  confusion_matrix=confusion_matrix(actual_truth,pred,labels=list(range(8))).tolist(),
                  target_met=bool(accuracy_score(actual_truth,pred)>.90),trained_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['limitations']=report['limitations']+['A scalar correction of recorded training class weights was selected on validation. This is an additional exploratory development stage after previous test reports, not a fresh blind evaluation.']
    (args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.output/'classes.json').write_text(json.dumps(classes,indent=2))
    np.savez_compressed(args.output/'test_predictions.npz',truth=actual_truth,probabilities=probabilities,image_ids=base_test['image_ids'])
    np.savez_compressed(args.output/'validation-predictions.npz',truth=truth,probabilities=adjusted_val,image_ids=cached['image_ids'])
    print(json.dumps(dict(beta=beta,validation=chosen,accuracy=report['accuracy'],macro_f1=report['macro_f1'],target_met=report['target_met']),indent=2),flush=True)


if __name__=='__main__':main()
