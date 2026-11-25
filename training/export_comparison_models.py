"""Export all three completed comparison checkpoints to browser ONNX, with parity checks."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import argparse
import hashlib
import json
import sys
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from training.compare_models import CaffePreprocessing, CLASSES, MODELS, image_array, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'frontend/public/model/comparison')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    protocol=json.loads((args.run/'protocol.json').read_text())
    records=json.loads((ROOT/'training/reports/expanded-dataset-manifest.json').read_text())
    assert sha(ROOT/'training/reports/expanded-dataset-manifest.json')==protocol['config']['dataset_manifest_sha256']
    test=[(i,r) for i,r in enumerate(records) if r['split']=='test']
    indices=set(np.linspace(0,len(test)-1,16,dtype=int).tolist())
    indices.update(next(i for i,(_,r) in enumerate(test) if r['label']==c) for c in CLASSES)
    selection=sorted(indices)
    arrays=[image_array(ROOT/'training/dataset',test[j][1],test[j][0],0,42)[None] for j in selection]
    args.output.mkdir(parents=True,exist_ok=True)
    args.report.mkdir(parents=True,exist_ok=True)
    exported=[]
    for name in MODELS:
        tf.keras.backend.clear_session()
        checkpoint=args.run/name/'model.keras'
        evaluation=json.loads((args.run/name/'evaluation.json').read_text())
        assert sha(checkpoint)==evaluation['model_sha256']
        model=tf.keras.models.load_model(checkpoint,compile=False)
        key=name.lower()
        target=args.output/(key+'.onnx')

        @tf.function(input_signature=[tf.TensorSpec([1,224,224,3],tf.float32,name='images')])
        def inference(images):
            return {'probabilities':model(images,training=False)}

        print('Exporting',name,flush=True)
        tf2onnx.convert.from_function(inference,input_signature=inference.input_signature,
                                     opset=17,output_path=str(target))
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(target),options,providers=['CPUExecutionProvider'])
        input_name=session.get_inputs()[0].name;output_name=session.get_outputs()[0].name
        maximum=0.;disagreements=0;parity=[]
        for index,array in zip(selection,arrays):
            expected=model(array,training=False).numpy()
            actual=session.run([output_name],{input_name:array})[0]
            delta=float(np.max(np.abs(expected-actual)));maximum=max(maximum,delta)
            disagreements+=int(expected.argmax()!=actual.argmax())
            parity.append({'id':test[index][1]['id'],'max_absolute_error':delta,
                           'keras_class':CLASSES[int(expected.argmax())],
                           'onnx_class':CLASSES[int(actual.argmax())]})
        assert maximum<1e-3 and disagreements==0,(name,maximum,disagreements)
        assert target.stat().st_size<100*1024*1024,'Model exceeds GitHub single-file limit.'
        item={'id':key,'name':name,'file':target.name,'format':'onnx',
            'model_version':f'{key}-comparison-20261125-{evaluation["model_sha256"][:10]}',
            'input_name':input_name,'output_name':output_name,'input_shape':[1,224,224,3],
            'size_bytes':target.stat().st_size,'sha256':sha(target),'keras_sha256':evaluation['model_sha256'],
            'total_parameters':evaluation['total_parameters'],
            'parity_samples':len(parity),'parity_max_absolute_error':maximum,'parity_top1_disagreements':disagreements,
            'native_test':{k:evaluation[k] for k in ['accuracy','macro_f1','correct','test_images']}}
        exported.append(item)
        (args.report/(key+'-export-parity.json')).write_text(json.dumps({'model':item,'samples':parity},indent=2)+'\n')
        print(name,'bytes:',item['size_bytes'],'max parity error:',maximum,flush=True)
        del session,model
    signature=hashlib.sha256(''.join(item['sha256'] for item in exported).encode()).hexdigest()[:16]
    catalog={'protocol_id':f'comparison-frozen-20261125-seed42-{signature}','classes':CLASSES,
        'dataset_manifest_sha256':protocol['config']['dataset_manifest_sha256'],
        'training':'Official ImageNet backbone frozen; identical head and budget; same group70/15/15 split.',
        'preprocessing':'Raw RGB float32 0–255,224x224; architecture-specific normalization embedded in each model.',
        'models':exported}
    (args.output/'manifest.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    (args.report/'browser-model-catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    print('All three models exported and verified.',flush=True)


if __name__=='__main__':main()
