"""Export TeaLeafAI Keras model to browser-compatible ONNX and verify parity."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',type=Path,default=ROOT/'backend/models/tea_leaf_mobilenetv2.keras')
parser.add_argument('--classes',type=Path,default=ROOT/'backend/models/classes.json')
parser.add_argument('--output',type=Path,default=ROOT/'frontend/public/model')
args=parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)
# Model includes its own Rescaling layer. Keep exported input in raw RGB units.
model=tf.keras.models.load_model(args.model,compile=False)
print('Model input',model.input_shape,'output',model.output_shape,flush=True)

@tf.function(input_signature=[tf.TensorSpec([1,224,224,3],tf.float32,name='images')])
def inference(images):
    return {'probabilities': model(images,training=False)}

onnx_path=args.output/'tea-leaf.onnx'
tf2onnx.convert.from_function(inference,input_signature=inference.input_signature,opset=17,output_path=str(onnx_path))
print('Exported',onnx_path,flush=True)
options=ort.SessionOptions()
options.intra_op_num_threads=2
options.inter_op_num_threads=2
session=ort.InferenceSession(str(onnx_path),sess_options=options,providers=['CPUExecutionProvider'])
print('ONNX input',[(i.name,i.shape) for i in session.get_inputs()],flush=True)
inputs=[np.random.default_rng(42).uniform(0,255,(1,224,224,3)).astype(np.float32)]
source=ROOT/'DuLieuGoc'
for folder in sorted(source.iterdir()) if source.exists() else []:
    if folder.is_dir():
        sample=next((p for p in folder.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}),None)
        if sample:
            with Image.open(sample) as image:
                inputs.append(np.asarray(image.convert('RGB').resize((224,224)),dtype=np.float32)[None])
max_difference=0
for idx,array in enumerate(inputs):
    expected=model(array,training=False).numpy()
    actual=session.run(None,{session.get_inputs()[0].name:array})[0]
    delta=float(np.max(np.abs(expected-actual)))
    print('Parity',idx,'max error',delta,'Keras',int(expected.argmax()),'ONNX',int(actual.argmax()),flush=True)
    assert delta<1e-4, delta
    assert int(expected.argmax())==int(actual.argmax())
    max_difference=max(max_difference,delta)
classes=json.loads(args.classes.read_text())
(args.output/'classes.json').write_text(json.dumps(classes,indent=2)+'\n')
manifest={
 'format':'onnx','file':onnx_path.name,'input_name':session.get_inputs()[0].name,
 'output_name':session.get_outputs()[0].name,'input_shape':[1,224,224,3],
 'pixel_range':[0,255],'normalization':'Embedded in model; do not normalize input twice.',
 'classes':classes,'sha256':hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
 'size_bytes':onnx_path.stat().st_size,'parity_max_absolute_error':max_difference,
 'parity_samples':len(inputs),
}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2),flush=True)
