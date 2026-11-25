"""Fine-tune only final MobileNetV2 layers, selecting by validation loss, not test accuracy."""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
import argparse,collections,hashlib,json,time,datetime,shutil
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=6);p.add_argument('--data',type=Path,default=ROOT/'training/dataset');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
DATA=args.data.resolve();manifest_bytes=(DATA/'manifest.json').read_bytes();records=json.loads(manifest_bytes);classes=json.loads((DATA/'classes.json').read_text())
initial_report=json.loads((args.initial/'evaluation.json').read_text());tf.keras.utils.set_random_seed(42);started=time.time()
if initial_report['dataset_manifest_sha256']!=hashlib.sha256(manifest_bytes).hexdigest():raise SystemExit('Initial checkpoint was trained on a different dataset split. Train a fresh head on this version first.')
if initial_report['classes']!=classes:raise SystemExit('Initial checkpoint class order differs from this dataset.')
model=tf.keras.models.load_model(args.initial/'tea_leaf_mobilenetv2.keras',compile=False)
backbone=next(layer for layer in model.layers if layer.name.startswith('mobilenetv2'))
backbone.trainable=True
for layer in backbone.layers:layer.trainable=False
for layer in backbone.layers[-30:]:
 if not isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
augmentation=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal_and_vertical',seed=42),tf.keras.layers.RandomRotation(.04,seed=43,fill_mode='reflect'),tf.keras.layers.RandomContrast(.1,seed=44),tf.keras.layers.RandomBrightness(.1,value_range=(0,255),seed=45)])
options=tf.data.Options();options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=1

def load_image(path,label):
 image=tf.io.decode_jpeg(tf.io.read_file(path),channels=3)
 return tf.image.resize(image,[224,224],method='bilinear'),label

def split(name,train=False):
 subset=[r for r in records if r['split']==name]
 paths=[str(DATA/r['path']) for r in subset];labels=[classes.index(r['label']) for r in subset]
 ds=tf.data.Dataset.from_tensor_slices((paths,labels)).map(load_image,num_parallel_calls=2).cache().with_options(options)
 if train:ds=ds.shuffle(len(subset),seed=42)
 ds=ds.batch(16)
 if train:ds=ds.map(lambda x,y:(augmentation(x,training=True),y),num_parallel_calls=1)
 return ds.prefetch(1)
train=split('train',True);val=split('val')
train_counts=collections.Counter(r['label'] for r in records if r['split']=='train');total=sum(train_counts.values());weights={classes.index(k):total/(8*v) for k,v in train_counts.items()}
base_loss,base_accuracy=model.evaluate(val,verbose=0);print('Validation before fine-tuning:',base_loss,base_accuracy,flush=True)
best=args.output/'tea_leaf_mobilenetv2.keras';model.save(best)
callbacks=[tf.keras.callbacks.ModelCheckpoint(best,monitor='val_loss',save_best_only=True,initial_value_threshold=base_loss),tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=1,min_lr=2e-7)]
print('Fine-tuning trainable parameters:',sum(int(np.prod(v.shape)) for v in model.trainable_weights),flush=True)
history=model.fit(train,validation_data=val,epochs=args.epochs,class_weight=weights,callbacks=callbacks,verbose=2)
model=tf.keras.models.load_model(best,compile=False)
entries=[r for r in records if r['split']=='test'];probabilities=[]
for offset in range(0,len(entries),16):
 arrays=[]
 for r in entries[offset:offset+16]:
  with Image.open(DATA/r['path']) as image:arrays.append(np.asarray(image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32))
 probabilities.append(model(np.stack(arrays),training=False).numpy())
probs=np.concatenate(probabilities);truth=np.array([classes.index(r['label']) for r in entries]);pred=probs.argmax(1)
report=initial_report.copy()
report.update({'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro')),'weighted_f1':float(f1_score(truth,pred,average='weighted')),'classification_report':classification_report(truth,pred,target_names=classes,labels=list(range(8)),output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(8))).tolist(),'trained_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':initial_report['elapsed_seconds']+time.time()-started})
report['initial_head_test_metrics']={k:initial_report[k] for k in ['accuracy','balanced_accuracy','macro_f1']}
report['training']['fine_tuning']={'maximum_epochs':args.epochs,'epochs_completed':len(history.history['loss']),'learning_rate_initial':1e-5,'layers':'last 30 layers of MobileNetV2, BatchNormalization frozen; backbone called with training=False','initial_validation_loss':base_loss,'initial_validation_accuracy':base_accuracy,'selected':'finetuned' if min(history.history['val_loss'])<base_loss else 'initial_head','selected_validation_loss':min(base_loss,min(history.history['val_loss'])),'selection':'minimum validation loss, with initial model kept unless validation improves','history':{k:[float(x) for x in v] for k,v in history.history.items()}}
report['limitations'].append('A preliminary frozen-head evaluation was performed on the same held-out test partition; final model selection used validation loss only. This is development evaluation, not external certification.')
report['dataset']=json.loads((DATA/'split_summary.json').read_text())
(args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(args.output/'classes.json').write_text(json.dumps(classes,indent=2));np.savez_compressed(args.output/'test_predictions.npz',truth=truth,probabilities=probs)
print(json.dumps({k:report[k] for k in ['accuracy','balanced_accuracy','macro_f1','test_images','elapsed_seconds']},indent=2),flush=True)
