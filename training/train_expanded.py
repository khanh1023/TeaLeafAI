"""Train a fresh MobileNetV2 head on the frozen 70/15/15 group split; never augment validation/test."""
import os
from pathlib import Path
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.cache/matplotlib'))
os.environ.setdefault('KERAS_HOME',str(Path(__file__).resolve().parent/'.cache/keras'))
import argparse,collections,hashlib,json,random,time,datetime
import numpy as np
from PIL import Image,ImageEnhance,ImageOps
import tensorflow as tf
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,f1_score

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--epochs',type=int,default=20)
p.add_argument('--augmented-views',type=int,default=2)
args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
DATA=ROOT/'training/dataset'
manifest_bytes=(DATA/'manifest.json').read_bytes();records=json.loads(manifest_bytes)
classes=json.loads((DATA/'classes.json').read_text());summary=json.loads((DATA/'split_summary.json').read_text())
SEED=42;tf.keras.utils.set_random_seed(SEED)
started=time.time()
base=tf.keras.applications.MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
base.trainable=False
raw=tf.keras.Input((224,224,3),name='images')
x=tf.keras.layers.Rescaling(1/127.5,offset=-1,name='pixel_rescaling')(raw)
x=base(x,training=False)
x=tf.keras.layers.GlobalAveragePooling2D(name='leaf_features')(x)
extractor=tf.keras.Model(raw,x,name='tea_mobilenet_features')
print('ImageNet backbone loaded. Extracting frozen features.',flush=True)

def image_array(record,index,view):
 with Image.open(DATA/record['path']) as image:
  image=image.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
  if view:
   rng=random.Random(SEED+index+view*100000)
   if rng.random()<0.5:image=ImageOps.mirror(image)
   angle=rng.uniform(-15,15)
   corner=tuple(int(v) for v in np.asarray(image)[0,0])
   image=image.rotate(angle,resample=Image.Resampling.BILINEAR,fillcolor=corner)
   image=ImageEnhance.Brightness(image).enhance(rng.uniform(0.85,1.15))
   image=ImageEnhance.Contrast(image).enhance(rng.uniform(0.85,1.15))
  return np.asarray(image,dtype=np.float32)

features={};targets={}
for split in ['train','val','test']:
 entries=[(i,r,view) for i,r in enumerate(records) if r['split']==split for view in range(args.augmented_views+1 if split=='train' else 1)]
 blocks=[];labels=[]
 for offset in range(0,len(entries),32):
  batch=entries[offset:offset+32]
  arrays=np.stack([image_array(r,i,view) for i,r,view in batch])
  blocks.append(extractor(arrays,training=False).numpy())
  labels.extend(classes.index(r['label']) for _,r,_ in batch)
  if offset%320==0:print(f'Features {split}: {min(offset+32,len(entries))}/{len(entries)}',flush=True)
 features[split]=np.concatenate(blocks);targets[split]=np.array(labels,dtype=np.int32)
np.savez_compressed(args.output/'features.npz',**{f'x_{s}':v for s,v in features.items()},**{f'y_{s}':v for s,v in targets.items()})
head_in=tf.keras.Input((1280,))
h=tf.keras.layers.Dropout(0.3)(head_in)
h=tf.keras.layers.Dense(128,activation='relu')(h)
h=tf.keras.layers.Dropout(0.2)(h)
out=tf.keras.layers.Dense(len(classes),activation='softmax')(h)
head=tf.keras.Model(head_in,out,name='tea_classifier')
head.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
counts=collections.Counter(targets['train'].tolist())
weights={k:len(targets['train'])/(len(classes)*v) for k,v in counts.items()}
options=tf.data.Options();options.threading.private_threadpool_size=2;options.threading.max_intra_op_parallelism=1
train=tf.data.Dataset.from_tensor_slices((features['train'],targets['train'])).shuffle(len(targets['train']),seed=SEED).batch(32).with_options(options).prefetch(1)
val=tf.data.Dataset.from_tensor_slices((features['val'],targets['val'])).batch(64).with_options(options).prefetch(1)
callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,min_delta=1e-4,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.3,patience=2,min_lr=1e-6),tf.keras.callbacks.ModelCheckpoint(args.output/'best_head.keras',monitor='val_loss',save_best_only=True)]
print('Training new head; class weights:',weights,flush=True)
history=head.fit(train,validation_data=val,epochs=args.epochs,class_weight=weights,callbacks=callbacks,verbose=2)
head=tf.keras.models.load_model(args.output/'best_head.keras')
model=tf.keras.Model(raw,head(extractor.output),name='tea_leaf_mobilenetv2_expanded')
model.save(args.output/'tea_leaf_mobilenetv2.keras')
(args.output/'classes.json').write_text(json.dumps(classes,indent=2))
probs=head(features['test'],training=False).numpy();truth=targets['test'];pred=probs.argmax(axis=1)
report={'model':'MobileNetV2','num_classes':len(classes),'classes':classes,'accuracy':float(accuracy_score(truth,pred)),'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),'macro_f1':float(f1_score(truth,pred,average='macro')),'weighted_f1':float(f1_score(truth,pred,average='weighted')),'classification_report':classification_report(truth,pred,target_names=classes,labels=list(range(8)),output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(truth,pred,labels=list(range(8))).tolist(),'test_images':len(truth),'dataset_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'dataset':summary,'training':{'seed':SEED,'maximum_epochs':args.epochs,'epochs_completed':len(history.history['loss']),'selected_epoch':int(np.argmin(history.history['val_loss']))+1,'selection':'minimum validation loss; test unused until final evaluation','backbone':'fresh official ImageNet MobileNetV2, frozen','head':'Dropout .3 → Dense128 ReLU → Dropout .2 → Dense8 softmax','class_weights':{classes[k]:v for k,v in weights.items()},'augmentation':'two deterministic training-only views per original: horizontal flip, +/-15 degree rotation, brightness/contrast .85–1.15','feature_cache_equivalence':'Backbone is frozen; train classifier on cached features, then save the complete raw-RGB-input model.','history':{k:[float(x) for x in v] for k,v in history.history.items()}},'trained_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':time.time()-started,'limitations':['Internal held-out evaluation from the same source datasets; not an independent Vietnamese field trial.','Perceptual groups reduce duplicate leakage, but plant/session identifiers were not supplied for every source image.','Source labels are inherited from dataset publishers; conflicting exact duplicates are excluded rather than relabeled.','More healthy photos do not guarantee improved real-world specificity; per-class metrics are reported.']}
(args.output/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
np.savez_compressed(args.output/'test_predictions.npz',truth=truth,probabilities=probs)
print(json.dumps({k:report[k] for k in ['accuracy','balanced_accuracy','macro_f1','test_images','elapsed_seconds']},indent=2),flush=True)
