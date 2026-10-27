from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

def load_training_configs(checkpoints, config_paths=None):
    if config_paths:
        paths = [Path(path) for path in config_paths]
        if len(paths) == 1:
            paths *= len(checkpoints)
        elif len(paths) != len(checkpoints):
            raise ValueError('Provide one shared configuration or one configuration per checkpoint.')
    else:
        paths = []
        for checkpoint in checkpoints:
            checkpoint = Path(checkpoint).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
            candidate = next(
                (parent / 'training_config.json' for parent in checkpoint.parents
                 if (parent / 'training_config.json').is_file()),
                None,
            )
            if candidate is None:
                raise FileNotFoundError(
                    f'No saved training_config.json found for {checkpoint}. '
                    'Keep this generated file in the training output directory, '
                    'or use --configs when loading archived or relocated checkpoints.'
                )
            paths.append(candidate)
    return [json.loads(path.expanduser().read_text(encoding='utf-8-sig')) for path in paths]

def checkpoint_config_hash(cfg):
    value=copy.deepcopy({k:v for k,v in cfg.items() if not k.startswith('_')})
    value.pop('cache',None)
    value.pop('reporting',None)
    revision=cfg.get('revision',{})
    if not (revision.get('resume_training') or revision.get('checkpoint_policy')=='fold_end'):
        for key in ('num_workers','prefetch_factor','persistent_workers','multiprocessing_context','pin_memory'):
            value.get('hardware',{}).pop(key,None)
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str,separators=(',',':')).encode()).hexdigest()

def validate_checkpoint_metadata(payload, cfg):
    if payload.get('model_key') != 'dctg_vmamba' or payload.get('config_sha256') != checkpoint_config_hash(cfg):
        raise ValueError('Checkpoint does not match the model and saved training configuration.')
    if any(payload.get(key) != cfg.get(key) for key in ('run_id', 'protocol_id')):
        raise ValueError('Checkpoint run/protocol mismatch.')
    fold = int(payload.get('fold', 0))
    if fold not in range(1, 6):
        raise ValueError('Checkpoint fold identity must be between 1 and 5.')
    return fold

def predict(configs,checkpoints,images,dataset=None,device_name='cuda'):
    import numpy as np
    import torch
    from PIL import Image,ImageOps
    from torchvision.transforms import functional as TF
    from bladder_project.v3_data import IMAGENET_MEAN,IMAGENET_STD
    from bladder_project.v3_models import create_study_model
    from bladder_project.v3_training import _effective_method_cfg
    if len(configs)!=len(checkpoints) or len(configs) not in (1,5):
        raise ValueError('Provide one checkpoint/config pair or five pairs for the fold ensemble.')
    cfg=configs[0]
    for other in configs[1:]:
        if any(other[key]!=cfg[key] for key in ('models','datasets','method')):
            raise ValueError('Model structure, source heads or loss configuration differ across folds.')
    sources=list(cfg['datasets'])
    if dataset is not None and dataset not in sources:
        raise ValueError('Requested fine-class head was not trained in this configuration.')
    selected=copy.deepcopy(next(m for m in cfg['models'] if m['key']=='dctg_vmamba'))
    device=torch.device(device_name)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable.')
    first_payload = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(first_payload, cfg)
    method=_effective_method_cfg(cfg['method'],selected)
    model=create_study_model(selected,{n:cfg['datasets'][n]['classes'] for n in sources},method).to(device)
    size=int(cfg['training']['image_size'])
    batch_size=int(cfg['training'].get('evaluation_batch_size',64))
    folds=[]
    seen=set()
    for index, (fold_cfg,path) in enumerate(zip(configs,checkpoints)):
        payload = first_payload if index == 0 else torch.load(path, map_location='cpu', weights_only=True)
        fold = validate_checkpoint_metadata(payload, fold_cfg)
        if fold in seen:
            raise ValueError('Checkpoint fold identities must be distinct and between 1 and 5.')
        seen.add(fold)
        model.load_state_dict(payload['model_state_dict'],strict=True)
        model.eval()
        values=[]
        with torch.inference_mode():
            for start in range(0,len(images),batch_size):
                tensors=[]
                for path in images[start:start+batch_size]:
                    with Image.open(path) as raw:
                        image=ImageOps.exif_transpose(raw).convert('RGB').resize((size,size),Image.Resampling.BILINEAR)
                        tensors.append(TF.normalize(TF.to_tensor(image),IMAGENET_MEAN,IMAGENET_STD))
                batch=torch.stack(tensors).to(device)
                routes=[]
                for name in ([dataset] if dataset else sources):
                    ids=torch.full((len(tensors),),sources.index(name),dtype=torch.long,device=device)
                    result=model(batch,ids)
                    indices,logits=result['fine_logits'][name]
                    if indices.tolist()!=list(range(len(tensors))):
                        raise ValueError('Unexpected head index permutation.')
                    probability=torch.softmax(logits.float(),dim=-1)
                    if dataset:
                        routes.append(probability.cpu().numpy())
                    else:
                        positive=[i for i,label in enumerate(cfg['datasets'][name]['classes'])
                                  if cfg['datasets'][name]['ontology_map'][label]=='neoplastic']
                        routes.append(probability[:,positive].sum(dim=1).cpu().numpy())
                values.extend(np.mean(np.stack(routes),axis=0).tolist())
        folds.append(values)
    mean=np.mean(np.asarray(folds),axis=0)
    results=[]
    for path,value in zip(images,mean):
        if dataset:
            classes=cfg['datasets'][dataset]['classes']
            result={'path':str(path),'dataset_head':dataset,'prediction':classes[int(np.argmax(value))],
                    'probabilities':dict(zip(classes,map(float,value)))}
        else:
            result={'path':str(path),'prob_neoplastic':float(value),'prediction':int(value>=.5),
                    'threshold':.5,'routing':'equal source-head mean, then equal checkpoint mean'}
        result['checkpoint_count']=len(checkpoints)
        results.append(result)
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Predict cystoscopy image classes with trained DCTG-VMamba checkpoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Use one checkpoint for a single-fold prediction or five distinct folds for an ensemble. '
               'Without --dataset, predictions use the mean neoplastic probability across trained source heads '
               'and checkpoints, with a fixed threshold of 0.5.',
    )
    parser.add_argument(
        '--configs', nargs='+', metavar='CONFIG_JSON',
        help='Optional configuration override for archived or relocated checkpoints. '
             'Provide one shared file or one file per checkpoint in matching order. '
             'Otherwise training_config.json is found automatically in each checkpoint\'s parent folders.',
    )
    parser.add_argument(
        '--checkpoints', nargs='+', required=True, metavar='CHECKPOINT_PT',
        help='One trained checkpoint, or five checkpoints from distinct folds. '
             'Each checkpoint must match its saved configuration.',
    )
    parser.add_argument(
        '--images', nargs='+', required=True, metavar='IMAGE',
        help='One or more image file paths. Images are converted to RGB, resized and normalized for the model.',
    )
    parser.add_argument(
        '--dataset', choices=('cystods', 'ebtc', 'kaggle_wli'),
        help='Use this trained source head to predict its original fine-grained classes. '
             'Omit this option to predict the shared binary neoplastic endpoint.',
    )
    parser.add_argument(
        '--device', choices=('cuda', 'cpu'), default='cuda',
        help='Inference device. CPU availability depends on the installed VMamba backend.',
    )
    parser.add_argument(
        '--output', metavar='OUTPUT_JSON',
        help='Write predictions to this JSON file. If omitted, print them to standard output.',
    )
    args=parser.parse_args()
    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    configs = load_training_configs(checkpoints, args.configs)
    result=predict(configs,checkpoints,[Path(p) for p in args.images],args.dataset,args.device)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
    else:
        print(text,end='')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
