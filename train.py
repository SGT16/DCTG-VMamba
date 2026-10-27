from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from bladder_project.configuration import create_config

def main():
    parser = argparse.ArgumentParser(
        description='Train DCTG-VMamba on existing fixed folds using the retained 100-epoch protocol.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-root', required=True, metavar='DIR',
        help='Working project directory. The default fold location is data/locked_folds_official under this directory.',
    )
    parser.add_argument(
        '--data-root', required=True, metavar='DIR',
        help='Root of the prepared source-data project used to resolve the dataset and fold image paths.',
    )
    parser.add_argument(
        '--output-root', required=True, metavar='DIR',
        help='Destination for training outputs and the saved configuration. Must be separate from source data.',
    )
    parser.add_argument(
        '--vmamba-root', required=True, metavar='DIR',
        help='Local VMamba repository containing classification/models and its required compiled backend.',
    )
    parser.add_argument(
        '--vmamba-checkpoint', required=True, metavar='WEIGHTS',
        help='Pretrained VMamba backbone weights used to initialize training.',
    )
    parser.add_argument(
        '--fold-root', metavar='DIR',
        help='Override the existing fixed-fold directory. Each dataset needs a split manifest and train/val/test CSV files.',
    )
    parser.add_argument(
        '--folds', nargs='+', type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5],
        help='Distinct fold numbers to train. Existing split membership is reused; new folds are not generated.',
    )
    parser.add_argument('--run-id', default='public_reproduction', help='Identifier recorded for this training run.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data-loader worker processes.')
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the resolved configuration and exit without importing model dependencies or starting training.',
    )
    args=parser.parse_args()
    cfg=create_config(**{k:v for k,v in vars(args).items() if k!='dry_run'})
    if args.dry_run:
        print(json.dumps(cfg,indent=2))
        return 0
    project=Path(cfg['_project_root'])
    inputs=[]
    for dataset in cfg['datasets'].values():
        folder=Path(dataset['locked_fold_root'])
        inputs.append(folder/'split_manifest.json')
        inputs.extend(folder/f'fold_{fold}'/f'{split}.csv' for fold in cfg['_only_folds'] for split in ('train','val','test'))
    missing=[str(p) for p in inputs if not p.is_file()]
    if missing:
        raise FileNotFoundError('Supply existing fixed-fold inputs: '+', '.join(missing))
    weights=Path(args.vmamba_checkpoint)
    if not weights.is_file() or not (Path(args.vmamba_root)/'classification/models').is_dir():
        raise FileNotFoundError('Supply the VMamba repository and pretrained checkpoint.')
    source=Path(__file__).resolve().parent
    cfg['revision']['provenance']={
        'code_sha256':{p.relative_to(source).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [source/'train.py',*sorted((source/'bladder_project').glob('*.py'))]},
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    output=Path(cfg['results']['root'])
    output.mkdir(parents=True,exist_ok=True)
    config_path=output/'training_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        raise FileExistsError('Output already contains a different training configuration.')
    config_path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
    from bladder_project.v3_training import run_study
    run_study(cfg)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
