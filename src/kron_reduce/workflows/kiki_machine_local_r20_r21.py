"""One-use equal-budget R20/R21 fit and frozen paired test entry point."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.runner import _run_experiment

WORKFLOW = 'kiki_machine_local_r20_r21_v1'
AUTH_SUFFIX = '_20261114'
MODULE = 'kron_reduce.workflows.kiki_machine_local_r20_r21'


def authorize(root, stage, name=None):
    assert_clean(root)
    if sys.version_info[:2] != (3, 12) or stage not in ('fit', 'test'):
        raise RuntimeError('registered stage and Python 3.12 required')
    historical = f'kiki_machine_local_r20_r21_{stage}_v1'
    name = historical if name is None else name
    entry = load_manifest(root)['experiments'][name]
    if entry.get('module') != MODULE or entry.get('stage') != stage:
        raise RuntimeError('execution registration differs')
    validate_run_policy(root, name, entry)
    if not approval_recorded(load_state(root, entry['workflow']), stage):
        raise RuntimeError('explicit autonomous stage approval must be recorded')
    category = 'development_evaluations' if stage == 'fit' else 'official_test_evaluations'
    key = 'development_authorization_id' if stage == 'fit' else 'test_authorization'
    authorization = entry.get(key, name + AUTH_SUFFIX if name == historical else None)
    auth = load_policy(root)[category].get(authorization, {})
    expected_fits = 2 if stage == 'fit' else 0
    if (auth.get('status') != 'active' or auth['max_executions'] != 1 or auth['retry_allowed']
            or auth['experiment'] != name or auth['fit_attempts'] != expected_fits):
        raise RuntimeError('scope not active or differs')
    verify_bound_file(root, auth['plan_path'], auth['plan_sha256'])
    verify_bound_file(root, auth['review_path'], auth['review_sha256'])
    plan = json.loads((root / auth['plan_path']).read_text(encoding='utf-8'))
    review = json.loads((root / auth['review_path']).read_text(encoding='utf-8'))
    if (review.get('approved') is not True or review.get('plan_sha256') != auth['plan_sha256']
            or review.get('implementation_bindings') != plan.get('implementation_bindings')):
        raise RuntimeError('exact implementation scientific review required')
    verify_bound_file(root, plan['protocol_path'], plan['protocol_sha256'])
    if auth['operation_limits'] != plan['operation_limits'][stage] or plan['settings'] != {
            'updates': 80, 'train_batch': 64, 'eval_batch': 16, 'val_interval': 10, 'val_patience': 3}:
        raise RuntimeError('operation budget or selection schedule differs')
    for path, digest in plan['implementation_bindings'].items():
        verify_bound_file(root, path, digest)
    device = plan.get('device', 'cpu')
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', device):
        raise RuntimeError('register cpu or an explicit CUDA device in the execution plan')
    pins = dict(line.strip().split('==') for line in
                (root/'requirements/constraints-py312.txt').read_text(encoding='utf-8').splitlines()
                if line.strip() and not line.startswith('#'))
    for package in ('numpy', 'scipy', 'torch', 'filelock', 'h5py'):
        installed = version(package)
        expected = {pins[package]}
        if package == 'torch':
            expected = ({pins[package]+'+cu130'} if device.startswith('cuda')
                        else expected | {pins[package]+'+cpu'})
        if installed not in expected:
            raise RuntimeError(f'{package} version differs from the registered runtime')
    if device.startswith('cuda'):
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise RuntimeError('set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting a CUDA run')
        import torch

        index = torch.device(device).index
        if not torch.cuda.is_available() or (index is not None and index >= torch.cuda.device_count()):
            raise RuntimeError('registered CUDA device is unavailable')
    source = plan['source_contract']
    verify_bound_file(root, source['path'], source['sha256'])
    source_plan = json.loads((root/source['path']).read_text(encoding='utf-8'))
    plan['sources'] = source_plan['sources'] | {'allowed_roles': ['train', 'val'] if stage == 'fit' else ['test']}
    plan['task3_root'] = source_plan['task3_root']
    if stage == 'fit' and plan['backbone'] != source_plan['backbone']:
        raise RuntimeError('common starting backbone differs from the approved source')
    for path, digest in source_plan['external_code'].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError('TASK3 scoring source changed')
    if plan['stage'] != stage or (name == historical and entry['workflow'] != WORKFLOW):
        raise RuntimeError('execution registration differs')
    if stage == 'test':
        binding = auth['frozen_model_manifest']
        planned = plan['frozen_model_manifest']
        path = root / binding['path']
        if (planned != binding or not path.is_file() or sha256_file(path) != binding['sha256']):
            raise RuntimeError('test must use the verified common R20/R21 frozen manifest')
    return name, entry, plan, authorization


def run(*, root, experiment, stage, result_dir, run_id):
    name = run_id.rsplit(f'-{stage}-', 1)[0]
    name, entry, plan, authorization = authorize(root, stage, name)
    if experiment != entry or result_dir.parent != root / 'results' or result_dir.name != run_id or any(result_dir.iterdir()):
        raise RuntimeError('run identity differs')
    common = Path(subprocess.run(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        cwd=root, capture_output=True, text=True, check=True).stdout.strip()).resolve(strict=True)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', authorization):
        raise RuntimeError('invalid authorization identifier')
    reservation = common / 'kiki_r20_r21_budgets' / (authorization + '.spent.json')
    reservation.parent.mkdir(exist_ok=True)
    with reservation.open('x', encoding='utf-8') as stream:
        json.dump({'run_id': run_id, 'source_commit': git_head(root), 'retry_allowed': False}, stream)
    commit = git_head(root)
    module = importlib.import_module('kron_reduce.evaluation.kiki_machine_local_r20_r21')
    result = module.execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError('source changed during execution')
    verify_artifact_directory(result_dir, strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'test'), required=True)
    parser.add_argument('--experiment', help='experiment registered for this stage')
    args = parser.parse_args()
    root = project_root()
    name = args.experiment or f'kiki_machine_local_r20_r21_{args.stage}_v1'
    workflow = load_manifest(root)['experiments'][name]['workflow']
    with workflow_lock(root, workflow):
        name, entry, _, _ = authorize(root, args.stage, name)
        print(_run_experiment(root, name, args.stage, experiment=entry))


if __name__ == '__main__':
    main()
