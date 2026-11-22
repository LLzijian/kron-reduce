"""Registered one-use full TASK3 fit and separately frozen test evaluation."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from project_harness.evidence import sha256_file, verify_artifact_directory
from project_harness.git_state import assert_clean, git_head, verify_bound_file
from project_harness.loop import approval_recorded, load_state, workflow_lock
from project_harness.manifest import load_manifest, project_root
from project_harness.policy import load_policy, validate_run_policy
from project_harness.report_storage import locate_report
from project_harness.runner import _run_experiment

WORKFLOW = 'kiki_machine_local_full_v1'


def authorize(root, stage):
    assert_clean(root)
    if sys.version_info[:2] != (3, 12) or stage not in ('fit', 'test'):
        raise RuntimeError('registered stage and Python 3.12 required')
    name = f'kiki_machine_local_full_{stage}_v1'
    entry = load_manifest(root)['experiments'][name]
    validate_run_policy(root, name, entry)
    if not approval_recorded(load_state(root, WORKFLOW), stage):
        raise RuntimeError('explicit autonomous stage approval must be recorded')
    category = 'development_evaluations' if stage == 'fit' else 'official_test_evaluations'
    auth = load_policy(root)[category][name+'_20261113']
    if (auth['status'] != 'active' or auth['max_executions'] != 1 or auth['retry_allowed']
            or auth['experiment'] != name or auth['fit_attempts'] != int(stage == 'fit')):
        raise RuntimeError('scope not active or differs')
    verify_bound_file(root, auth['plan_path'], auth['plan_sha256'])
    verify_bound_file(root, auth['review_path'], auth['review_sha256'])
    plan = json.loads((root/locate_report(root, auth['plan_path'], auth['plan_sha256'])).read_text(encoding='utf-8'))
    review = json.loads((root/locate_report(root, auth['review_path'], auth['review_sha256'])).read_text(encoding='utf-8'))
    if (review.get('approved') is not True or review['plan_sha256'] != auth['plan_sha256']
            or review['implementation_bindings'] != plan['implementation_bindings']):
        raise RuntimeError('exact implementation scientific review required')
    for path, digest in plan['implementation_bindings'].items():
        verify_bound_file(root, path, digest)
    for path, digest in plan['external_code'].items():
        if sha256_file(Path(path)) != digest:
            raise RuntimeError('TASK3 scoring source changed')
    if (plan['stage'] != stage or entry['module'] != 'kron_reduce.workflows.kiki_machine_local_full'
            or entry['workflow'] != WORKFLOW):
        raise RuntimeError('execution registration differs')
    return name, entry, plan


def run(*, root, experiment, stage, result_dir, run_id):
    name, entry, plan = authorize(root, stage)
    if experiment != entry or result_dir.parent != root/'results' or result_dir.name != run_id or any(result_dir.iterdir()):
        raise RuntimeError('run identity differs')
    reservation = root/'.project_state'/(name+'_20261113.spent.json')
    with reservation.open('x', encoding='utf-8') as stream:
        json.dump({'run_id': run_id, 'source_commit': git_head(root), 'retry_allowed': False}, stream)
    commit = git_head(root)
    module = importlib.import_module('kron_reduce.evaluation.kiki_machine_local_full')
    result = module.execute(root, result_dir, plan, stage)
    if git_head(root) != commit:
        raise RuntimeError('source changed during execution')
    verify_artifact_directory(result_dir, strict=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('fit', 'test'), required=True)
    args = parser.parse_args()
    root = project_root()
    with workflow_lock(root, WORKFLOW):
        name, entry, _ = authorize(root, args.stage)
        print(_run_experiment(root, name, args.stage, experiment=entry))


if __name__ == '__main__':
    main()
