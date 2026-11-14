"""Paired full-roster training, validation selection and frozen test scoring."""

from __future__ import annotations

import json
import math
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.kiki_machine_local_full import (
    batch_objective,
    evaluate,
    make_model,
    train_updates,
)
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.models.machine_local_forecast import FrozenMachineLocalForecaster
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head

GROUPS = ('r20', 'r21')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bind(root, path):
    return {'path': path.relative_to(root).as_posix(), 'sha256': sha256_file(path)}


def verified(root, binding):
    relative = Path(binding['path'])
    path = (root/relative).resolve(strict=True)
    if relative.is_absolute() or not path.is_relative_to(root) or sha256_file(path) != binding['sha256']:
        raise RuntimeError('frozen artifact identity differs')
    return path


def monitor(model, events, scales, budget, batch_size):
    value, values, failures = 0., [], []
    for offset in range(0, len(events), batch_size):
        try:
            with torch.no_grad():
                objective, event_values = batch_objective(model, events[offset:offset+batch_size], len(events),
                                                          scales, budget, return_events=True)
                value += float(objective)
                values.extend(event_values.tolist())
        except NUMERICAL_ERRORS as error:
            failures.append({'offset': offset, 'error': str(error)})
            values.extend([None]*len(events[offset:offset+batch_size]))
    return {'nmse': value if not failures and math.isfinite(value) else None, 'event_nmse': values,
            'qualified': sum(x is not None for x in values), 'failures': failures}


def fit_pair(models, train, val, scales, budget, directory, settings):
    bounds = (math.log(.5), math.log(2.))
    active = {}
    for group, model in models.items():
        target = directory/group
        target.mkdir()
        active[group] = train_updates(model, train, scales, budget, target, settings,
                                      stiffness_bounds=bounds if group == 'r21' else None)
    best, anchors, checkpoints, completion, latest = {}, {}, [], {}, {}
    stagnant = 0
    reason = 'update_cap'
    try:
        for update in range(1, settings['updates']+1):
            for group, steps in tuple(active.items()):
                try:
                    latest[group] = next(steps)
                except StopIteration as error:
                    completion[group] = error.value
                    del active[group]
            if not active:
                reason = 'all_training_failed'
                break
            if update % settings['val_interval']:
                continue
            rows, improved = {}, False
            for group in active:
                budget.spend('val_checks')
                result = monitor(models[group], val, scales, budget, settings['train_batch'])
                rows[group] = result
                state = snapshot(models[group])
                np.savez_compressed(directory/group/f'val-checkpoint-{update:02d}.npz',
                                    **{k: v.cpu().numpy() for k, v in state.items()})
                value = result['nmse']
                if value is None:
                    continue
                if group not in best or value < best[group][0]:
                    best[group] = (value, update, state)
                anchor = anchors.get(group)
                if anchor is None or value < anchor-1e-4*max(abs(anchor), 1e-12):
                    anchors[group] = value
                    improved = True
            stagnant = 0 if improved or any(r['nmse'] is None for r in rows.values()) else stagnant+1
            record = {'update': update, 'val': rows, 'joint_patience_count': stagnant}
            checkpoints.append(record)
            write_json(directory/f'val-check-{update:02d}.json', record)
            print(f"val check {update}: { {g: r['nmse'] for g, r in rows.items()} }", flush=True)
            if stagnant == settings['val_patience']:
                reason = 'joint_val_plateau'
                break
    finally:
        for steps in active.values():
            steps.close()
    training = {}
    for group, model in models.items():
        last = latest.get(group, {}).get('update', 0)
        row = {'status': completion.get(group, {}).get('status', 'COMPLETED'),
               'message': completion.get(group, {}).get('message'), 'completed_updates': last,
               'selected_update': last, 'selected_val_nmse': None, 'selection': 'last_finite_state_fallback'}
        if group in best:
            score, selected, state = best[group]
            model.load_state_dict(state)
            row.update(selected_update=selected, selected_val_nmse=score, selection='first_minimum_complete_val')
        scalar = model.log_stiffness_factor
        row['gamma'] = math.exp(float(scalar.detach())) if scalar is not None else 1.
        row['log_factor'] = float(scalar.detach()) if scalar is not None else 0.
        row['at_stiffness_bound'] = row['log_factor'] in bounds
        training[group] = row
    write_json(directory/'selection.json', {'stop_reason': reason, 'checkpoints': checkpoints, 'groups': training})
    return training


def freeze_pair(root, directory, name, models, scales, plan, training):
    files = {}
    for group, model in models.items():
        path = directory/f'{name}-{group}.pt'
        FrozenMachineLocalForecaster(model.freeze_effective_stiffness(), scales['history'],
                                      max_step=.0025).save_checkpoint(path)
        files[group] = bind(root, path)
    checkpoints = directory/f'{name}-checkpoints.json'
    write_json(checkpoints, files)
    preprocessing = directory/'preprocessing.json'
    frozen = {}
    for key, asset in (('checkpoint', checkpoints), ('preprocessing', preprocessing)):
        binding = bind(root, asset)
        frozen.update({key+'_path': binding['path'], key+'_sha256': binding['sha256']})
    frozen.update(phase=name, models=files, source_commit=git_head(root),
                  plan_sha256=sha256_file(root/plan['plan_path']), training=training,
                  input_contract='measured_history_100ms_zoh_v1', settings=plan['settings'],
                  scale_population=bind(root, directory/'train_scale_population.json'),
                  val_population=bind(root, directory/'val_monitor_population.json'),
                  fitting_closed=name == 'freeze', selection_closed=name == 'freeze')
    path = directory/f'{name}.json'
    write_json(path, frozen)
    verify_freeze(root, frozen)
    return path


def verify_freeze(root, frozen):
    for key in ('scale_population', 'val_population'):
        verified(root, frozen[key])
    paths = {}
    for key in ('checkpoint', 'preprocessing'):
        paths[key] = verified(root, {'path': frozen[key+'_path'], 'sha256': frozen[key+'_sha256']})
    files = read_json(paths['checkpoint'])
    if set(files) != set(GROUPS) or files != frozen['models']:
        raise RuntimeError('common freeze does not identify both models')
    return {group: verified(root, binding) for group, binding in files.items()}, read_json(paths['preprocessing'])


def score(model, events, scales, budget, directory, role, settings):
    forecasters = {step: FrozenMachineLocalForecaster(model, scales['history'], max_step=step)
                   for step in (.0025, .00125)}

    def predict(batch, step):
        try:
            return forecasters[step].predict_batch([e['history'] for e in batch],
                np.stack([e['equilibrium'] for e in batch]), 800, spend=budget.spend)
        except ValueError as error:
            if str(error) == 'model inputs must be finite':
                raise FloatingPointError('nonfinite intermediate RK4 state') from error
            raise

    result = evaluate(model, events, scales['history'], budget, directory, role, settings, predictor=predict,
                      target_scale=scales['target'])
    return {key: result[key] for key in ('summary', 'unstable_prefix', 'stability', 'task3_six_term_score')}


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(plan.get('device', 'cpu'))
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('registered CUDA device is unavailable')
        device = torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan['task3_root'])/'src'))
    settings = plan['settings']
    budget = Budget(directory/'operations.jsonl', plan['operation_limits'][stage])
    journal = AppendOnlyAuditJournal(directory/'access.jsonl')
    started = time.perf_counter()
    try:
        reader = Task3FullReader(root, plan['sources'], journal)
        training = None
        if stage == 'fit':
            budget.spend('checkpoint_loads')
            backbone = LearnedNonlinearModel.load(verified(root, plan['backbone'])).closure
            models = {group: make_model(backbone, train_stiffness=group == 'r21') for group in GROUPS}
            states = {group: snapshot(model) for group, model in models.items()}
            scalar = states['r21'].pop('log_stiffness_factor')
            if scalar.item() != 0. or states['r20'].keys() != states['r21'].keys() or any(
                    not torch.equal(v, states['r21'][key]) for key, v in states['r20'].items()):
                raise RuntimeError('R20/R21 initial states differ')
            train = reader.load('train')
            scales = {'history': np.maximum(np.sqrt(np.mean([np.mean(e['x'][:11]**2, axis=0) for e in train], axis=0)), 1e-5),
                      'target': np.maximum(np.sqrt(np.mean([np.mean(e['x'][11:311]**2, axis=0) for e in train], axis=0)), 1e-5)}
            write_json(directory/'preprocessing.json', {k: v.tolist() for k, v in scales.items()})
            write_json(directory/'train_scale_population.json', {'ids': [e['id'] for e in train],
                'history_points': 11, 'target_points': [min(300, len(e['x'])-11) for e in train]})
            write_json(directory/'val_monitor_population.json', {'ids': [
                e['id'] for e in plan['sources']['rosters']['val']]})
            initial = freeze_pair(root, directory, 'initial', models, scales, plan, None)
            val = reader.load('val', freeze_path=initial)
            write_json(directory/'val_monitor_points.json', {'ids': [e['id'] for e in val],
                'target_points': [min(300, len(e['x'])-11) for e in val]})
            if not all(np.allclose(e['mass'], models['r20'].mass.numpy(), rtol=0, atol=1e-12) for e in train+val):
                raise ValueError('backbone and data mass differ')
            for model in models.values():
                model.to(device)
            training = fit_pair(models, train, val, scales, budget, directory, settings)
            freeze_pair(root, directory, 'freeze', models, scales, plan, training)
            models = {group: model.freeze_effective_stiffness() for group, model in models.items()}
            roles = {'train': train, 'val': val}
        else:
            freeze_path = verified(root, plan['frozen_model_manifest'])
            frozen = read_json(freeze_path)
            if frozen['phase'] != 'freeze' or not frozen['fitting_closed'] or not frozen['selection_closed']:
                raise RuntimeError('test requires completed model selection')
            paths, scales = verify_freeze(root, frozen)
            models = {}
            for group, path in paths.items():
                budget.spend('checkpoint_loads')
                forecaster = FrozenMachineLocalForecaster.load_checkpoint(path)
                if forecaster.max_step != .0025 or not np.array_equal(forecaster.history_scale.numpy(), scales['history']):
                    raise RuntimeError('checkpoint forecast settings differ from common freeze')
                models[group] = forecaster.model.to(device).requires_grad_(False)
                (directory/group).mkdir()
            roles = {'test': reader.load('test', freeze_path=freeze_path)}
        records = {group: {role: score(model, events, scales, budget, directory/group, role, settings)
                           for role, events in roles.items()} for group, model in models.items()}
        summary = {'process_status': 'PASS', 'source_commit': git_head(root), 'training': training,
            'records': records, 'operations': dict(budget.counts), 'wall_seconds': time.perf_counter()-started,
            'runtime': {'python': sys.version, 'executable': sys.executable, 'device': str(device),
                        'cuda_build': torch.version.cuda,
                        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                        **{name: version(name) for name in ('torch', 'numpy', 'scipy', 'filelock', 'h5py')}},
            'promotion_decision': 'KEEP_BASELINE', 'scientific_pass': False,
            'claim_boundary': 'Full TASK3 paired R20/R21 with common initialization and frozen validation selection.'}
        write_json(directory/'summary.json', summary)
        write_json(directory/'stage_evidence.json', {k: summary[k] for k in (
            'process_status', 'source_commit', 'promotion_decision', 'scientific_pass', 'claim_boundary')}
                   | {'status': 'PASS', 'data_policy': 'kiki_official_train_test_v1'})
        return summary
    finally:
        journal.close()
        budget.stream.close()
        write_json(directory/'artifact_hashes.json', {p.relative_to(directory).as_posix(): sha256_file(p)
            for p in sorted(directory.rglob('*')) if p.is_file()})
