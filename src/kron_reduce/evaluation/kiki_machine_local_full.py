"""Full-roster learning and frozen reporting of machine-local hidden dynamics."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kron_reduce.data.durable_audit_ledger import AppendOnlyAuditJournal
from kron_reduce.data.task3_full import Task3FullReader
from kron_reduce.evaluation.constant_c_observability import write_json
from kron_reduce.evaluation.machine_local_clean_learning import NUMERICAL_ERRORS, snapshot
from kron_reduce.evaluation.machine_local_synthetic import Budget
from kron_reduce.evaluation.state_port_training import loss
from kron_reduce.evaluation.trajectory_stability import summarize
from kron_reduce.models.machine_local import DiagonalMemoryModel
from kron_reduce.models.nonlinear_history import predict_from_history
from kron_reduce.models.nonlinear_latent import LearnedNonlinearModel
from project_harness.evidence import sha256_file
from project_harness.git_state import git_head


def predict_checked(*args, **kwargs):
    try:
        return predict_from_history(*args, **kwargs)
    except ValueError as error:
        if str(error) == 'model inputs must be finite':
            raise FloatingPointError('nonfinite intermediate RK4 state') from error
        raise


def inputs(events, scale):
    return (np.stack([e['x'][:11, :10] for e in events]),
            np.stack([e['x'][:11, 10:] for e in events]),
            np.stack([e['p0'] for e in events]),
            np.stack([e['equilibrium'] for e in events]), scale)


def make_model(backbone, *, train_stiffness=False):
    projection = np.eye(10)-np.outer(backbone.mass, np.ones(10))/backbone.mass.sum()
    amplitude = np.linalg.norm(backbone.coupling)/np.linalg.norm(projection)
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('backbone port scale must be finite and positive')
    model = DiagonalMemoryModel(backbone, port_kind='local', port=np.full(10, amplitude),
        tau=np.geomspace(.06, 1.6, 10), encoder=np.zeros((10, 80)), feature_scale=np.ones(80),
        train_stiffness=train_stiffness)
    model.encoder.requires_grad_(False)
    return model


def batch_objective(model, batch, population, scales, budget, *, return_events=False, predictor=None):
    counts = np.array([min(300, len(e['x'])-11) for e in batch])
    count = int(counts.max())
    target = np.stack([np.pad(e['x'][10:11+c], ((0, count-c), (0, 0)), mode='edge')
                       for e, c in zip(batch, counts, strict=True)])
    result = (predict_checked(model, *inputs(batch, scales['history']), count, max_step=.005, spend=budget.spend)
              if predictor is None else predictor(model, batch, scales['history'], .005, budget.spend, count=count))
    args = (result['forecast'], target, np.full(len(batch), 1/population), scales['target'], counts)
    value = loss(*args, return_events=True) if return_events else loss(*args)
    if not torch.isfinite(value[0] if return_events else value):
        raise FloatingPointError('nonfinite objective')
    return value


def train_updates(model, events, scales, budget, directory, settings, *, stiffness_bounds=None,
                  damping_bounds=None, spatial_rate_limit=None, predictor=None, learning_rate=.01, gradient_clip=10.,
                  objective_fn=None, minimum_port=None):
    if minimum_port is not None and (model.port_kind != "local" or not 0. < minimum_port < float("inf")):
        raise ValueError("a positive finite port floor requires local ports")
    parameters = (model.port_parameter, model.rate_parameter)
    scalar = getattr(model, 'log_stiffness_factor', None)
    if (scalar is None) != (stiffness_bounds is None):
        raise ValueError("trainable stiffness requires an explicit parameter range")
    if scalar is not None:
        parameters += (scalar,)
    damping = getattr(model, 'log_damping_factor', None)
    if (damping is None) != (damping_bounds is None):
        raise ValueError('trainable damping requires an explicit parameter range')
    if damping is not None:
        parameters += (damping,)
    rates = getattr(model, 'spatial_relaxation_rates', None)
    if (rates is None) != (spatial_rate_limit is None):
        raise ValueError('trainable spatial relaxation requires an explicit rate limit')
    shared_parameters = parameters
    groups = [{"params": shared_parameters}]
    if rates is not None:
        parameters += (rates,)
        groups.append({"params": (rates,), "lr": settings['spatial_rate_learning_rate']})
    optimizer = torch.optim.Adam(groups, lr=learning_rate, betas=(.9, .999), eps=1e-8,
        weight_decay=0., amsgrad=False, maximize=False, foreach=False, fused=False)
    completed, started = 0, time.perf_counter()
    status, message = 'COMPLETED', None
    budget.spend('fits')
    with (directory/'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(settings['updates']):
            previous = snapshot(model)
            optimizer.zero_grad(set_to_none=True)
            objective = 0.
            terms = {}
            try:
                for offset in range(0, len(events), settings['train_batch']):
                    batch = events[offset:offset+settings['train_batch']]
                    if objective_fn is None:
                        value = batch_objective(model, batch, len(events), scales, budget, predictor=predictor)
                    else:
                        value, components = objective_fn(model, batch, len(events), scales, budget)
                        if not torch.isfinite(value) or not all(torch.isfinite(x) for x in components.values()):
                            raise FloatingPointError('nonfinite train objective')
                        for name in components:
                            terms[name] = terms.get(name, 0.)+float(components[name].detach())
                        del components
                    budget.spend('gradients')
                    value.backward()
                    objective += float(value.detach())
                    del value
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('nonfinite train gradient')
                norm = torch.nn.utils.clip_grad_norm_(shared_parameters, gradient_clip)
                spatial_norm = torch.nn.utils.clip_grad_norm_((rates,), gradient_clip) if rates is not None else None
                if not torch.isfinite(norm) or (spatial_norm is not None and not torch.isfinite(spatial_norm)):
                    raise FloatingPointError('nonfinite gradient norm')
                budget.spend('optimizer_updates')
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError('nonfinite updated parameters')
                if any(not torch.isfinite(v).all() for state in optimizer.state.values()
                       for v in state.values() if isinstance(v, torch.Tensor)):
                    raise FloatingPointError('nonfinite optimizer state')
                if scalar is not None:
                    log_factor_before_projection = float(scalar.detach())
                    with torch.no_grad():
                        scalar.clamp_(*stiffness_bounds)
                if damping is not None:
                    with torch.no_grad():
                        damping.clamp_(*damping_bounds)
                if rates is not None:
                    with torch.no_grad():
                        rates.clamp_(0., spatial_rate_limit)
                if minimum_port is not None:
                    with torch.no_grad():
                        model.port_parameter.clamp_min_(np.log(minimum_port))
                completed += 1
            except NUMERICAL_ERRORS as error:
                np.savez_compressed(directory/'failed_parameters.npz',
                                    **{k: v.cpu().numpy() for k, v in snapshot(model).items()})
                write_json(directory/'training_failure.json', {'update_attempt': update+1,
                    'completed_updates': completed, 'batch_offset': offset,
                    'error_type': type(error).__name__, 'message': str(error)})
                model.load_state_dict(previous)
                optimizer.zero_grad(set_to_none=True)
                status, message = 'NUMERICAL_FAILURE', str(error)
                break
            row = {'update': update+1, 'train_nmse_1s_3s': objective,
                   'gradient_norm': float(norm), 'elapsed_seconds': time.perf_counter()-started}
            if objective_fn is not None:
                row.update(terms, train_objective=objective)
            if scalar is not None:
                row.update(log_factor_before_projection=log_factor_before_projection,
                           log_factor=float(scalar.detach()),
                           projection_applied=log_factor_before_projection != float(scalar.detach()))
            if rates is not None:
                row['spatial_relaxation_rates'] = rates.detach().tolist()
                row['spatial_gradient_norm'] = float(spatial_norm)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            quantity = "NMSE" if objective_fn is None else "objective"
            print(f"{directory.name} update {completed}/{settings['updates']} train {quantity}={objective:.6g}", flush=True)
            yield row
    return {'status': status, 'completed_updates': completed, 'message': message,
            'selection': 'fixed_endpoint_or_last_completed_update_on_numerical_failure'}


def train(model, events, scales, budget, directory, settings):
    updates = train_updates(model, events, scales, budget, directory, settings)
    while True:
        try:
            next(updates)
        except StopIteration as done:
            return done.value


def _train_witnesses(events, predictions, stability_events):
    from kron_reduce.evaluation.train_mechanism_audit import _row_margin, _window_result

    witnesses = []
    for i, event in enumerate(events):
        if event["label"] != "unstable":
            continue
        count = min(300, len(event["x"])-11)

        def score(path, event=event):
            if not np.isfinite(path).all():
                return None
            return _window_result(_row_margin(path, event["equilibrium"], event["mass"], event["omega_s"]), min_end=11)

        measures = {"id": event["id"], "future_points": count, "truth": score(event["x"][:11+count]),
                    "prediction": {str(step): score(np.concatenate((event["x"][:11], values[i, 1:1+count])))
                                   if stability_events[i]["qualified"] else None for step, values in predictions.items()}}
        witnesses.append(measures)
    return witnesses


def evaluate(model, events, scale, budget, directory, role, settings, *, predictor=None, target_scale=None, save=True):
    from task3.c2.evaluation import evaluate as task3_evaluate
    from task3.protocol import classify as task3_classify
    predictions = {step: np.full((len(events), 801, 20), np.nan) for step in (.0025, .00125)}
    failures, diagnostics = [], []

    def forward(batch, offset, step):
        with torch.no_grad():
            result = (predict_checked(model, *inputs(batch, scale), 800, max_step=step, spend=budget.spend)
                      if predictor is None else predictor(batch, step))
        fields = ('forecast', 'hidden_start', 'history', 'gn_singular')
        detailed = 'iteration_residuals' in result
        if detailed:
            fields += ('iteration_residuals', 'gn_hidden_start', 'iterates')
        result = {field: result[field].cpu().numpy() for field in fields}
        forecast = result['forecast']
        predictions[step][offset:offset+len(batch)] = forecast[:, :, :20]
        for i, event in enumerate(batch):
            diagnostics.append({'id': event['id'], 'step': step,
                'hidden_start_energy': float(np.sum(result['hidden_start'][i]**2)/2),
                'hidden_origin_energy': float(np.sum(result['history'][i, -1, 20:]**2)/2),
                'history_nrmse': float(np.sqrt(np.mean(((result['history'][i, :, :20]-event['x'][:11])/scale)**2))),
                'singular_min': float(result['gn_singular'][:, i, -1].min())})
            if detailed:
                singular = result['gn_singular'][:, i]
                residual = result['iteration_residuals'][:, i]
                final = (result['history'][i, 1:, :20]-event['x'][1:11])/scale
                diagnostics[-1].update(
                    gn_residual_rms=np.r_[np.sqrt(np.mean(residual**2, axis=1)),
                                          np.sqrt(np.mean(final**2))].tolist(),
                    gn_increment_norm=np.linalg.norm(result['gn_hidden_start'][:, i], axis=1).tolist(),
                    gn_hidden_norm=np.linalg.norm(result['iterates'][:, i], axis=1).tolist(),
                    gn_singular_values=singular.tolist(),
                    gn_rank=np.count_nonzero(singular > 64*np.finfo(np.float64).eps*200*singular[:, :1], axis=1).tolist(),
                    gn_condition=(singular[:, 0]/singular[:, -1]).tolist(),
                    visible_reset=(event['x'][10]-result['history'][i, -1, :20]).tolist())

    for offset in range(0, len(events), settings['eval_batch']):
        batch = events[offset:offset+settings['eval_batch']]
        for step in predictions:
            try:
                forward(batch, offset, step)
            except NUMERICAL_ERRORS as error:
                failures.append({'offset': offset, 'count': len(batch), 'step': step, 'error': str(error)})
                # A failed batch is isolated once per event under the frozen evaluation budget.
                for i, event in enumerate(batch):
                    try:
                        forward([event], offset+i, step)
                    except NUMERICAL_ERRORS as individual:
                        failures.append({'id': event['id'], 'step': step, 'error': str(individual)})
        print(f'{role} evaluated {min(offset+len(batch), len(events))}/{len(events)}', flush=True)
    scoring = [{**e, 'x': e['raw_x'], 'equilibrium_delta': e['equilibrium'],
                'event_time_s': e.get('event_time_s')} for e in events]
    result = task3_evaluate(predictions[.0025], scoring, predictions[.00125])
    stability_events = []
    for index, (event, classification) in enumerate(zip(events, result['classifications'], strict=True)):
        predicted = ('unstable' if classification['predicted_los'] else
                     'stable' if classification['window_settled'] else None)
        refined = task3_classify(predictions[.00125][index], scoring[index])
        refined_label = ('unstable' if refined['predicted_los'] else
                         'stable' if refined['window_settled'] else None)
        qualified = classification['qualified'] and predicted == refined_label
        stability_events.append({'id': event['id'], 'truth_label': event['label'],
            'predicted_label': predicted, 'qualified': qualified,
            'prospective_eligible': True,
            'numerical_failure': not qualified, 'unsettled': qualified and predicted is None})
    result['stability'] = summarize(stability_events)
    s = result['stability']
    for name, numerator, denominator in (('recall_on_resolved', s['tp'], s['tp']+s['fn']),
            ('miss_rate_on_resolved', s['fn'], s['tp']+s['fn']),
            ('false_alarm_rate_on_resolved', s['fp'], s['tn']+s['fp'])):
        s[name] = numerator/denominator if denominator else None
    result['stability_events'] = stability_events
    if role == "train" and settings.get("matched_train_witnesses"):
        result["matched_train_witnesses"] = _train_witnesses(events, predictions, stability_events)
    result['failures'] = failures
    result['initialization'] = diagnostics
    result['common_frequency_output'] = 'zero_common_frequency_relative_dynamics_only'
    result['task3_six_term_score'] = result.pop('selection_score')
    if target_scale is not None:
        terminal = {}
        for step, paths in predictions.items():
            values = []
            for i, event in enumerate(events):
                if not stability_events[i]['qualified']:
                    values.append(None)
                    continue
                count = min(300, len(event['x'])-11)
                difference = (paths[i, 1:count+1]-event['x'][11:11+count])/target_scale
                value = .5*(np.mean(difference[:100]**2)+np.mean(difference**2))
                values.append(float(value) if np.isfinite(value) else None)
            terminal[str(step)] = {'event_values': values, 'qualified': sum(v is not None for v in values),
                'allocated': len(values), 'mean': float(np.mean(values)) if all(v is not None for v in values) else None}
        result['terminal_train_scale_nmse_1s_3s'] = terminal
    if save:
        np.savez_compressed(directory/(role+'_predictions.npz'),
                            ids=np.array([e['id'] for e in events]), fine=predictions[.0025], refined=predictions[.00125],
                            truth=np.concatenate([e['raw_x'] for e in events]),
                            lengths=np.array([len(e['raw_x']) for e in events]),
                            mass=np.stack([e['mass'] for e in events]),
                            equilibrium=np.stack([e['equilibrium'] for e in events]),
                            labels=np.array([e['label'] for e in events]))
        write_json(directory/(role+'.json'), result)
    return result


def execute(root, directory, plan, stage):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(plan['task3_root'])/'src'))
    settings = plan['settings']
    budget = Budget(directory/'operations.jsonl', plan['operation_limits'][stage])
    journal = AppendOnlyAuditJournal(directory/'access.jsonl')
    started = time.perf_counter()
    try:
        reader = Task3FullReader(root, plan['sources'], journal)
        source = root/plan['backbone']['path']
        if sha256_file(source) != plan['backbone']['sha256']:
            raise ValueError('backbone identity mismatch')
        budget.spend('checkpoint_loads')
        model = make_model(LearnedNonlinearModel.load(source).closure)
        training, records = None, {}
        if stage == 'fit':
            events = sorted(reader.load('train'), key=lambda event: event['id'])
            if not all(np.allclose(e['mass'], model.mass.numpy(), rtol=0, atol=1e-12) for e in events):
                raise ValueError('backbone and data mass differ')
            scales = {'history': np.maximum(np.sqrt(np.mean([np.mean(e['x'][:11]**2, axis=0) for e in events], axis=0)), 1e-5),
                      'target': np.maximum(np.sqrt(np.mean([np.mean(e['x'][11:311]**2, axis=0) for e in events], axis=0)), 1e-5)}
            np.savez_compressed(directory/'initial_model.npz', **{k: v.numpy() for k, v in snapshot(model).items()})
            write_json(directory/'train_scale_population.json', {
                'ids': [e['id'] for e in events], 'history_points_per_event': 11,
                'target_points': [min(300, len(e['x'])-11) for e in events],
                'weighting': 'event_equal_mean_square', 'floor': 1e-5})
            training = train(model, events, scales, budget, directory, settings)
            np.savez_compressed(directory/'model.npz', **{k: v.numpy() for k, v in snapshot(model).items()})
            write_json(directory/'preprocessing.json', {k: v.tolist() for k, v in scales.items()})
            freeze = {key+'_path': (directory/name).relative_to(root).as_posix()
                      for key, name in (('checkpoint', 'model.npz'), ('preprocessing', 'preprocessing.json'))}
            freeze.update({key+'_sha256': sha256_file(root/freeze[key+'_path']) for key in ('checkpoint', 'preprocessing')})
            freeze.update(source_commit=git_head(root), plan_sha256=sha256_file(root/plan['plan_path']),
                          training=training, fitting_closed=True, selection_closed=True)
            write_json(directory/'freeze.json', freeze)
            freeze_path = directory/'freeze.json'
            records['train'] = evaluate(model, events, scales['history'], budget, directory, 'train', settings)
            records['val'] = evaluate(model, reader.load('val', freeze_path=freeze_path), scales['history'], budget, directory, 'val', settings)
        else:
            freeze_path = root/plan['frozen_model_manifest']['path']
            if sha256_file(freeze_path) != plan['frozen_model_manifest']['sha256']:
                raise ValueError('frozen manifest differs from registered test scope')
            freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
            for key in ('checkpoint', 'preprocessing'):
                if sha256_file(root/freeze[key+'_path']) != freeze[key+'_sha256']:
                    raise ValueError('frozen model or preprocessing changed')
            budget.spend('checkpoint_loads')
            with np.load(root/freeze['checkpoint_path'], allow_pickle=False) as saved:
                model.load_state_dict({k: torch.tensor(saved[k]) for k in saved.files})
            scales = json.loads((root/freeze['preprocessing_path']).read_text(encoding='utf-8'))
            records['test'] = evaluate(model, reader.load('test', freeze_path=freeze_path), np.array(scales['history']), budget, directory, 'test', settings)
        summary = {'process_status': 'PASS', 'source_commit': git_head(root), 'training': training,
            'records': {role: {'summary': r['summary'], 'stability': r['stability']} for role, r in records.items()},
            'operations': dict(budget.counts), 'wall_seconds': time.perf_counter()-started,
            'runtime': {'python': sys.version, 'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__},
            'promotion_decision': 'KEEP_BASELINE', 'scientific_pass': False,
            'claim_boundary': 'One fixed full-roster L10 comparison with twenty learned port/time-scale parameters.'}
        write_json(directory/'summary.json', summary)
        write_json(directory/'stage_evidence.json', {k: summary[k] for k in ('process_status', 'source_commit', 'promotion_decision', 'scientific_pass', 'claim_boundary')}
                   | {'status': 'PASS', 'data_policy': 'kiki_official_train_test_v1'})
        return summary
    finally:
        journal.close()
        budget.stream.close()
        write_json(directory/'artifact_hashes.json', {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()})
