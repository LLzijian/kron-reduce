"""Gradients of the registered first-swing objectives at a fixed model."""

import torch

from kron_reduce.evaluation.first_swing_loss import objective_terms, training_weights


def gradient_probe(model, events, scales, budget, settings):
    """Accumulate full-train gradients in the shared model's parameter order."""
    terms = ["weighted_rollout", "one_sided_velocity", "weak_momentum"]
    scalar_names = ("log_stiffness_factor", "log_damping_factor")
    named_parameters = tuple(model.named_parameters())
    parameters = tuple(parameter for _, parameter in named_parameters)
    coordinates = [coordinate for name, parameter in named_parameters
                   for coordinate in ([name] if parameter.ndim == 0 else
                                      [f"{name}[{index}]" for index in range(parameter.numel())])]
    if (len(coordinates) != 22 or not all(name in coordinates for name in scalar_names)
            or not all(parameter.requires_grad for parameter in parameters)):
        raise ValueError("gradient probe requires the 22 trainable shared coordinates")
    scalar_indices = {name: coordinates.index(name) for name in scalar_names}
    accumulated = parameters[0].new_zeros((len(terms), len(coordinates)))
    weights = training_weights(events, settings["large_event_weight"], settings["history_kinetic_fraction"])
    batch_size = settings["train_batch"]
    for offset in range(0, len(events), batch_size):
        batch = events[offset:offset+batch_size]
        objective, components = objective_terms(model, batch, len(events), scales, budget, settings, weights=weights)
        active = [index for index, name in enumerate(terms) if components[name].requires_grad]
        for index in active:
            budget.spend("gradients")
            gradients = torch.autograd.grad(components[terms[index]], parameters, allow_unused=True,
                                            retain_graph=index != active[-1])
            pieces = [gradient.flatten() if gradient is not None else torch.zeros_like(parameter).flatten()
                      for gradient, parameter in zip(gradients, parameters, strict=True)]
            accumulated[index] += torch.cat(pieces)
        del objective, components

    lengths = accumulated.norm(dim=1)
    if not torch.isfinite(lengths).all():
        raise FloatingPointError("nonfinite first-swing gradient norm")
    normalized = accumulated / torch.where(lengths > 0., lengths, 1.)[:, None]
    cosines = (normalized @ normalized.T).clamp(-1., 1.).tolist()
    gradients = dict(zip(terms, accumulated.tolist(), strict=True))
    norms = dict(zip(terms, lengths.tolist(), strict=True))
    return {"terms": terms, "parameter_coordinates": coordinates, "gradients": gradients, "norms": norms,
            "cosines": [[value if norms[left] and norms[right] else None
                         for right, value in zip(terms, row, strict=True)]
                        for left, row in zip(terms, cosines, strict=True)],
            "scalars": {term: {name: values[index] for name, index in scalar_indices.items()}
                        for term, values in gradients.items()}}
