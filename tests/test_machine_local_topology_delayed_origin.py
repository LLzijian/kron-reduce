import pytest
import torch

from kron_reduce.evaluation.machine_local_topology import make_model
from kron_reduce.evaluation.machine_local_topology_delayed_origin import replace_block
from kron_reduce.evaluation.machine_local_topology_truth import parameters


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("stiffness", {"log_stiffness_factor"}),
        ("damping", {"log_damping_factor"}),
        ("memory", {"port_parameter", "rate_parameter"}),
    ],
)
def test_replace_block_copies_only_requested_truth_parameters(block, expected):
    source = make_model(parameters(), "T1M1", "cpu", train_scales=True)
    truth = make_model(parameters(), "T0M1", "cpu", train_scales=True)
    with torch.no_grad():
        for i, parameter in enumerate(truth.parameters()):
            parameter.fill_(0.2 + i * 0.1)
        for i, parameter in enumerate(source.parameters()):
            parameter.fill_(-0.2 - i * 0.1)

    original = {name: value.detach().clone() for name, value in source.named_parameters()}
    truth_values = {name: value.detach().clone() for name, value in truth.named_parameters()}
    result = replace_block(source, truth, block)

    assert result is not source
    assert type(result) is type(source)
    assert {name for name, value in result.named_parameters()
            if not torch.equal(value, original[name])} == expected
    for name, value in result.named_parameters():
        torch.testing.assert_close(value, truth_values[name] if name in expected else original[name])
    for name, value in source.named_parameters():
        torch.testing.assert_close(value, original[name])


def test_replace_block_rejects_unknown_block():
    source = make_model(parameters(), "T1M1", "cpu", train_scales=True)
    truth = make_model(parameters(), "T0M1", "cpu", train_scales=True)
    with pytest.raises(ValueError):
        replace_block(source, truth, "topology")
