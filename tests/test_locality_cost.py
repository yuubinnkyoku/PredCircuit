import pytest

from predcircuit.locality_cost import estimate_two_phase_pc_vs_bptt


def test_flyvis_extent1_training_state_crossover() -> None:
    cost = estimate_two_phase_pc_vs_bptt(
        nodes=443,
        edges=7698,
        batch_size=1,
        recurrent_steps=14,
        bytes_per_value=4,
    )

    assert cost.local_elements == 8584
    assert cost.bptt_state_lower_bound_elements == 6645
    assert cost.local_bytes == 34336
    assert cost.bptt_state_lower_bound_bytes == 26580
    assert cost.crossover_recurrent_steps == 19
    assert cost.bptt_over_local_ratio == pytest.approx(6645 / 8584)


def test_bptt_history_grows_with_temporal_depth() -> None:
    shallow = estimate_two_phase_pc_vs_bptt(
        nodes=100,
        edges=500,
        batch_size=1,
        recurrent_steps=4,
        bytes_per_value=2,
    )
    deep = estimate_two_phase_pc_vs_bptt(
        nodes=100,
        edges=500,
        batch_size=1,
        recurrent_steps=40,
        bytes_per_value=2,
    )

    assert shallow.local_elements == deep.local_elements
    assert deep.bptt_state_lower_bound_elements > shallow.bptt_state_lower_bound_elements
    assert deep.bptt_over_local_ratio > 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"nodes": 0, "edges": 1, "batch_size": 1, "recurrent_steps": 1}, "nodes"),
        ({"nodes": 1, "edges": -1, "batch_size": 1, "recurrent_steps": 1}, "edges"),
        ({"nodes": 1, "edges": 1, "batch_size": 0, "recurrent_steps": 1}, "batch_size"),
        ({"nodes": 1, "edges": 1, "batch_size": 1, "recurrent_steps": 0}, "recurrent_steps"),
    ],
)
def test_invalid_cost_arguments(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_two_phase_pc_vs_bptt(**kwargs)
