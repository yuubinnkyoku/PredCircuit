from predcircuit.hardware import estimate_local_learning_cost, residual_mlp_weight_scalars


def test_depth32_width8_counts_match_current_experiment() -> None:
    weights = residual_mlp_weight_scalars(depth=32, width=8, input_dim=8, output_dim=4)
    assert weights == 2016

    spc = estimate_local_learning_cost(
        family="spc",
        depth=32,
        width=8,
        input_dim=8,
        output_dim=4,
        batch_size=4,
        state_bits=12,
        weight_bits=12,
    )
    pcalm = estimate_local_learning_cost(
        family="pcalm",
        depth=32,
        width=8,
        input_dim=8,
        output_dim=4,
        batch_size=4,
        state_bits=12,
        dual_bits=12,
        weight_bits=12,
    )

    assert spc.free_state_scalars == 992
    assert pcalm.dual_state_scalars == 992
    assert spc.macs_per_relaxation_step == 16_128
    assert pcalm.macs_per_relaxation_step == 16_128
    assert spc.persistent_state_bits == 11_904
    assert pcalm.persistent_state_bits == 23_808
    assert pcalm.dual_scalar_updates_per_step == 992


def test_cycle_lower_bound_can_overlap_dual_update() -> None:
    pcalm = estimate_local_learning_cost(
        family="pcalm",
        depth=32,
        width=8,
        input_dim=8,
        output_dim=4,
        batch_size=4,
        state_bits=12,
        dual_bits=12,
        weight_bits=12,
    )
    assert pcalm.cycles_per_step_lower_bound(mac_lanes=128, dual_lanes=32) == 126
    assert (
        pcalm.cycles_per_step_lower_bound(
            mac_lanes=128,
            dual_lanes=32,
            overlap_dual_update=False,
        )
        == 157
    )
