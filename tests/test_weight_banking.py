from predcircuit.weight_banking import analyze, bank


def test_cyclic_mapping_is_conflict_free_for_planned_parallelism() -> None:
    for parallelism in (8, 16, 32, 64):
        result = analyze(width=64, parallelism=parallelism, weight_bits=14)
        assert result.forward_conflicts == 0
        assert result.transpose_conflicts == 0
        assert result.max_bank_occupancy == 4096 // parallelism


def test_single_layer_fragmentation_matches_exact_bank_model() -> None:
    expected_ramb36 = {8: 8, 16: 16, 32: 32, 64: 64}
    for parallelism, expected in expected_ramb36.items():
        result = analyze(width=64, parallelism=parallelism, weight_bits=14)
        assert result.total_ramb36 == expected


def test_each_aligned_group_is_a_permutation_of_banks() -> None:
    for parallelism in (8, 16, 32, 64):
        expected = list(range(parallelism))
        for fixed in range(64):
            for start in range(0, 64, parallelism):
                forward = sorted(
                    bank(fixed, start + lane, parallelism) for lane in range(parallelism)
                )
                transpose = sorted(
                    bank(start + lane, fixed, parallelism) for lane in range(parallelism)
                )
                assert forward == expected
                assert transpose == expected
