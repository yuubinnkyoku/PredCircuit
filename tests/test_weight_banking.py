from predcircuit.weight_banking import analyze, bank, bank_address


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


def test_fourteen_bit_weights_use_legal_18_by_2048_ramb36_mode() -> None:
    for parallelism in (8, 16, 32, 64):
        result = analyze(width=64, parallelism=parallelism, weight_bits=14, layers=30)
        assert result.ramb36_word_width == 18
        assert result.ramb36_word_depth == 2048


def test_thirty_layers_are_depth_packed_into_shared_banks() -> None:
    # A bit-capacity-only model incorrectly predicts 48/48/64/64. RAMB36E1
    # stores a 14-bit writable word in its legal 18-bit x 2048 TDP shape, so
    # the exact primitive count is 64 at every planned P.
    expected_ramb36 = {8: 64, 16: 64, 32: 64, 64: 64}
    expected_per_bank = {8: 8, 16: 4, 32: 2, 64: 1}
    for parallelism, expected in expected_ramb36.items():
        result = analyze(width=64, parallelism=parallelism, weight_bits=14, layers=30)
        assert result.total_ramb36 == expected
        assert result.ramb36_per_bank == expected_per_bank[parallelism]
        assert result.max_bank_occupancy == 30 * 4096 // parallelism
        assert result.forward_conflicts == 0
        assert result.transpose_conflicts == 0


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


def test_depth_packed_physical_addresses_are_bijective() -> None:
    width = 64
    layers = 30
    for parallelism in (8, 16, 32, 64):
        expected_depth = layers * width * width // parallelism
        seen: set[tuple[int, int]] = set()
        per_bank = [set() for _ in range(parallelism)]
        for layer in range(layers):
            for row in range(width):
                for col in range(width):
                    location = bank_address(layer, row, col, width, parallelism)
                    assert location not in seen
                    seen.add(location)
                    physical_bank, address = location
                    assert 0 <= address < expected_depth
                    per_bank[physical_bank].add(address)

        assert len(seen) == layers * width * width
        for addresses in per_bank:
            assert addresses == set(range(expected_depth))
