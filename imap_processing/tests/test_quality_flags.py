from imap_processing.quality_flags import L1bQualityFlags


def test_l1b_quality_flags():
    """Test the L1bQualityFlags bitwise operations."""

    assert L1bQualityFlags.NONE == 0x0
    assert L1bQualityFlags.BASE_INF == 0x1
    assert L1bQualityFlags.BASE_MISSING_TELEM == 0x2
    assert L1bQualityFlags.BASE_NEG == 0x4
    assert L1bQualityFlags.BASE_RES1 == 0x8
    assert L1bQualityFlags.ENA_RES1 == 0x10
    assert L1bQualityFlags.ULTRA_RES1 == 0x20
    assert L1bQualityFlags.ULTRA_RES2 == 0x40

    assert L1bQualityFlags.ALL == (
        L1bQualityFlags.BASE_INF
        | L1bQualityFlags.BASE_MISSING_TELEM
        | L1bQualityFlags.BASE_NEG
        | L1bQualityFlags.BASE_RES1
        | L1bQualityFlags.ENA_RES1
        | L1bQualityFlags.ULTRA_RES1
        | L1bQualityFlags.ULTRA_RES2
    )
    assert L1bQualityFlags.BASE_ALL == (
        L1bQualityFlags.BASE_INF
        | L1bQualityFlags.BASE_MISSING_TELEM
        | L1bQualityFlags.BASE_NEG
        | L1bQualityFlags.BASE_RES1
    )

    flag = L1bQualityFlags.BASE_INF | L1bQualityFlags.ENA_RES1
    assert flag & L1bQualityFlags.BASE_INF
    assert flag & L1bQualityFlags.ENA_RES1
    assert not flag & L1bQualityFlags.BASE_MISSING_TELEM

    assert L1bQualityFlags.NONE.name == "NONE"
    assert L1bQualityFlags.BASE_INF.name == "BASE_INF"
    combined_flags = L1bQualityFlags.BASE_INF | L1bQualityFlags.ENA_RES1
    assert combined_flags.name == "BASE_INF|ENA_RES1"
