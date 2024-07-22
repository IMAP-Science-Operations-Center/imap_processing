from imap_processing.quality_flags import BaseQualityFlags, UltraQualityFlags


def test_quality_flags():
    """Test the QualityFlags bitwise operations."""

    # Test BaseQualityFlags
    assert BaseQualityFlags.NONE == 0x0
    assert BaseQualityFlags.BASE_INF == 0x1
    assert BaseQualityFlags.BASE_MISSING_TELEM == 0x2
    assert BaseQualityFlags.BASE_NEG == 0x4
    assert BaseQualityFlags.BASE_RES1 == 0x8

    assert BaseQualityFlags.BASE_ALL == (
        BaseQualityFlags.BASE_INF
        | BaseQualityFlags.BASE_MISSING_TELEM
        | BaseQualityFlags.BASE_NEG
        | BaseQualityFlags.BASE_RES1
    )

    flag = BaseQualityFlags.BASE_INF | BaseQualityFlags.BASE_RES1
    assert flag & BaseQualityFlags.BASE_INF
    assert flag & BaseQualityFlags.BASE_RES1
    assert not flag & BaseQualityFlags.BASE_MISSING_TELEM

    assert BaseQualityFlags.NONE.name == "NONE"
    assert BaseQualityFlags.BASE_INF.name == "BASE_INF"
    combined_flags = BaseQualityFlags.BASE_INF | BaseQualityFlags.BASE_RES1
    assert combined_flags.name == "BASE_INF|BASE_RES1"

    # Test UltraQualityFlags
    assert UltraQualityFlags.NONE == 0x0
    assert UltraQualityFlags.ULTRA_RES1 == 0x10
    assert UltraQualityFlags.ULTRA_RES2 == 0x20
    assert UltraQualityFlags.ULTRA_RES3 == 0x40

    assert UltraQualityFlags.ALL == (
        UltraQualityFlags.ULTRA_RES3
        | UltraQualityFlags.ULTRA_RES1
        | UltraQualityFlags.ULTRA_RES2
    )

    flag = UltraQualityFlags.ULTRA_RES3 | UltraQualityFlags.ULTRA_RES1
    assert flag & UltraQualityFlags.ULTRA_RES3
    assert flag & UltraQualityFlags.ULTRA_RES1
    assert not flag & UltraQualityFlags.ULTRA_RES2

    assert UltraQualityFlags.NONE.name == "NONE"
    assert UltraQualityFlags.ULTRA_RES3.name == "ULTRA_RES3"
    combined_flags = UltraQualityFlags.ULTRA_RES3 | UltraQualityFlags.ULTRA_RES1
    assert combined_flags.name == "ULTRA_RES1|ULTRA_RES3"
