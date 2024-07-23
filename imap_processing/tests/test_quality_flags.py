from imap_processing.quality_flags import QualityFlags


def test_quality_flags():
    """Test the QualityFlags bitwise operations."""

    # Test individual flags
    assert QualityFlags.NONE == 0x0
    assert QualityFlags.INF == 2**0
    assert QualityFlags.MISSING_TELEM == 2**1
    assert QualityFlags.NEG == 2**2
    assert QualityFlags.RES1 == 2**3
    assert QualityFlags.BAD_SPIN == 2**4
    assert QualityFlags.FOV == 2**5

    flag = QualityFlags.INF | QualityFlags.RES1
    assert flag & QualityFlags.INF
    assert flag & QualityFlags.RES1
    assert not flag & QualityFlags.MISSING_TELEM

    assert QualityFlags.NONE.name == "NONE"
    assert QualityFlags.INF.name == "INF"
    combined_flags = QualityFlags.INF | QualityFlags.RES1
    assert combined_flags.name == "INF|RES1"

    combined_flags = QualityFlags.MISSING_TELEM | QualityFlags.FOV
    assert combined_flags.name == "MISSING_TELEM|FOV"

    combined_flags = (
        QualityFlags.INF
        | QualityFlags.MISSING_TELEM
        | QualityFlags.NEG
        | QualityFlags.RES1
    )
    assert combined_flags.name == "INF|MISSING_TELEM|NEG|RES1"

    combined_flags = QualityFlags.BAD_SPIN | QualityFlags.FOV | QualityFlags.INF
    assert combined_flags.name == "INF|BAD_SPIN|FOV"

    combined_flags = QualityFlags.FOV | QualityFlags.RES1 | QualityFlags.MISSING_TELEM
    assert combined_flags.name == "MISSING_TELEM|RES1|FOV"

    combined_flags = QualityFlags.BAD_SPIN | QualityFlags.FOV
    assert combined_flags.name == "BAD_SPIN|FOV"
