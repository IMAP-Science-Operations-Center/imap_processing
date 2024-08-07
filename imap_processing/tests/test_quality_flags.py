"""Test bitwise flagging."""

import numpy as np

from imap_processing.quality_flags import HitFlags, ImapLoFlags, ImapUltraFlags


def test_quality_flags():
    """Test the bitwise operations."""

    # Test individual flags
    assert HitFlags.NONE == 0x0
    assert ImapUltraFlags.NONE == 0x0
    assert ImapLoFlags.NONE == 0x0
    assert HitFlags.INF == 2**0
    assert ImapUltraFlags.INF == 2**0
    assert ImapLoFlags.INF == 2**0
    assert HitFlags.NEG == 2**1
    assert ImapUltraFlags.NEG == 2**1
    assert ImapLoFlags.NEG == 2**1
    assert ImapUltraFlags.BADSPIN == 2**2
    assert ImapLoFlags.BADSPIN == 2**2
    assert ImapUltraFlags.FLAG1 == 2**3
    assert ImapLoFlags.FLAG2 == 2**3
    assert HitFlags.FLAG3 == 2**2

    # Test combined flags for Ultra
    flag = (
        ImapUltraFlags.INF
        | ImapUltraFlags.NEG
        | ImapUltraFlags.BADSPIN
        | ImapUltraFlags.FLAG1
    )
    assert flag & ImapUltraFlags.INF
    assert flag & ImapUltraFlags.BADSPIN
    assert flag & ImapUltraFlags.FLAG1
    assert flag.name == "INF|NEG|BADSPIN|FLAG1"
    assert flag.value == 15

    # Test combined flags for Lo
    flag = ImapLoFlags.INF | ImapLoFlags.NEG | ImapLoFlags.BADSPIN | ImapLoFlags.FLAG2
    assert flag & ImapLoFlags.INF
    assert flag & ImapLoFlags.BADSPIN
    assert flag & ImapLoFlags.FLAG2
    assert flag.name == "INF|NEG|BADSPIN|FLAG2"
    assert flag.value == 15

    # Test combined flags for HIT
    flag = HitFlags.INF | HitFlags.NEG | HitFlags.FLAG3
    assert flag & HitFlags.INF
    assert flag & HitFlags.FLAG3
    assert flag.name == "INF|NEG|FLAG3"
    assert flag.value == 7

    # Test use-case for Ultra
    data = np.array([-6, np.inf, 2, 3])
    quality = np.array(
        [
            ImapUltraFlags.INF | ImapUltraFlags.NEG,
            ImapUltraFlags.INF,
            ImapUltraFlags.NONE,
            ImapUltraFlags.NONE,
        ]
    )
    # Select data without INF flags
    non_inf_mask = (quality & ImapUltraFlags.INF.value) == 0
    np.array_equal(data[non_inf_mask], np.array([-6, 2, 3]))

    # Select data without NEG or INF flags
    non_neg_mask = (quality & ImapUltraFlags.NEG.value) == 0
    np.array_equal(data[non_inf_mask & non_neg_mask], np.array([2, 3]))
