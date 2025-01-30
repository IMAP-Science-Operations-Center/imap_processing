"""Test bitwise flagging."""

import numpy as np

from imap_processing.quality_flags import HitFlags, ImapHkUltraFlags, ImapLoFlags


def test_quality_flags():
    """Test the bitwise operations."""

    # Test individual flags
    assert HitFlags.NONE == 0x0
    assert ImapHkUltraFlags.NONE == 0x0
    assert ImapLoFlags.NONE == 0x0
    assert HitFlags.INF == 2**0
    assert ImapHkUltraFlags.INF == 2**0
    assert ImapLoFlags.INF == 2**0
    assert HitFlags.NEG == 2**1
    assert ImapHkUltraFlags.NEG == 2**1
    assert ImapLoFlags.NEG == 2**1
    assert ImapHkUltraFlags.BADSPIN == 2**2
    assert ImapLoFlags.BADSPIN == 2**2
    assert ImapHkUltraFlags.FLAG1 == 2**3
    assert ImapLoFlags.FLAG2 == 2**3
    assert HitFlags.FLAG3 == 2**2

    # Test combined flags for Ultra
    flag = (
        ImapHkUltraFlags.INF
        | ImapHkUltraFlags.NEG
        | ImapHkUltraFlags.BADSPIN
        | ImapHkUltraFlags.FLAG1
    )
    assert flag & ImapHkUltraFlags.INF
    assert flag & ImapHkUltraFlags.BADSPIN
    assert flag & ImapHkUltraFlags.FLAG1
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
            ImapHkUltraFlags.INF | ImapHkUltraFlags.NEG,
            ImapHkUltraFlags.INF,
            ImapHkUltraFlags.NONE,
            ImapHkUltraFlags.NONE,
        ]
    )
    # Select data without INF flags
    non_inf_mask = (quality & ImapHkUltraFlags.INF.value) == 0
    np.array_equal(data[non_inf_mask], np.array([-6, 2, 3]))

    # Select data without NEG or INF flags
    non_neg_mask = (quality & ImapHkUltraFlags.NEG.value) == 0
    np.array_equal(data[non_inf_mask & non_neg_mask], np.array([2, 3]))
