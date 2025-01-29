"""Multiple level MAG tests"""

import pytest

from imap_processing.mag.l1b.mag_l1b import mag_l1b


def test_l1a_to_l1b(validation_l1a):
    # Convert l1a input validation packet file to l1b
    with pytest.raises(ValueError, match="Raw L1A"):
        mag_l1b(validation_l1a[0], "v000")

    l1b = [mag_l1b(i, "v000") for i in validation_l1a[1:]]

    assert len(l1b) == len(validation_l1a) - 1

    assert l1b[0].attrs["Logical_source"] == "imap_mag_l1b_norm-mago"
    assert l1b[1].attrs["Logical_source"] == "imap_mag_l1b_norm-magi"

    assert len(l1b[0]["vectors"].data) > 0
    assert len(l1b[1]["vectors"].data) > 0
