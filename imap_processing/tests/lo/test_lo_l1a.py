from pathlib import Path

import pytest

from imap_processing.lo.l1a.lo_l1a import lo_l1a


@pytest.mark.parametrize(
    ("dependency", "expected_out"),
    [
        (Path("imap_lo_l0_de_20100101_v001.pkts"), "imap_lo_l1a_de_20100101_v001.cdf"),
        (
            Path("imap_lo_l0_spin_20100101_v001.pkt"),
            "imap_lo_l1a_spin_20100101_v001.cdf",
        ),
    ],
)
def test_lo_l1a(dependency, expected_out):
    # Act
    output_file = lo_l1a(dependency, "001")

    # Assert
    assert expected_out == output_file.name
