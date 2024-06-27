from pathlib import Path

import pytest

from imap_processing.lo.l1a.lo_l1a import lo_l1a


@pytest.mark.parametrize(
    ("dependency", "expected_logical_source"),
    [
        (Path("imap_lo_l0_de_20100101_v001.pkts"), "imap_lo_l1a_de"),
        (
            Path("imap_lo_l0_spin_20100101_v001.pkt"),
            "imap_lo_l1a_spin",
        ),
    ],
)
def test_lo_l1a(dependency, expected_logical_source):
    # Act
    output_dataset = lo_l1a(dependency, "001")

    # Assert
    assert expected_logical_source == output_dataset.attrs["Logical_source"]
