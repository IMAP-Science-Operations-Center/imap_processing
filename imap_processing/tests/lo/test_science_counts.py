from collections import namedtuple

import pytest

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts


@pytest.fixture()
def science_count():
    fake_data_field = namedtuple("fake_packet", ["raw_value", "derived_value"])
    sc = ScienceCounts.__new__(ScienceCounts)
    sc.ccsds_header = CcsdsData(
        {
            "VERSION": fake_data_field(0, 0),
            "TYPE": fake_data_field(0, 0),
            "SEC_HDR_FLG": fake_data_field(0, 0),
            "PKT_APID": fake_data_field(706, 706),
            "SEQ_FLGS": fake_data_field(0, 0),
            "SRC_SEQ_CTR": fake_data_field(0, 0),
            "PKT_LEN": fake_data_field(0, 0),
        }
    )
    return sc


def test_science_counts(science_count):
    """Test the science counts parsing, decompression, and shaping."""
    ## Arrange
    # sc = ScienceCounts("fake_packet", "version", "packet_name")
    science_count.SCI_CNT = "0" * 26880

    ## Act
    science_count._decompress_data()

    ## Assert
    assert science_count.START_A.shape == (6, 7)
    assert science_count.START_C.shape == (6, 7)
    assert science_count.STOP_B0.shape == (6, 7)
    assert science_count.STOP_B3.shape == (6, 7)
    assert science_count.TOF0.shape == (6, 7)
    assert science_count.TOF1.shape == (6, 7)
    assert science_count.TOF2.shape == (6, 7)
    assert science_count.TOF3.shape == (6, 7)
    assert science_count.TOF0_TOF1.shape == (60, 7)
    assert science_count.TOF0_TOF2.shape == (60, 7)
    assert science_count.TOF1_TOF2.shape == (60, 7)
    assert science_count.SILVER.shape == (60, 7)
    assert science_count.DISC_TOF0.shape == (6, 7)
    assert science_count.DISC_TOF1.shape == (6, 7)
    assert science_count.DISC_TOF2.shape == (6, 7)
    assert science_count.DISC_TOF3.shape == (6, 7)
    assert science_count.POS0.shape == (6, 7)
    assert science_count.POS1.shape == (6, 7)
    assert science_count.POS2.shape == (6, 7)
    assert science_count.POS3.shape == (6, 7)
    assert science_count.HYDROGEN.shape == (60, 7)
    assert science_count.OXYGEN.shape == (60, 7)
