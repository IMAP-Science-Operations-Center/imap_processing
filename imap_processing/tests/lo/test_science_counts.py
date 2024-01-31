import pytest

from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts


# Going to wait until validation data is available to test that
# the values are correct. Currently only check for the shape
# of the resulting data.
@pytest.mark.skip(reason="no data to initialize with")
def test_science_counts():
    """Test the science counts parsing, decompression, and shaping."""
    ## Arrange
    sc = ScienceCounts("fake_packet", "version", "packet_name")
    sc.SCI_CNT = "0" * 26880

    ## Act
    sc._parse_binary()

    ## Assert
    assert sc.START_A.shape == (6, 7)
    assert sc.START_C.shape == (6, 7)
    assert sc.STOP_B0.shape == (6, 7)
    assert sc.STOP_B3.shape == (6, 7)
    assert sc.TOF0.shape == (6, 7)
    assert sc.TOF1.shape == (6, 7)
    assert sc.TOF2.shape == (6, 7)
    assert sc.TOF3.shape == (6, 7)
    assert sc.TOF0_TOF1.shape == (60, 7)
    assert sc.TOF0_TOF2.shape == (60, 7)
    assert sc.TOF1_TOF2.shape == (60, 7)
    assert sc.SILVER.shape == (60, 7)
    assert sc.DISC_TOF0.shape == (6, 7)
    assert sc.DISC_TOF1.shape == (6, 7)
    assert sc.DISC_TOF2.shape == (6, 7)
    assert sc.DISC_TOF3.shape == (6, 7)
    assert sc.POS0.shape == (6, 7)
    assert sc.POS1.shape == (6, 7)
    assert sc.POS2.shape == (6, 7)
    assert sc.POS3.shape == (6, 7)
    assert sc.HYDROGEN.shape == (60, 7)
    assert sc.OXYGEN.shape == (60, 7)
