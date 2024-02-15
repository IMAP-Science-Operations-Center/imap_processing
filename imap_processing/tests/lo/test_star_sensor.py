import pytest

from imap_processing.lo.l0.data_classes.star_sensor import StarSensor


# Going to wait until validation data is available to test that
# the values are correct. Currently only check for the shape
# of the resulting data.
@pytest.mark.skip(reason="no data to initialize with")
def test_science_counts():
    ## Arrange
    sc = StarSensor("fake_packet", "version", "packet_name")
    sc.DATA_COMPRESSED = "0" * 5760

    ## Act
    sc._decompress_data()

    ## Assert
    assert sc.DATA.shape == (720,)
