from collections import namedtuple

import pytest

from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.data_classes.star_sensor import StarSensor


@pytest.fixture()
def star_sensor():
    fake_data_field = namedtuple("fake_packet", ["raw_value", "derived_value"])
    star_sensor = StarSensor.__new__(StarSensor)
    star_sensor.ccsds_header = CcsdsData(
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
    return star_sensor


def test_science_counts(star_sensor):
    ## Arrange
    star_sensor.DATA_COMPRESSED = "0" * 5760

    ## Act
    star_sensor._decompress_data()

    ## Assert
    assert star_sensor.DATA.shape == (720,)
