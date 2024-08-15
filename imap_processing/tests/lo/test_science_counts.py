from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from imap_processing import decom, imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts
from imap_processing.lo.l0.lo_apid import LoAPID


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


@pytest.skip()
@pytest.fixture()
def sample_packet_data():
    # original file name:
    # Instrument_Emulator_ILO_Emulator_v3.4_HVSCI_Sample_20240627T204953.CCSDS
    test_file = (
        imap_module_directory
        / "tests/lo/sample_data/lo_emulator_v3.4_HVSCI_sample_20240627T204953.CCSDS"
    )
    xtce_file = imap_module_directory / "lo/packet_definitions/P_ILO_SCI_CNT.xml"

    packets = decom.decom_packets(test_file.resolve(), xtce_file.resolve())
    de_packets = list(
        filter(
            lambda x: x.header["PKT_APID"].derived_value == LoAPID.ILO_SCI_CNT, packets
        )
    )

    return de_packets


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


pytest.skip()


def test_validation_data(sample_packet_data):
    histogram_fields = [
        "START_A",
        "START_C",
        "STOP_B0",
        "STOP_B3",
        "TOF0",
        "TOF1",
        "TOF2",
        "TOF3",
        "TOF0_TOF1",
        "TOF0_TOF2",
        "TOF1_TOF2",
        "SILVER",
        "DISC_TOF0",
        "DISC_TOF1",
        "DISC_TOF2",
        "DISC_TOF3",
        "POS0",
        "POS1",
        "POS2",
        "POS3",
        "HYDROGEN",
        "OXYGEN",
    ]
    # original file name:
    # Instrument_Emulator_ILO_Emulator_v3.4_HVSCI
    # _Sample_ILO_SCI_CNT_20240627T205042_EU.csv
    validation_file = (
        imap_module_directory / "tests/lo/validation_data/"
        "lo_emulator_v3.4_HVSCI_Sample_SCI_CNT_20240627T205042_EU.csv"
    )
    validation_data = pd.read_csv(validation_file)

    for pkt_idx, packet in enumerate(sample_packet_data):
        histogram = ScienceCounts(packet, "000", "packet_name")

        for field in histogram_fields:
            print()
            print(f"FIELD: {field}")
            validation_column = get_validation_column(validation_data, field)
            print(f"COLUMN: {validation_column}")
            if field in [
                "TOF0_TOF1",
                "TOF0_TOF2",
                "TOF1_TOF2",
                "SILVER",
                "HYDROGEN",
                "OXYGEN",
            ]:
                shape = (60, 7)
            else:
                shape = (6, 7)
            validation_field = np.array(
                validation_data.loc[pkt_idx, validation_column].tolist()
            ).reshape(shape)
            print(f"VALIDATION FIELD: {validation_field}")
            print(f"DATA: {getattr(histogram, field)}")
            np.testing.assert_array_equal(getattr(histogram, field), validation_field)


def get_validation_column(validation_data: pd.DataFrame, column_prefix: str) -> list:
    # the validation data contains a duplicate set of columns for the same values.
    # They are formatted as
    # - START_A_EGX_AZ_Y
    # - START_A_EGX_AZ[Y]
    # adding a condition to remove the one with [ when combining those columns into
    # a list
    if column_prefix in ["TOF0", "TOF1", "TOF2", "TOF3"]:
        return [
            col
            for col in validation_data.columns
            if col.startswith(column_prefix)
            and "[" not in col
            and col.count("TOF") == 1
        ]
    else:
        return [
            col
            for col in validation_data.columns
            if col.startswith(column_prefix) and "[" not in col
        ]
