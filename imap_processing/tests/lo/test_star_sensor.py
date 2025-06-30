import numpy as np
import pytest

from imap_processing import imap_module_directory
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l0.lo_star_sensor import process_star_sensor
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture
def star_sensor_ds():
    xtce_file = imap_module_directory / "lo/packet_definitions/lo_xtce.xml"
    dependency = (
        imap_module_directory / "tests/lo/test_pkts/imap_lo_l0_raw_20240803_v002.pkts"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
        use_derived_value=False,
    )
    return datasets_by_apid[LoAPID.ILO_STAR]


def test_star_sensor(star_sensor_ds):
    validation_file = (
        imap_module_directory
        / "tests/lo/validation_data"
        / "Instrument_FM1_T104_R129_20240803_ILO_STAR_EU_trimmed.csv"
    )
    validation_arr = np.loadtxt(validation_file, delimiter=",", skiprows=1, dtype=int)
    validation_shcoarse = validation_arr[:, 0]
    validation_count = validation_arr[:, 1]
    # The first 720
    # validation_data_compressed = validation_arr[:, 2:722]
    validation_data_decompressed = validation_arr[:, 722:-1]
    validation_checksum = validation_arr[:, -1]

    ## Act
    ds = process_star_sensor(star_sensor_ds)

    ## Assert
    # 45 times and 720 count values
    assert ds["data"].shape == (45, 720)
    assert ds["data"].dtype == np.uint16

    # We are only spot checking a few values from the validation file
    # the first 3 and the final value.
    small_ds = ds.isel(epoch=[0, 1, 2, -1])
    assert len(small_ds["epoch"]) == 4
    np.testing.assert_array_equal(small_ds["shcoarse"], validation_shcoarse)
    np.testing.assert_array_equal(small_ds["count"], validation_count)
    np.testing.assert_array_equal(small_ds["data"], validation_data_decompressed)
    np.testing.assert_array_equal(small_ds["chksum"], validation_checksum)
