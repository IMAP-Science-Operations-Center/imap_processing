import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo.l0.lo_science import parse_events


@pytest.fixture()
def fake_de_dataset():
    # DE One
    absent_1 = "0000"  # case 0
    time_1 = "000001100100"  # 100
    energy_1 = "010"  # 2
    mode_1 = "1"
    tof0_1 = "0000000000"
    # TOF1 not transmitted
    tof2_1 = "000000010"  # 2
    tof3_1 = "000011"  # 3
    cksm_1 = "0000"  # 0
    # POS not transmitted

    # DE Two
    absent_2 = "1010"  # case 10
    time_2 = "000001100100"  # 100
    energy_2 = "010"  # 2
    mode_2 = "1"
    # TOF0 not transmitted
    tof1_2 = "000000001"  # 1
    # TOF2, TOF3, CKSM not transmitted
    pos_2 = "00"  # 0

    de_data = (
        absent_1
        + time_1
        + energy_1
        + mode_1
        + tof0_1
        + tof2_1
        + tof3_1
        + cksm_1
        + absent_2
        + time_2
        + energy_2
        + mode_2
        + tof1_2
        + pos_2
    )
    dataset = xr.Dataset(
        data_vars=dict(
            count=(["time"], np.array([2])),
            data=(["time"], np.array([de_data])),
        )
    )

    return dataset


@pytest.fixture()
def attr_mgr():
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", "v000")
    return attr_mgr


def test_parse_events(fake_de_dataset, attr_mgr):
    # Act
    dataset = parse_events(fake_de_dataset, attr_mgr)

    # Assert
    np.testing.assert_array_equal(dataset["coincidence_type"].values, np.array([0, 10]))
    np.testing.assert_array_equal(dataset["de_time"].values, np.array([100, 100]))
    np.testing.assert_array_equal(dataset["esa_step"].values, np.array([2, 2]))
    np.testing.assert_array_equal(dataset["mode"].values, np.array([1, 1]))
    np.testing.assert_array_equal(dataset["tof0"].values, np.array([0 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof1"].values, np.array([65535, 1 << 1]))
    np.testing.assert_array_equal(dataset["tof2"].values, np.array([2 << 1, 65535]))
    np.testing.assert_array_equal(dataset["tof3"].values, np.array([3 << 1, 65535]))
    np.testing.assert_array_equal(dataset["cksm"].values, np.array([0 << 1, 255]))
    np.testing.assert_array_equal(dataset["pos"].values, np.array([255, 0]))
