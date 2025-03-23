from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.spacecraft import quaternions


@pytest.fixture()
def l1a_quaternion_ds():
    time = np.arange(100)
    data = np.linspace(-0.1, 0.1, 100)
    data_vars = {"SCIENCEDATA1HZ_QUAT_10_HZ_TIME".lower(): time}
    for i in range(4):
        for j in range(10):
            index = i * 10 + j
            # 0-0.1, 0.1-0.2, 0.2-0.3, ... for each variable
            data_vars[f"FSW_ACS_QUAT_10_HZ_BUFFERED_{index}".lower()] = data * j

    ds = xr.Dataset(data_vars=data_vars)
    return ds


def test_quaternion_packet_file():
    # Single packet extracted with the following command:
    # head --bytes 196 SSR_2024_190_20_08_12_0483851794_2_DA_apid0594.pkts > output.pkts
    packet_file = (
        Path(__file__).parent
        / "data"
        / "SSR_2024_190_20_08_12_0483851794_2_DA_apid0594_1packet.pkts"
    )
    l1a_ds = quaternions.load_quaternion_packets(packet_file)

    # 1 packet
    assert len(l1a_ds["epoch"]) == 1

    # Make sure we have all of the expected variables
    for i in range(40):
        var = f"FSW_ACS_QUAT_10_HZ_BUFFERED_{i}".lower()
        assert var in l1a_ds.data_vars
        # Make sure it is a float32 between -1 and 1
        assert l1a_ds[var].dtype == np.float32
        assert -1 <= l1a_ds[var].values[0] <= 1


def test_quaternion_unpacking(l1a_quaternion_ds):
    l1b_ds = quaternions.assemble_quaternions(l1a_quaternion_ds)

    # Should be 10x the amount of data we got from the input dataset
    # input was 0, 1, 2, ...
    # expected is 0, 0.1, 0.2, ...
    np.testing.assert_allclose(l1b_ds["epoch"], np.arange(0, 100, 0.1))

    data = np.linspace(-0.1, 0.1, 100)
    for var in ["quat_x", "quat_y", "quat_z", "quat_s"]:
        # 10 increasing values, then reset and shift right by 0.2 / 99 for the linspace
        # -0., -0.1, -0.2, ..., -0.9, -0., -0.09, -0.19, -0.29, ...
        expected = np.stack([data * i for i in range(10)], axis=1).ravel()
        np.testing.assert_allclose(l1b_ds[var], expected)


def test_process_quaternions():
    packet_file = (
        Path(__file__).parent
        / "data"
        / "SSR_2024_190_20_08_12_0483851794_2_DA_apid0594_1packet.pkts"
    )
    l1a_ds, l1b_ds = quaternions.process_quaternions(packet_file)
    assert l1a_ds.attrs["Data_level"] == "1A"
    assert l1b_ds.attrs["Data_level"] == "1B"
    assert len(l1a_ds["epoch"]) == 1
    assert len(l1b_ds["epoch"]) == 10
