import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.ialirt.l0.decom_ialirt import generate_xarray


@pytest.fixture(scope="session")
def xtce_hit_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_hit.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce auxiliary directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "hit_ialirt_sample.ccsds"
    )


@pytest.fixture(scope="session")
def hit_test_data():
    """Returns the xtce auxiliary directory."""
    header_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "hit_ialirt_sample_header.csv"
    )
    header = list(pd.read_csv(header_path).columns)

    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "hit_ialirt_sample.csv"
    )
    data = pd.read_csv(
        data_path, header=None, names=header, na_values=[" ", ""]
    ).astype("float")

    return data


@pytest.fixture()
def decom_packets_data(binary_packet_path, xtce_hit_path):
    """Read packet data from file using decom_packets"""
    data_packet_list = decom_packets(binary_packet_path, xtce_hit_path)
    return data_packet_list


def test_length(decom_packets_data, hit_test_data):
    """Test if total packets in data file is correct"""
    assert len(decom_packets_data) == len(hit_test_data)


def test_decom_packets(binary_packet_path, xtce_hit_path, hit_test_data):
    """This function checks that all instrument parameters are accounted for."""

    xarray_data = generate_xarray(
        binary_packet_path, xtce_hit_path, time_keys={"HIT": "HIT_SC_TICK"}
    )
    fast_rate_1 = xarray_data["HIT"]["HIT_FAST_RATE_1"]
    fast_rate_2 = xarray_data["HIT"]["HIT_FAST_RATE_2"]
    slow_rate = xarray_data["HIT"]["HIT_SLOW_RATE"]

    # Define test data names
    prefixes = ["L1A_TRIG", "IA_EVNT_TRIG", "A_EVNT_TRIG", "L3A_TRIG"]
    HIT_FAST_RATE_1 = [f"{prefix}_{i:02d}" for i in range(15) for prefix in prefixes]

    prefixes = ["L1B_TRIG", "IB_EVNT_TRIG", "B_EVNT_TRIG", "L3B_TRIG"]
    HIT_FAST_RATE_2 = [f"{prefix}_{i:02d}" for i in range(15) for prefix in prefixes]

    HIT_SLOW_RATE = [
        "L1A",
        "L2A",
        "L3A",
        "L1A0AHG",
        "L1B0AHG",
        "L1C0AHG",
        "L4IAHG",
        "L4OAHG",
    ]
    HIT_SLOW_RATE.extend(["SLOW_RATE"] * 3)
    HIT_SLOW_RATE.extend(["L1A0BHG", "L1B0BHG", "L1C0BHG", "L4IBHG", "L4OBHG"])
    HIT_SLOW_RATE.extend([f"IALRT_RATE_{i}" for i in range(1, 21)])
    HIT_SLOW_RATE.extend(
        ["TRIG_IA_EVNT", "TRIG_IB_EVNT", "NASIDE_IALRT", "NBSIDE_IALRT"]
    )
    HIT_SLOW_RATE.extend([f"ERATE_{i}" for i in range(1, 6)])
    HIT_SLOW_RATE.extend(["L12A", "L123A", "PENA", "L12B", "L123B", "PENB"])
    HIT_SLOW_RATE.extend(["SLOW_RATE"] * 4)
    HIT_SLOW_RATE.extend(["H_06_08", "H_12_15", "H_15_70", "HE4_06_08", "HE4_15_70"])

    # The sequence begins where "HIT_MET" != 0
    start_index = hit_test_data.index[hit_test_data["MET"] != 0][0]
    ccsds_fast_rate_1 = fast_rate_1[start_index : start_index + 60]

    filtered_data = hit_test_data.loc[start_index : start_index + 59]
    mask = filtered_data.columns.str.startswith(tuple(HIT_FAST_RATE_1))
    selected_columns = filtered_data.columns[mask]
    filtered_data_2 = filtered_data[selected_columns]
    data_array = filtered_data_2.to_numpy()
    flattened_array = data_array.flatten()
    cleaned_array = flattened_array[~np.isnan(flattened_array)]

    np.testing.assert_array_equal(ccsds_fast_rate_1.values, cleaned_array)
