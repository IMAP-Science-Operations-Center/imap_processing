import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets
from imap_processing.ialirt.l0.decom_ialirt import generate_xarray
from imap_processing.ialirt.l0.process_hit import HITPrefixes, create_l1, process_hit


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


def generate_prefixes(prefixes):
    return [f"{prefix}_{i:02d}" for i in range(15) for prefix in prefixes]


def test_prefixes():
    """Tests HITPrefixes Enum"""
    expected_fast_rate_1 = generate_prefixes(
        ["L1A_TRIG", "IA_EVNT_TRIG", "A_EVNT_TRIG", "L3A_TRIG"]
    )
    expected_fast_rate_2 = generate_prefixes(
        ["L1B_TRIG", "IB_EVNT_TRIG", "B_EVNT_TRIG", "L3B_TRIG"]
    )
    expected_slow_rate = [
        "L1A",
        "L2A",
        "L3A",
        "L1A0AHG",
        "L1B0AHG",
        "L1C0AHG",
        "L4IAHG",
        "L4OAHG",
        "SLOW_RATE_08",
        "SLOW_RATE_09",
        "SLOW_RATE_10",
        "L1A0BHG",
        "L1B0BHG",
        "L1C0BHG",
        "L4IBHG",
        "L4OBHG",
        *[f"IALRT_RATE_{i}" for i in range(1, 21)],
        "TRIG_IA_EVNT",
        "TRIG_IB_EVNT",
        "NASIDE_IALRT",
        "NBSIDE_IALRT",
        *[f"ERATE_{i}" for i in range(1, 6)],
        "L12A",
        "L123A",
        "PENA",
        "L12B",
        "L123B",
        "PENB",
        "SLOW_RATE_51",
        "SLOW_RATE_52",
        "SLOW_RATE_53",
        "SLOW_RATE_54",
        "H_06_08",
        "H_12_15",
        "H_15_70",
        "HE4_06_08",
        "HE4_15_70",
    ]

    # Perform the assertions
    assert HITPrefixes.FAST_RATE_1.value == expected_fast_rate_1
    assert HITPrefixes.FAST_RATE_2.value == expected_fast_rate_2
    assert HITPrefixes.SLOW_RATE.value == expected_slow_rate


def test_create_l1(binary_packet_path, xtce_hit_path, hit_test_data):
    """Tests create_l1"""

    xarray_data = generate_xarray(
        binary_packet_path, xtce_hit_path, time_keys={"HIT": "HIT_SC_TICK"}
    )
    fast_rate_1 = xarray_data["HIT"]["HIT_FAST_RATE_1"]
    fast_rate_2 = xarray_data["HIT"]["HIT_FAST_RATE_2"]
    slow_rate = xarray_data["HIT"]["HIT_SLOW_RATE"]

    # The sequence begins where "HIT_MET" != 0
    start_index = hit_test_data.index[hit_test_data["MET"] != 0][3]

    fast_rate_1_subset = fast_rate_1[start_index : start_index + 60]
    fast_rate_2_subset = fast_rate_2[start_index : start_index + 60]
    slow_rate_subset = slow_rate[start_index : start_index + 60]

    l1 = create_l1(fast_rate_1_subset, fast_rate_2_subset, slow_rate_subset)

    assert l1["L1A_TRIG_08"] == 39
    assert l1["L3A_TRIG_10"] == 7
    assert l1["IB_EVNT_TRIG_07"] == 6
    assert l1["L4IBHG"] == 2


def test_process_hit(binary_packet_path, xtce_hit_path, hit_test_data):
    """This function checks that all instrument parameters are accounted for."""

    xarray_data = generate_xarray(
        binary_packet_path, xtce_hit_path, time_keys={"HIT": "HIT_SC_TICK"}
    )
    hit_data = xarray_data["HIT"]
    non_zero_indices = (hit_data["HIT_MET"] != 0).values.nonzero()[0]
    third_non_zero_index = non_zero_indices[3]
    subset = hit_data.isel(
        HIT_SC_TICK=slice(third_non_zero_index, third_non_zero_index + 60)
    )

    hit_product = process_hit({"HIT": subset})
    assert hit_product["HIT_medium_energy_e_B_side"] == 1

    # replace values for the test to spot-check
    hit_data["HIT_SLOW_RATE"].values[
        third_non_zero_index : third_non_zero_index + 60
    ] = 2
    subset = hit_data.isel(
        HIT_SC_TICK=slice(third_non_zero_index, third_non_zero_index + 60)
    )

    hit_product_2 = process_hit({"HIT": subset})
    assert hit_product_2["HIT_lo_energy_e_A_side"] == 4
    assert hit_product_2["HIT_medium_energy_e_A_side"] == 4
    assert hit_product_2["HIT_low_energy_e_B_side"] == 4
    assert hit_product_2["HIT_high_energy_e_B_side"] == 2
    assert hit_product_2["HIT_medium_energy_H_omni"] == 4
    assert hit_product_2["HIT_high_energy_He_omni"] == 2


def test_decom_packets(binary_packet_path, xtce_hit_path, hit_test_data):
    """This function checks that all instrument parameters are accounted for."""

    xarray_data = generate_xarray(
        binary_packet_path, xtce_hit_path, time_keys={"HIT": "HIT_SC_TICK"}
    )
    fast_rate_1 = xarray_data["HIT"]["HIT_FAST_RATE_1"]
    fast_rate_2 = xarray_data["HIT"]["HIT_FAST_RATE_2"]
    slow_rate = xarray_data["HIT"]["HIT_SLOW_RATE"]

    # The sequence begins where "HIT_MET" != 0
    start_index = hit_test_data.index[hit_test_data["MET"] != 0][3]

    # Test Fast Rate 1
    ccsds_fast_rate_1 = fast_rate_1[start_index : start_index + 60]
    test_fast_rate_1 = hit_test_data.loc[
        start_index : start_index + 59,
        hit_test_data.columns.str.startswith(tuple(HITPrefixes.FAST_RATE_1.value)),
    ]
    flat_test_fast_rate_1 = test_fast_rate_1.to_numpy().flatten()
    np.testing.assert_array_equal(
        ccsds_fast_rate_1.values,
        flat_test_fast_rate_1[~np.isnan(flat_test_fast_rate_1)],
    )

    # Test Fast Rate 2
    ccsds_fast_rate_2 = fast_rate_2[start_index : start_index + 60]
    test_fast_rate_2 = hit_test_data.loc[
        start_index : start_index + 59,
        hit_test_data.columns.str.startswith(tuple(HITPrefixes.FAST_RATE_2.value)),
    ]
    flat_test_fast_rate_2 = test_fast_rate_2.to_numpy().flatten()
    np.testing.assert_array_equal(
        ccsds_fast_rate_2.values,
        flat_test_fast_rate_2[~np.isnan(flat_test_fast_rate_2)],
    )

    # Test Slow Rate
    ccsds_slow_rate = slow_rate[start_index : start_index + 60]
    test_slow_rate = hit_test_data.loc[
        start_index : start_index + 59,
        hit_test_data.columns.isin(HITPrefixes.SLOW_RATE.value),
    ]
    flat_test_slow_rate = test_slow_rate.to_numpy().flatten()
    np.testing.assert_array_equal(
        ccsds_slow_rate.values, flat_test_slow_rate[~np.isnan(flat_test_slow_rate)]
    )
