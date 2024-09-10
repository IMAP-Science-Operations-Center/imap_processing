import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.process_hit import (
    HIT_PREFIX_TO_RATE_TYPE,
    create_l1,
    find_groups,
    process_hit,
)
from imap_processing.utils import packet_file_to_datasets


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
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "test_data"
        / "l0"
        / "hit_ialirt_sample.csv"
    )
    data = pd.read_csv(data_path, na_values=[" ", ""]).astype("float")

    return data


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_hit_path):
    """Create xarray data"""
    apid = 1253

    xarray_data = packet_file_to_datasets(binary_packet_path, xtce_hit_path)[apid]
    return xarray_data


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
    assert HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_1"] == expected_fast_rate_1
    assert HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_2"] == expected_fast_rate_2
    assert HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"] == expected_slow_rate


def test_find_groups(xarray_data):
    """Tests find_groups"""

    filtered_data = find_groups(xarray_data)

    np.testing.assert_array_equal(
        filtered_data["hit_subcom"], np.tile(np.arange(60), 15)
    )


def test_create_l1(xarray_data):
    """Tests create_l1"""

    filtered_data = find_groups(xarray_data)

    fast_rate_1 = filtered_data["hit_fast_rate_1"][(filtered_data["group"] == 4).values]
    fast_rate_2 = filtered_data["hit_fast_rate_2"][(filtered_data["group"] == 4).values]
    slow_rate = filtered_data["hit_slow_rate"][(filtered_data["group"] == 4).values]

    l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

    assert l1["L1A_TRIG_08"] == 39
    assert l1["L3A_TRIG_10"] == 7
    assert l1["IB_EVNT_TRIG_07"] == 6
    assert l1["L4IBHG"] == 2


def test_process_hit(xarray_data):
    """Tests process_hit."""

    # Tests that it functions normally
    hit_product = process_hit(xarray_data)
    assert len(hit_product) == 15

    # Make a subset of data that has values to check the calculations of process hit.
    indices = (xarray_data["hit_met"] != 0).values.nonzero()[0]
    xarray_data["hit_slow_rate"].values[indices[0] : indices[0] + 60] = 2
    subset = xarray_data.isel(epoch=slice(indices[0], indices[0] + 60))

    hit_product = process_hit(subset)

    assert hit_product[0]["hit_lo_energy_e_A_side"] == 4
    assert hit_product[0]["hit_medium_energy_e_A_side"] == 4
    assert hit_product[0]["hit_low_energy_e_B_side"] == 4
    assert hit_product[0]["hit_high_energy_e_B_side"] == 2
    assert hit_product[0]["hit_medium_energy_H_omni"] == 4
    assert hit_product[0]["hit_high_energy_He_omni"] == 2

    # Create a scrambled set of subcom values.
    xarray_data["hit_subcom"].values[indices[0] : indices[0] + 60] = [
        i for i in range(29) for _ in range(2)
    ] + [59, 59]

    with pytest.raises(
        ValueError, match="does not contain all values from 0 to 59 without duplicates"
    ):
        process_hit(subset)


def test_decom_packets(xarray_data, hit_test_data):
    """This function checks that all instrument parameters are accounted for."""

    fast_rate_1 = xarray_data["hit_fast_rate_1"]
    fast_rate_2 = xarray_data["hit_fast_rate_2"]
    slow_rate = xarray_data["hit_slow_rate"]

    groups = np.arange(15)

    for group in groups:
        # The sequence begins where "HIT_MET" != 0
        start_index = hit_test_data.index[hit_test_data["MET"] != 0][group]

        # Test Fast Rate 1
        ccsds_fast_rate_1 = fast_rate_1[start_index : start_index + 60]
        test_fast_rate_1 = hit_test_data.loc[
            start_index : start_index + 59,
            hit_test_data.columns.str.startswith(
                tuple(HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_1"])
            ),
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
            hit_test_data.columns.str.startswith(
                tuple(HIT_PREFIX_TO_RATE_TYPE["FAST_RATE_2"])
            ),
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
            hit_test_data.columns.isin(HIT_PREFIX_TO_RATE_TYPE["SLOW_RATE"]),
        ]
        flat_test_slow_rate = test_slow_rate.to_numpy().flatten()
        np.testing.assert_array_equal(
            ccsds_slow_rate.values, flat_test_slow_rate[~np.isnan(flat_test_slow_rate)]
        )
