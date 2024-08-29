import pytest

from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.decom_ialirt import generate_xarray
from imap_processing.ialirt.l0.process_hit import (
    HITPrefixes,
    create_l1,
    find_groups,
    process_hit,
)


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


@pytest.fixture()
def xarray_data(binary_packet_path, xtce_hit_path):
    """Create xarray data"""
    apid = 1253

    xarray_data = generate_xarray(binary_packet_path, xtce_hit_path)[apid]
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
    assert HITPrefixes.FAST_RATE_1.value == expected_fast_rate_1
    assert HITPrefixes.FAST_RATE_2.value == expected_fast_rate_2
    assert HITPrefixes.SLOW_RATE.value == expected_slow_rate


def test_find_groups(xarray_data):
    """Tests find_groups"""

    filtered_data = find_groups(xarray_data)

    assert filtered_data["hit_subcom"].values[0] == 0
    assert filtered_data["hit_subcom"].values[-1] == 59
    assert len(filtered_data["hit_subcom"]) / 60 == 15
    assert len(filtered_data["hit_sc_tick"][filtered_data["hit_subcom"] == 0]) == 15
    assert len(filtered_data["hit_sc_tick"][filtered_data["hit_subcom"] == 59]) == 15


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


def test_process_hit(xarray_data, caplog):
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

    # Check if the logger was called with the expected message
    with caplog.at_level("INFO"):
        process_hit(subset)
        assert any(
            "Incorrect number of packets" in record.message for record in caplog.records
        )
