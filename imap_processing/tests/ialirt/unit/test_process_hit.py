import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.hit.hit_utils import HitAPID, get_datasets_by_apid
from imap_processing.hit.l0.decom_hit import decom_hit
from imap_processing.hit.l1b.hit_l1b import livetime_fraction_calculation
from imap_processing.ialirt.l0.process_hit import (
    HIT_PREFIX_TO_RATE_TYPE,
    calculate_livetime_fraction,
    create_l1,
    process_hit,
)
from imap_processing.ialirt.utils.grouping import find_groups
from imap_processing.ialirt.utils.time import calculate_time
from imap_processing.utils import packet_file_to_datasets


@pytest.fixture(scope="session")
def xtce_hit_path():
    """Returns the xtce auxiliary directory."""
    return imap_module_directory / "ialirt" / "packet_definitions" / "ialirt_hit.xml"


@pytest.fixture(scope="session")
def binary_packet_path():
    """Returns the xtce directory."""
    return (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "hit_ialirt_sample.ccsds"
    )


@pytest.fixture(scope="session")
def hit_test_data():
    """Returns the test data directory."""
    data_path = (
        imap_module_directory
        / "tests"
        / "ialirt"
        / "data"
        / "l0"
        / "hit_ialirt_sample.csv"
    )
    data = pd.read_csv(data_path, na_values=[" ", ""]).astype("float")

    return data


@pytest.fixture
def xarray_data(binary_packet_path, xtce_hit_path):
    """Create xarray data."""
    apid = 1253

    xarray_data = packet_file_to_datasets(binary_packet_path, xtce_hit_path)[apid]
    return xarray_data


@pytest.fixture
def hit_ialirt_and_science_20260707():
    """I-ALiRT packets and HIT science packets covering the same minutes.

    I-ALiRT: 5 one-minute packet files from 2026-07-07 17:54-17:58 UTC.
    Science: the HIT science frames covering the same time, cut
    from imap_hit_l0_raw_20260707_v001.0001.pkts.
    """
    directory = imap_module_directory / "tests" / "ialirt" / "data" / "l0"
    xtce = imap_module_directory / "ialirt" / "packet_definitions" / "ialirt.xml"
    filenames = [
        "iois_1_packets_2026_188_17_54_04",
        "iois_1_packets_2026_188_17_55_05",
        "iois_1_packets_2026_188_17_56_06",
        "iois_1_packets_2026_188_17_57_07",
        "iois_1_packets_2026_188_17_58_08",
    ]
    ialirt = xr.concat(
        [
            packet_file_to_datasets(directory / f, xtce, use_derived_value=False)[478]
            for f in filenames
        ],
        dim="epoch",
    )
    science = decom_hit(
        get_datasets_by_apid(
            str(directory / "hit_science_20260707_ialirt_overlap.pkts")
        )[HitAPID.HIT_SCIENCE]
    )
    return ialirt, science


@pytest.mark.external_test_data
def test_process_spacecraft_packet(sc_packet_path):
    """Tests Spacecraft Packet processing."""
    packet_path, xtce_ialirt_path = sc_packet_path
    sc_xarray_data = packet_file_to_datasets(
        packet_path, xtce_ialirt_path, use_derived_value=False
    )[478]
    hit_product = process_hit(sc_xarray_data)

    assert len(hit_product[0].keys()) == 18


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


def test_create_l1(xarray_data):
    """Tests create_l1() function."""

    # Add a dummy value to the hit_met variable.
    xarray_data["sc_sclk_sec"] = xarray_data["hit_sc_tick"]
    xarray_data["sc_sclk_sub_sec"] = (
        ("epoch",),
        np.zeros_like(xarray_data["hit_sc_tick"]),
    )

    # Subsecond time conversion specified in 7516-9054 GSW-FSW ICD.
    # Value of SCLK subseconds, unsigned, (LSB = 1/256 sec)
    met = calculate_time(
        xarray_data["sc_sclk_sec"], xarray_data["sc_sclk_sub_sec"], 256
    )

    # Add required parameters.
    xarray_data["met"] = met

    filtered_data = find_groups(xarray_data, (0, 59), "hit_subcom", "met")

    fast_rate_1 = filtered_data["hit_fast_rate_1"][(filtered_data["group"] == 4).values]
    fast_rate_2 = filtered_data["hit_fast_rate_2"][(filtered_data["group"] == 4).values]
    slow_rate = filtered_data["hit_slow_rate"][(filtered_data["group"] == 4).values]

    l1 = create_l1(fast_rate_1, fast_rate_2, slow_rate)

    assert l1["L1A_TRIG_08"] == 39
    assert l1["L3A_TRIG_10"] == 7
    assert l1["IB_EVNT_TRIG_07"] == 6
    assert l1["L4IBHG"] == 2


def test_calculate_livetime_fraction_no_livetime():
    """Tests that a group with all ERATES zero has no livetime."""
    l1 = {f"ERATE_{i}": 0 for i in range(1, 6)}
    assert np.isnan(calculate_livetime_fraction(l1))


def test_process_hit(xarray_data, caplog):
    """Tests process_hit."""

    # Add a dummy value to the hit_met variable.
    xarray_data["sc_sclk_sec"] = xarray_data["hit_sc_tick"]
    xarray_data["sc_sclk_sub_sec"] = (
        ("epoch",),
        np.zeros_like(xarray_data["hit_sc_tick"]),
    )

    # Tests that it functions normally
    hit_product = process_hit(xarray_data)
    assert len(hit_product) == 15

    assert hit_product[0]["hit_e_a_side_low_en"] == 0
    assert hit_product[0]["hit_e_a_side_med_en"] == 0
    assert hit_product[0]["hit_e_b_side_low_en"] == 0
    assert hit_product[0]["hit_e_b_side_high_en"] == 0
    assert hit_product[0]["hit_e_b_side_med_en"] == 0
    assert hit_product[0]["hit_he_omni_high_en"] == 0


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


@pytest.mark.external_test_data
def test_ialirt_matches_science(hit_ialirt_and_science_20260707):
    """Tests I-ALiRT HIT slow rates and livetime against HIT science data.

    Each I-ALiRT group is paired with science frames by start
    time: the I-ALiRT rates match the frame starting at HIT_MET, and the
    ERATES match the frame starting one minute earlier.
    """
    ialirt, science = hit_ialirt_and_science_20260707

    ialirt["met"] = calculate_time(
        ialirt["sc_sclk_sec"], ialirt["sc_sclk_sub_sec"], 256
    )
    groups = find_groups(ialirt, (0, 59), "hit_subcom", "met")

    # Science frames are 20 packets; start time = sc_tick of the first packet.
    frame_start = science["sc_tick"].values[::20]
    science_livetime = livetime_fraction_calculation(science["livetime_counter"])

    n_groups = 0
    for group in np.unique(groups["group"]):
        mask = (groups["group"] == group).values
        if not np.array_equal(groups["hit_subcom"][mask], np.arange(60)):
            continue
        n_groups += 1
        slow_rate = groups["hit_slow_rate"][mask]
        l1 = create_l1(
            groups["hit_fast_rate_1"][mask], groups["hit_fast_rate_2"][mask], slow_rate
        )
        hit_met = int(groups["hit_met"][mask].values[0])
        same = int(np.flatnonzero(frame_start == hit_met)[0])
        # livetime come from 1 minute earlier
        prev = int(np.flatnonzero(frame_start == hit_met - 60)[0])

        # ERATES and livetime fraction match the previous science minute.
        assert l1["ERATE_1"] == science["livetime_counter"].values[prev]
        assert l1["ERATE_2"] == science["num_trig"].values[prev]
        assert l1["ERATE_3"] == science["num_reject"].values[prev]
        assert calculate_livetime_fraction(l1) == pytest.approx(
            float(science_livetime.values[prev])
        )

        # I-ALiRT rates match the same science minute, within 1 count (the
        # I-ALiRT values are read just before the minute closes). Rate 3 is
        # skipped: onboard it duplicates Rate 2 (open question with HIT team).
        ialirt_rates = np.array([l1[f"IALRT_RATE_{i}"] for i in range(1, 21)])
        keep = np.arange(20) != 2
        np.testing.assert_allclose(
            ialirt_rates[keep], science["ialirtrates"].values[same][keep], atol=1
        )

    assert n_groups == 4
