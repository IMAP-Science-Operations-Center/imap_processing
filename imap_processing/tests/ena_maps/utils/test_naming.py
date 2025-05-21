from datetime import timedelta

import pytest

from imap_processing.ena_maps.utils.naming import (
    MappableInstrumentShortName,
    build_friendly_date_descriptor,
    build_l2_map_descriptor,
    ns_to_duration_months,
)
from imap_processing.spice.geometry import SpiceFrame


def test_build_l2_map_descriptor_with_timedelta():
    descriptor = build_l2_map_descriptor(
        instrument=MappableInstrumentShortName.HI,
        frame="hf",
        resolution_str="2deg",
        duration=timedelta(days=60),
        sensor="45",
        principal_data="ena",
        species="he",
        survival_corrected="sp",
        spin_phase="ram",
        coordinate_system="hgi",
    )
    assert descriptor == "h45-ena-he-hf-sp-ram-hgi-2deg-2mo"


def test_build_l2_map_descriptor_with_int_duration():
    descriptor = build_l2_map_descriptor(
        instrument=MappableInstrumentShortName.LO_HI_THROUGHPUT,
        frame="sf",
        resolution_str="4deg",
        duration=120,
        sensor=75,
        principal_data="ena",
        species="o",
        survival_corrected="nsp",
        spin_phase="anti",
        coordinate_system="rc",
    )
    assert descriptor == "t075-ena-o-sf-nsp-anti-rc-4deg-4mo"


def test_build_l2_map_descriptor_with_spice_frame():
    descriptor = build_l2_map_descriptor(
        instrument=MappableInstrumentShortName.IDEX,
        frame=SpiceFrame.IMAP_DPS,
        resolution_str="nside32",
        duration=365,
        principal_data="drt",
        species="dust",
        survival_corrected="sp",
        spin_phase="full",
        coordinate_system="hae",
    )
    assert descriptor == "idx-drt-dust-sf-sp-full-hae-nside32-1yr"


def test_build_l2_map_descriptor_invalid_frame():
    with pytest.raises(NotImplementedError):
        build_l2_map_descriptor(
            instrument=MappableInstrumentShortName.IDEX,
            frame=SpiceFrame.IMAP_GLOWS,
            resolution_str="4deg",
            duration="6mo",
            sensor="",
            principal_data="ena",
            species="uv",
            survival_corrected="nsp",
            spin_phase="ram",
            coordinate_system="hgi",
        )


# Parameterize over the start_datestring
@pytest.mark.parametrize(
    "start_datestring",
    [
        "2026-05",
        "2026-05-15",
        "2026-05-15T12:00:00",
    ],
)
def test_build_friendly_date_descriptor(start_datestring):
    friendly_date_descriptor = build_friendly_date_descriptor(
        start_datestring=start_datestring,
        duration_months=3,
    )
    assert friendly_date_descriptor == "202605m03"


def test_ns_to_duration_months():
    days_per_avg_year = 365.25
    ns_per_day = 24 * 60 * 60 * 1e9

    for fraction_of_year, expected_months in [
        (1 / 2, 6),
        (1 / 3, 4),
        (1 / 4, 3),
        (3 / 4, 9),
    ]:
        assert (
            ns_to_duration_months(fraction_of_year * days_per_avg_year * ns_per_day)
            == expected_months
        )
