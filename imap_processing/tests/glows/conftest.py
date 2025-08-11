from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from imap_processing.glows.l0 import decom_glows
from imap_processing.glows.l1a.glows_l1a import glows_l1a, process_de_l0
from imap_processing.glows.l1a.glows_l1a_data import HistogramL1A
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l1b.glows_l1b_data import (
    AncillaryExclusions,
    AncillaryParameters,
)
from imap_processing.glows.l2.glows_l2 import glows_l2


@pytest.fixture
def packet_path():
    current_directory = Path(__file__).parent
    return current_directory / "validation_data" / "glows_test_packet_20110921_v01.pkts"


@pytest.fixture
def decom_test_data(packet_path):
    """Read test data from file"""

    data_packet_list = decom_glows.decom_packets(packet_path)
    return data_packet_list


@pytest.fixture
def l1a_test_data(decom_test_data):
    hist_l1a = []

    for hist in decom_test_data[0]:
        hist_l1a.append(HistogramL1A(hist))

    de_l1a = process_de_l0(decom_test_data[1])

    return hist_l1a, de_l1a


@pytest.fixture
def l1a_dataset(packet_path):
    return glows_l1a(packet_path)


@pytest.fixture
def l1b_hist_dataset(l1a_dataset, mock_ancillary_exclusions):
    return glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
    )


@pytest.fixture
def l2_hist_dataset(l1b_datasets):
    return glows_l2(l1b_datasets)


@pytest.fixture
def mock_ancillary_exclusions():
    """Create a mock AncillaryExclusions object for testing."""
    # Create mock datasets with epoch dimension that can be used with limit_by_day()
    # Use a range of dates that covers the expected test data dates
    epoch_range = np.arange(
        np.datetime64("2010-01-01"), np.datetime64("2010-12-31"), dtype="datetime64[D]"
    )

    # Create datasets with epoch dimension and some mock data
    mock_excluded_regions = xr.Dataset(
        {
            "ecliptic_longitude_deg": (
                ["epoch", "region"],
                np.random.rand(len(epoch_range), 5),
            ),
            "ecliptic_latitude_deg": (
                ["epoch", "region"],
                np.random.rand(len(epoch_range), 5),
            ),
        },
        coords={"epoch": epoch_range},
    )

    mock_uv_sources = xr.Dataset(
        {
            "object_name": (
                ["epoch", "source"],
                [["star1", "star2", "star3"]] * len(epoch_range),
            ),
            "ecliptic_longitude_deg": (
                ["epoch", "source"],
                np.random.rand(len(epoch_range), 3),
            ),
            "ecliptic_latitude_deg": (
                ["epoch", "source"],
                np.random.rand(len(epoch_range), 3),
            ),
            "angular_radius_for_masking": (
                ["epoch", "source"],
                np.random.rand(len(epoch_range), 3),
            ),
        },
        coords={"epoch": epoch_range},
    )

    mock_suspected_transients = xr.Dataset(
        {
            "l1b_unique_block_identifier": (
                ["epoch", "time_block"],
                [["block1", "block2"]] * len(epoch_range),
            ),
            "histogram_mask_array": (
                ["epoch", "time_block"],
                [["mask1", "mask2"]] * len(epoch_range),
            ),
        },
        coords={"epoch": epoch_range},
    )

    mock_exclusions_by_instr_team = xr.Dataset(
        {
            "l1b_unique_block_identifier": (
                ["epoch", "time_block"],
                [["block1", "block2"]] * len(epoch_range),
            ),
            "histogram_mask_array": (
                ["epoch", "time_block"],
                [["mask1", "mask2"]] * len(epoch_range),
            ),
        },
        coords={"epoch": epoch_range},
    )

    return AncillaryExclusions(
        excluded_regions=mock_excluded_regions,
        uv_sources=mock_uv_sources,
        suspected_transients=mock_suspected_transients,
        exclusions_by_instr_team=mock_exclusions_by_instr_team,
    )


@pytest.fixture
def mock_ancillary_parameters():
    """Create a mock AncillaryParameters object for testing."""
    mock_table = {
        "description": "Table for conversion/decoding ancillary parameters collected "
        "onboard by IMAP/GLOWS",
        "version": "0.1",
        "date_of_creation_yyyymmdd": "20230527",
        "filter_temperature": {
            "min": -30.0,
            "max": 80.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "hv_voltage": {
            "min": 0.0,
            "max": 3500.0,
            "n_bits": 12,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
        "spin_period": {"min": 0.0, "max": 20.9712, "n_bits": 16},
        "spin_phase": {"min": 0.0, "max": 360.0, "n_bits": 16},
        "pulse_length": {
            "min": 0.0,
            "max": 255.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
        },
    }
    return AncillaryParameters(mock_table)
