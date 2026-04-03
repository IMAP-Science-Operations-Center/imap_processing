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
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve


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
def l1b_hist_dataset(
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
):
    return glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
        mock_pipeline_settings,
        mock_conversion_table_dict,
    )


@pytest.fixture
def l2_hist_dataset(l1b_hist_dataset, mock_pipeline_settings):
    return glows_l2(l1b_hist_dataset, mock_pipeline_settings)


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
            # degrees in [0, 360)
            "ecliptic_longitude_deg": (
                ["epoch", "source"],
                np.tile(
                    np.array([202.0812, 120.0, 250.0], dtype=np.float64),
                    (len(epoch_range), 1),
                ),
            ),
            # degrees in [-90, 90]
            "ecliptic_latitude_deg": (
                ["epoch", "source"],
                np.tile(
                    np.array([18.4119, 0.0, 35.0], dtype=np.float64),
                    (len(epoch_range), 1),
                ),
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
            # degrees in [0, 360)
            "ecliptic_longitude_deg": (
                ["epoch", "source"],
                np.tile(
                    np.array([202.0812, 120.0, 250.0], dtype=np.float64),
                    (len(epoch_range), 1),
                ),
            ),
            # degrees in [-90, 90]
            "ecliptic_latitude_deg": (
                ["epoch", "source"],
                np.tile(
                    np.array([18.4119, 0.0, 35.0], dtype=np.float64),
                    (len(epoch_range), 1),
                ),
            ),
            # masking radius in degrees
            "angular_radius_for_masking": (
                ["epoch", "source"],
                np.tile(
                    np.array([2.0, 0.0, 0.0], dtype=np.float64), (len(epoch_range), 1)
                ),
            ),
        },
        coords={"epoch": epoch_range},
    )

    # Mask array based on data in imap_glows_suspected-transients_20250923_v002.dat.
    mock_suspected_transients = xr.Dataset(
        {
            "l1b_unique_block_identifier": (
                ["epoch", "time_block"],
                [["2026-01-01T15:00:00", "2026-01-01T15:01:00"]] * len(epoch_range),
            ),
            "histogram_mask_array": (
                ["epoch", "time_block"],
                [["0" * 3600, "0" * 600 + "1" * 100 + "0" * 2900]] * len(epoch_range),
            ),
        },
        coords={"epoch": epoch_range},
    )

    mock_exclusions_by_instr_team = xr.Dataset(
        {
            "l1b_unique_block_identifier": (
                ["epoch", "time_block"],
                [["2026-01-01T15:00:00", "2026-01-01T15:01:00"]] * len(epoch_range),
            ),
            "histogram_mask_array": (
                ["epoch", "time_block"],
                [["0" * 100 + "1" * 10 + "0" * 3490, "0" * 3600]] * len(epoch_range),
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
def mock_ancillary_parameters(mock_conversion_table_dict):
    """Create a mock AncillaryParameters object for testing."""

    return AncillaryParameters(mock_conversion_table_dict)


@pytest.fixture
def mock_conversion_table_dict():
    """Create a mock conversion table dataset for testing.

    This aligns with the validation output for GLOWS unit testing."""

    mock_dict = {
        "description": "Table for conversion/decoding ancillary parameters",
        "version": "v001",
        "date_of_creation_yyyymmdd": "20230527",
        "filter_temperature": {
            "min": -30.0,
            "max": 80.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
            "physical_unit": "Celsius degree",
        },
        "hv_voltage": {
            "min": 0.0,
            "max": 3500.0,
            "n_bits": 12,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
            "physical_unit": "Celsius degree",
        },
        "spin_period": {
            "min": 0.0,
            "max": 20.9712,
            "n_bits": 16,
            "physical_unit": "Celsius degree",
        },
        "spin_phase": {
            "min": 0.0,
            "max": 360.0,
            "n_bits": 16,
            "physical_unit": "Celsius degree",
        },
        "pulse_length": {
            "min": 0.0,
            "max": 255.0,
            "n_bits": 8,
            "p01": 0.0,
            "p02": 0.0,
            "p03": 0.0,
            "p04": 0.0,
            "physical_unit": "Celsius degree",
        },
    }

    return mock_dict


@pytest.fixture
def mock_calibration_dataset():
    """Create a mock CalibrationDataset object for testing."""
    return xr.Dataset(
        {"cps_per_r": xr.DataArray([0.849, 1.020], dims=["start_time_utc"])},
        coords={
            "start_time_utc": np.array(
                ["2011-09-19T09:58:04", "2011-09-20T18:12:48"], dtype="datetime64[s]"
            )
        },
    )


@pytest.fixture
def mock_pipeline_settings():
    """Create a mock PipelineSettings dataset for testing."""
    # Create a mock dataset with pipeline settings data which matches the output
    # from GlowsAncillaryCombiner
    epoch_range = np.arange(
        np.datetime64("2010-01-01"), np.datetime64("2010-12-31"), dtype="datetime64[D]"
    )

    mock_pipeline_dataset = xr.Dataset(
        {
            "active_bad_angle_flags": (
                ["epoch", "flag_index"],
                np.tile([True, True, True, True], (len(epoch_range), 1)),
            ),
            "active_bad_time_flags": (
                ["epoch", "time_flag_index"],
                np.tile(
                    [True] * 17, (len(epoch_range), 1)
                ),  # 17 bad time flags from the JSON
            ),
            "sunrise_offset": (["epoch"], [0.0] * len(epoch_range)),
            "sunset_offset": (["epoch"], [0.0] * len(epoch_range)),
            "n_sigma_threshold_lower": (["epoch"], [3.0] * len(epoch_range)),
            "n_sigma_threshold_upper": (["epoch"], [3.0] * len(epoch_range)),
            "relative_difference_threshold": (["epoch"], [7.0e-5] * len(epoch_range)),
            "std_dev_threshold__celsius_deg": (["epoch"], [2.03] * len(epoch_range)),
            "std_dev_threshold__volt": (["epoch"], [50.0] * len(epoch_range)),
            "std_dev_threshold__sec": (["epoch"], [0.033333] * len(epoch_range)),
            "std_dev_threshold__usec": (["epoch"], [1.0] * len(epoch_range)),
            "angular_radius_for_excl_regions__deg": (
                ["epoch"],
                [2.0] * len(epoch_range),
            ),
            "number_of_good_histograms_at_night": (["epoch"], [3] * len(epoch_range)),
            "l3a_nominal_number_of_bins": (["epoch"], [90] * len(epoch_range)),
        },
        coords={"epoch": epoch_range},
    )

    return mock_pipeline_dataset


@pytest.fixture
def mock_ecliptic_bin_centers(monkeypatch):
    """Mock ecliptic coordinates for bin centers."""

    def _mock_compute_coords(
        _data_start_time_et: float, spin_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_bins = len(spin_angle)
        return np.zeros(n_bins, dtype=float), np.zeros(n_bins, dtype=float)

    monkeypatch.setattr(
        DailyLightcurve,
        "compute_ecliptic_coords_of_bin_centers",
        staticmethod(_mock_compute_coords),
    )


def mock_update_spice_parameters(self, *args, **kwargs):
    self.spin_period_ground_average = np.float64(0.0)
    self.spin_period_ground_std_dev = np.float64(0.0)
    self.position_angle_offset_average = np.float64(0.0)
    self.position_angle_offset_std_dev = np.float64(0.0)
    self.spin_axis_orientation_average = np.zeros(2, dtype=np.float64)
    self.spin_axis_orientation_std_dev = np.zeros(2, dtype=np.float64)
    self.spacecraft_location_average = np.zeros(3, dtype=np.float64)
    self.spacecraft_location_std_dev = np.zeros(3, dtype=np.float64)
    self.spacecraft_velocity_average = np.zeros(3, dtype=np.float64)
    self.spacecraft_velocity_std_dev = np.zeros(3, dtype=np.float64)
