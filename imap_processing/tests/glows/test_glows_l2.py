from unittest.mock import patch

import cdflib
import numpy as np
import pytest
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import write_cdf
from imap_processing.glows.l1b.glows_l1b import glows_l1b
from imap_processing.glows.l1b.glows_l1b_data import (
    HistogramL1B,
    PipelineSettings,
)
from imap_processing.glows.l2.glows_l2 import create_l2_dataset, glows_l2
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve, HistogramL2
from imap_processing.glows.utils.constants import GlowsConstants
from imap_processing.spice.time import et_to_datetime64, ttj2000ns_to_et
from imap_processing.tests.glows.conftest import (
    mock_update_spice_parameters,
)


@pytest.fixture
def l1b_hists():
    epoch = xr.DataArray(np.arange(4), name="epoch", dims=["epoch"])
    bins = xr.DataArray(np.arange(5), name="bins", dims=["bins"])
    hist = xr.DataArray(
        np.ones((4, 5)), dims=["epoch", "bins"], coords={"epoch": epoch, "bins": bins}
    )
    hist[1, 0] = GlowsConstants.HISTOGRAM_FILLVAL
    hist[2, 0] = GlowsConstants.HISTOGRAM_FILLVAL
    hist[1, 1] = GlowsConstants.HISTOGRAM_FILLVAL
    hist[2, 3] = GlowsConstants.HISTOGRAM_FILLVAL

    input = xr.Dataset(coords={"epoch": epoch, "bins": bins})
    input["histogram"] = hist

    return input


@patch.object(HistogramL2, "compute_position_angle", return_value=42.0)
@patch.object(
    HistogramL1B,
    "flag_uv_and_excluded",
    return_value=(np.zeros(3600, dtype=bool), np.zeros(3600, dtype=bool)),
)
@patch.object(HistogramL1B, "update_spice_parameters", autospec=True)
def test_glows_l2(
    mock_spice_function,
    mock_flag_uv_and_excluded,
    mock_compute_position_angle,
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
    mock_ecliptic_bin_centers,
    mock_calibration_dataset,
    caplog,
):
    mock_spice_function.side_effect = mock_update_spice_parameters

    l1b_hist_dataset = glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
        mock_pipeline_settings,
        mock_conversion_table_dict,
    )
    l1b_hist_dataset.attrs["Repointing"] = "repoint00047"

    # Test case 1: L1B dataset has good times
    l2 = glows_l2(l1b_hist_dataset, mock_pipeline_settings, mock_calibration_dataset)[0]
    assert l2.attrs["Logical_source"] == "imap_glows_l2_hist"
    assert np.allclose(l2["filter_temperature_average"].values, [57.6], rtol=0.1)
    assert l2["identifier"].values[0] == 47
    assert "flight_software_version" in l2.attrs
    assert "pkts_file_name" in l2.attrs
    assert "flight_software_version" not in l2.data_vars
    assert "pkts_file_name" not in l2.data_vars

    # Test case 2: L1B dataset has no good times (all flags 0)
    l1b_hist_dataset_no_good_times = l1b_hist_dataset.copy(deep=True)
    l1b_hist_dataset_no_good_times["flags"].values = np.zeros(
        l1b_hist_dataset_no_good_times.flags.shape
    )
    caplog.set_level("WARNING")
    result = glows_l2(
        l1b_hist_dataset_no_good_times, mock_pipeline_settings, mock_calibration_dataset
    )
    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)

    # Test case 3: Dataset has zero exposure and flux values
    l1b_hist_dataset_zero_values = l1b_hist_dataset.copy(deep=True)
    l1b_hist_dataset_zero_values["spin_period_average"].data[:] = 0
    l1b_hist_dataset_zero_values["number_of_spins_per_block"].data[:] = 0
    caplog.set_level("WARNING")
    result = glows_l2(
        l1b_hist_dataset_zero_values, mock_pipeline_settings, mock_calibration_dataset
    )
    assert result == []
    assert any(record.levelname == "WARNING" for record in caplog.records)


@patch.object(HistogramL2, "compute_position_angle", return_value=42.0)
@patch.object(
    HistogramL1B,
    "flag_uv_and_excluded",
    return_value=(np.zeros(3600, dtype=bool), np.zeros(3600, dtype=bool)),
)
@patch.object(HistogramL1B, "update_spice_parameters", autospec=True)
def test_generate_l2(
    mock_spice_function,
    mock_flag_uv_and_excluded,
    mock_compute_position_angle,
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
    mock_ecliptic_bin_centers,
    mock_calibration_dataset,
):
    mock_spice_function.side_effect = mock_update_spice_parameters

    l1b_hist_dataset = glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
        mock_pipeline_settings,
        mock_conversion_table_dict,
    )
    l1b_hist_dataset.attrs["Repointing"] = "repoint00047"
    day = et_to_datetime64(ttj2000ns_to_et(l1b_hist_dataset["epoch"].data[0]))
    pipeline_settings = PipelineSettings(
        mock_pipeline_settings.sel(epoch=day, method="nearest")
    )

    # Test case 1: L1B dataset has good times
    with patch.object(HistogramL2, "get_calibration_factor", return_value=1):
        l2 = HistogramL2(l1b_hist_dataset, pipeline_settings, mock_calibration_dataset)

        expected_values = {
            "filter_temperature_average": [57.59],
            "filter_temperature_std_dev": [0.21],
            "hv_voltage_average": [1715.4],
            "hv_voltage_std_dev": [0.0],
        }

        assert np.isclose(
            l2.filter_temperature_average,
            expected_values["filter_temperature_average"],
            0.01,
        )
        assert np.isclose(
            l2.filter_temperature_std_dev,
            expected_values["filter_temperature_std_dev"],
            0.01,
        )
        assert np.isclose(
            l2.hv_voltage_average, expected_values["hv_voltage_average"], 0.01
        )
        assert np.isclose(
            l2.hv_voltage_std_dev, expected_values["hv_voltage_std_dev"], 0.01
        )

        cdf_attrs = ImapCdfAttributes()
        cdf_attrs.add_instrument_global_attrs("glows")
        cdf_attrs.add_instrument_variable_attrs("glows", "l2")
        output = create_l2_dataset(l2, cdf_attrs, l1b_hist_dataset.attrs)
        assert output["epoch"].data[0] == (l2.start_time + l2.end_time) / 2
        for var in ["photon_flux", "ecliptic_lon"]:
            assert np.issubdtype(output[var].dtype, np.floating)

        # Test case 2: L1B dataset has no good times (all flags 0)
        l1b_hist_dataset["flags"].values = np.zeros(l1b_hist_dataset.flags.shape)
        ds = HistogramL2(l1b_hist_dataset, pipeline_settings, mock_calibration_dataset)
        expected_number_of_good_l1b_inputs = 0
        assert ds.number_of_good_l1b_inputs == expected_number_of_good_l1b_inputs
        assert ds.bad_time_flag_occurrences.dtype == np.uint16


@patch.object(HistogramL2, "compute_position_angle", return_value=42.0)
@patch.object(
    HistogramL1B,
    "flag_uv_and_excluded",
    return_value=(np.zeros(3600, dtype=bool), np.zeros(3600, dtype=bool)),
)
@patch.object(HistogramL1B, "update_spice_parameters", autospec=True)
def test_glows_l2_cdf_fillvals(
    mock_spice_function,
    mock_flag_uv_and_excluded,
    mock_compute_position_angle,
    l1a_dataset,
    mock_ancillary_exclusions,
    mock_pipeline_settings,
    mock_conversion_table_dict,
    mock_ecliptic_bin_centers,
    mock_calibration_dataset,
):
    mock_spice_function.side_effect = mock_update_spice_parameters

    l1b_hist_dataset = glows_l1b(
        l1a_dataset[0],
        mock_ancillary_exclusions.excluded_regions,
        mock_ancillary_exclusions.uv_sources,
        mock_ancillary_exclusions.suspected_transients,
        mock_ancillary_exclusions.exclusions_by_instr_team,
        mock_pipeline_settings,
        mock_conversion_table_dict,
    )
    l1b_hist_dataset.attrs["Repointing"] = "repoint00047"

    l2_dataset = glows_l2(
        l1b_hist_dataset, mock_pipeline_settings, mock_calibration_dataset
    )[0]
    l2_dataset.attrs["Data_version"] = "001"
    cdf_file_path = write_cdf(l2_dataset)
    cdf_file = cdflib.CDF(cdf_file_path)

    histogram_flag_info = cdf_file.varinq("histogram_flag_array")
    histogram_flag_attrs = cdf_file.varattsget("histogram_flag_array")
    number_of_bins_info = cdf_file.varinq("number_of_bins")
    number_of_bins_attrs = cdf_file.varattsget("number_of_bins")
    photon_flux_info = cdf_file.varinq("photon_flux")
    photon_flux_attrs = cdf_file.varattsget("photon_flux")
    ecliptic_lon_info = cdf_file.varinq("ecliptic_lon")
    ecliptic_lon_attrs = cdf_file.varattsget("ecliptic_lon")

    assert histogram_flag_info.Data_Type_Description == "CDF_UINT1"
    assert histogram_flag_attrs["FILLVAL"] == np.uint8(255)
    assert number_of_bins_info.Data_Type_Description == "CDF_UINT2"
    assert number_of_bins_attrs["FILLVAL"] == np.uint16(65535)
    assert photon_flux_info.Data_Type_Description == "CDF_DOUBLE"
    assert np.isclose(photon_flux_attrs["FILLVAL"], np.float64(-1.0e31))
    assert ecliptic_lon_info.Data_Type_Description == "CDF_DOUBLE"
    assert np.isclose(ecliptic_lon_attrs["FILLVAL"], np.float64(-1.0e31))


def test_bin_exclusions(l1b_hists):
    # TODO test excluding bins as well

    raw_hists = DailyLightcurve.calculate_histogram_sums(l1b_hists["histogram"].data)
    expected_values = [2, 3, 4, 3, 4]

    assert np.array_equal(raw_hists, expected_values)
