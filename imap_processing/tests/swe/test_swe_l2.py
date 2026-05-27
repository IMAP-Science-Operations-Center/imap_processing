from unittest.mock import patch

import cdflib
import numpy as np
import pytest
import xarray as xr
from imap_data_access.processing_input import (
    AncillaryInput,
    ProcessingInputCollection,
    ScienceInput,
)

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.quality_flags import SweL1bFlags
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l2.swe_l2 import (
    calculate_flux,
    calculate_phase_space_density,
    find_angle_bin_indices,
    put_data_into_angle_bins,
    swe_l2,
)
from imap_processing.swe.utils import swe_constants

pytestmark = pytest.mark.external_test_data

OLD_GEOMETRIC_FACTORS = np.array(
    [
        435e-6,
        599e-6,
        808e-6,
        781e-6,
        876e-6,
        548e-6,
        432e-6,
    ]
)


@patch(
    "imap_processing.swe.utils.swe_constants.GEOMETRIC_FACTORS",
    new=np.full(swe_constants.N_CEMS, 1),
)
def test_calculate_phase_space_density():
    """Test calculate_phase_space_density function."""
    # Create a dummy l1b dataset
    total_sweeps = 2
    np.random.seed(0)
    l1b_dataset = xr.Dataset(
        {
            "science_data": (
                ["epoch", "energy", "angle", "cem"],
                np.full(
                    (
                        total_sweeps,
                        swe_constants.N_ESA_STEPS,
                        swe_constants.N_ANGLE_SECTORS,
                        swe_constants.N_CEMS,
                    ),
                    1,
                ),
            ),
            "esa_energy": (
                ["epoch", "energy", "angle"],
                np.full(
                    (
                        total_sweeps,
                        swe_constants.N_ESA_STEPS,
                        swe_constants.N_ANGLE_SECTORS,
                    ),
                    1,
                ),
            ),
        }
    )
    phase_space_density = calculate_phase_space_density(
        l1b_dataset["science_data"].data, l1b_dataset["esa_energy"].data
    )
    assert phase_space_density.shape == (
        total_sweeps,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
        swe_constants.N_CEMS,
    )

    # Test that first sweep has correct values. In patch,
    #   1. we have set GEOMETRIC_FACTORS to 1.
    #   2. we have set energy to 1.
    #   3. we have set science_data to 1.
    # Using this in the formula, we calculate expected density value.
    expected_calculated_density = (2 * 1) / (
        1 * swe_constants.VELOCITY_CONVERSION_FACTOR * 1**2
    )
    expected_density = np.full(
        (
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_SECTORS,
            swe_constants.N_CEMS,
        ),
        expected_calculated_density,
    )
    np.testing.assert_array_equal(phase_space_density[0], expected_density)


def test_calculate_flux():
    """Test calculate_flux function."""
    # Create a dummy l1b dataset
    total_sweeps = 2
    phase_space_density = np.full(
        (
            total_sweeps,
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_SECTORS,
            swe_constants.N_CEMS,
        ),
        1,
    )

    esa_energy = np.full(
        (
            total_sweeps,
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_SECTORS,
        ),
        1,
    )

    flux = calculate_flux(phase_space_density, esa_energy)
    assert flux.shape == (
        total_sweeps,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
        swe_constants.N_CEMS,
    )
    assert isinstance(flux, np.ndarray)


def test_find_angle_bin_indices():
    """Test find_angle_bin_indices function."""
    spin_angle_bin_center = np.arange(6, 360, 12)

    # List with various start, middle, end angles or
    # far left or far right edge angles
    start_angles = [0, 0.11, 3, 6, 11.99, 12, 355]
    expected_angle_bin_indices = np.array([0, 0, 0, 0, 0, 1, 29])
    angle_bin_indices = find_angle_bin_indices(start_angles)
    np.testing.assert_array_equal(angle_bin_indices, expected_angle_bin_indices)
    # Test that angle bin indices lands in correct center bin
    np.testing.assert_array_equal(
        spin_angle_bin_center[angle_bin_indices],
        np.array([6, 6, 6, 6, 6, 18, 354]),
    )

    # Test for angles that are outside the range
    with pytest.raises(
        ValueError, match=r"Input angle values must be in the range \[0, 360\)"
    ):
        find_angle_bin_indices(np.array([-1]))

    with pytest.raises(
        ValueError, match=r"Input angle values must be in the range \[0, 360\)"
    ):
        find_angle_bin_indices(np.array([360]))


def test_put_data_into_angle_bins():
    """Test put_data_into_angle_bins function."""
    num_cycles = 1
    # Create test counts data to test
    # Find all even numbers in the range 0 to 30
    even_numbers = np.arange(0, 30, 2)
    # repeat it twice now to get:
    # [0, 0, 2, 2, ...., 28, 28]
    example_data = np.repeat(even_numbers, 2)
    energy_angle_test_data = np.tile(
        example_data, (num_cycles, swe_constants.N_ESA_STEPS, 1)
    )
    # Expand to include 7 CEMs by repeating across last dimension
    test_data = np.repeat(
        energy_angle_test_data[..., np.newaxis], swe_constants.N_CEMS, axis=-1
    )

    # Took this example from intermediate output from actual data
    angle_bins_example = [
        12,
        14,
        14,
        16,
        16,
        18,
        18,
        20,
        20,
        22,
        22,
        24,
        24,
        26,
        26,
        28,
        28,
        0,
        0,
        2,
        2,
        4,
        4,
        6,
        6,
        8,
        8,
        10,
        10,
        12,
    ]
    # Now data with every row to be same as angle_bins_example
    test_angle_bin_indices_data = np.full(
        (num_cycles, swe_constants.N_ESA_STEPS, swe_constants.N_ANGLE_SECTORS),
        angle_bins_example,
    )

    binned_data = put_data_into_angle_bins(test_data, test_angle_bin_indices_data)
    assert binned_data.shape == (
        num_cycles,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_BINS,
        swe_constants.N_CEMS,
    )

    # Test that the binned data has correct values in correct bins by
    # checking that odd number columns are filled with nan
    expected_binned_data = np.full(
        (
            num_cycles,
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_BINS,
            swe_constants.N_CEMS,
        ),
        np.nan,
    )
    np.testing.assert_array_equal(
        binned_data[0, 0, 1::2, 0], expected_binned_data[0, 0, 1::2, 0]
    )

    # Now check that mean calculation is correct
    even_col_mean_data = binned_data[0, 0, 0::2, 0]
    # Expected mean of even columns is below
    expected_mean_data = np.array(
        [
            17.0,
            19.0,
            21.0,
            23.0,
            25.0,
            27.0,
            14.0,
            1.0,
            3.0,
            5.0,
            7.0,
            9.0,
            11.0,
            13.0,
            15.0,
        ]
    )
    np.testing.assert_array_equal(even_col_mean_data, expected_mean_data)


@patch(
    "imap_processing.swe.utils.swe_constants.GEOMETRIC_FACTORS",
    new=OLD_GEOMETRIC_FACTORS,
)
@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
@patch(
    "imap_processing.spice.spin.get_spacecraft_to_instrument_spin_phase_offset",
    return_value=153 / 360,
)
@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_swe_l2_15sec(
    mock_phase_offset,
    mock_get_file_paths,
    use_fake_spin_data_for_time,
    l2_sector_validation_df,
    l2_binned_flux_validation_df,
    l2_binned_psd_validation_df,
):
    """Test L2 processing with 15 seconds spin period."""
    data_start_time = 453051293.0
    data_end_time = 453070000.0
    use_fake_spin_data_for_time(data_start_time, data_end_time)

    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path)
    l1a_ds = l1a_datasets[0]
    l1a_ds.attrs["Data_version"] = "000"
    l1a_cdf_filepath = write_cdf(l1a_ds)
    assert l1a_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v000.cdf"

    def get_file_paths_side_effect(descriptor):
        if descriptor == "sci":
            return [l1a_cdf_filepath]
        elif descriptor == "l1b-in-flight-cal":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
            ]
        elif descriptor == "eu-conversion":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_eu-conversion_20240510_v000.csv"
            ]
        elif descriptor == "esa-lut":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_esa-lut_20250301_v000.csv"
            ]
        elif descriptor == "raw":
            return []
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = get_file_paths_side_effect
    science_input = ScienceInput(l1a_cdf_filepath.name)
    inflight_anc = AncillaryInput(
        "imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
    )
    eu_anc = AncillaryInput("imap_swe_eu-conversion_20240510_v000.csv")
    dependencies = ProcessingInputCollection(science_input, inflight_anc, eu_anc)
    l1b_dataset = swe_l1b(dependencies)[0]
    l1b_dataset.attrs["Data_version"] = "000"

    # Test data acquisition times (~453051355) are before cal_times[-2] (553051294),
    # so no epoch should have LAST_CAL_INTERVAL set.
    assert not np.any(
        l1b_dataset["data_quality"].values & SweL1bFlags.LAST_CAL_INTERVAL.value
    )

    l2_dataset = swe_l2(l1b_dataset)

    # Verify data_quality is propagated from L1B to L2 for downstream (L3) use.
    assert "data_quality" in l2_dataset
    np.testing.assert_array_equal(
        l2_dataset["data_quality"].values,
        l1b_dataset["data_quality"].values,
    )

    assert isinstance(l2_dataset, xr.Dataset)
    assert l2_dataset["phase_space_density_spin_sector"].shape == (
        6,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
        swe_constants.N_CEMS,
    )
    assert l2_dataset["flux_spin_sector"].shape == (
        6,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
        swe_constants.N_CEMS,
    )
    assert l2_dataset["acquisition_time"].shape == (
        6,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
    )

    rate = l1b_dataset.science_data.to_numpy()
    psd = l2_dataset.phase_space_density_spin_sector.to_numpy()
    second_cycle_rate = rate[2, :, :, 3]  # nonzero counts at all energy & spin
    psd = psd[2, :, :, 3]
    cal_factor = (
        psd / second_cycle_rate
    )  # same CEM, should be constant at a given energy
    assert np.allclose(cal_factor, cal_factor[:, 0:1], rtol=1e-9, atol=0)

    # Write L2 to CDF
    l2_dataset.attrs["Data_version"] = "002"
    l2_cdf_filepath = write_cdf(l2_dataset)
    assert l2_cdf_filepath.name == "imap_swe_l2_sci_20240510_v002.cdf"
    with cdflib.CDF(l2_cdf_filepath) as cdf_file:
        acq_duration_info = cdf_file.varinq("acq_duration")
        acq_duration_attrs = cdf_file.varattsget("acq_duration")
        psd_attrs = cdf_file.varattsget("phase_space_density_spin_sector")
        psd_binned_attrs = cdf_file.varattsget("phase_space_density")
        flux_attrs = cdf_file.varattsget("flux_spin_sector")
        flux_binned_attrs = cdf_file.varattsget("flux")
        psd_uncert_attrs = cdf_file.varattsget("psd_stat_uncert")
        flux_uncert_attrs = cdf_file.varattsget("flux_stat_uncert")
        inst_az_attrs = cdf_file.varattsget("inst_az")
        inst_el_attrs = cdf_file.varattsget("inst_el")
        inst_az_spin_sector_attrs = cdf_file.varattsget("inst_az_spin_sector")
        global_attrs = cdf_file.globalattsget()

        assert acq_duration_info.Data_Type_Description == "CDF_UINT4"
        assert acq_duration_attrs["FILLVAL"] == np.uint32(4294967295)
        assert psd_uncert_attrs["UNITS"] == psd_attrs["UNITS"]
        assert flux_uncert_attrs["UNITS"] == flux_attrs["UNITS"]
        assert psd_uncert_attrs["VAR_TYPE"] == "support_data"
        assert flux_uncert_attrs["VAR_TYPE"] == "support_data"
        assert psd_attrs["DELTA_PLUS_VAR"] == "psd_stat_uncert"
        assert psd_attrs["DELTA_MINUS_VAR"] == "psd_stat_uncert"
        assert flux_attrs["DELTA_PLUS_VAR"] == "flux_stat_uncert"
        assert flux_attrs["DELTA_MINUS_VAR"] == "flux_stat_uncert"
        assert psd_attrs["FIELDNAM"] == "Phase Space Density Spin Sector"
        assert psd_binned_attrs["FIELDNAM"] == "Phase Space Density"
        assert flux_attrs["FIELDNAM"] == "Flux Spin Sector"
        assert flux_binned_attrs["FIELDNAM"] == "Flux"
        assert flux_attrs["UNITS"] == "1 / (eV * cm^2 * s * ster)"
        assert flux_binned_attrs["UNITS"] == "1 / (eV * cm^2 * s * ster)"
        assert (
            inst_az_attrs["CATDESC"]
            == "Spin angle in despun spacecraft coordinates. Angle resolution "
            "is 12 degree, with bin centers 6 - 354 degree (30 bins)"
        )
        assert (
            inst_el_attrs["CATDESC"]
            == "Polar angle of each CEM detector relative to spin axis. Angle "
            "resolution is 21 degree, with bin centers -63 - 63 degree (7 bins)"
        )
        assert inst_az_spin_sector_attrs["FIELDNAM"] == "Spin Angle Spin Sector"
        assert (
            inst_az_spin_sector_attrs["CATDESC"]
            == "Spin angle in despun spacecraft coordinates organized by ESA step "
            "and spin sector"
        )
        assert "Los Alamos National Laboratory" in global_attrs["TEXT"][0]
        assert (
            "https://imap.princeton.edu/spacecraft/instruments/"
            "solar-wind-electron-swe" in global_attrs["TEXT"][0]
        )

    # --------- sector validation--------
    sector_psd_data = l2_dataset["phase_space_density_spin_sector"].data
    validation_science = l2_sector_validation_df.values[:, 1:].reshape(6, 24, 30, 7)

    np.testing.assert_allclose(sector_psd_data, validation_science, rtol=1e-6)

    # ------binned validation--------
    bin_flux_val = l2_binned_flux_validation_df.values[:, 1:].reshape(6, 24, 30, 7)
    bin_psd_val = l2_binned_psd_validation_df.values[:, 1:].reshape(6, 24, 30, 7)
    bin_flux_data = l2_dataset["flux"].data
    bin_psd_data = l2_dataset["phase_space_density"].data

    np.testing.assert_allclose(bin_flux_data, bin_flux_val, rtol=1e-6)
    np.testing.assert_allclose(bin_psd_data, bin_psd_val, rtol=1e-6)


@patch(
    "imap_processing.swe.utils.swe_constants.GEOMETRIC_FACTORS",
    new=OLD_GEOMETRIC_FACTORS,
)
@patch("imap_data_access.processing_input.ProcessingInputCollection.get_file_paths")
@patch(
    "imap_processing.spice.spin.get_spacecraft_to_instrument_spin_phase_offset",
    return_value=153 / 360,
)
@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_swe_l2_14_6sec(
    mock_phase_offset,
    mock_get_file_paths,
    use_fake_spin_data_for_time,
    l2_binned_flux_14sec_validation_df,
    l2_binned_psd_14sec_validation_df,
):
    """Test L2 processing with 14.6 seconds spin period."""
    data_start_time = 453051293.0
    data_end_time = 453070000.0
    use_fake_spin_data_for_time(data_start_time, data_end_time, spin_period=14.6)

    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path)
    l1a_ds = l1a_datasets[0]
    l1a_ds.attrs["Data_version"] = "000"
    l1a_cdf_filepath = write_cdf(l1a_ds)
    assert l1a_cdf_filepath.name == "imap_swe_l1a_sci_20240510_v000.cdf"

    def get_file_paths_side_effect(descriptor):
        if descriptor == "sci":
            return [l1a_cdf_filepath]
        elif descriptor == "l1b-in-flight-cal":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
            ]
        elif descriptor == "eu-conversion":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_eu-conversion_20240510_v000.csv"
            ]
        elif descriptor == "esa-lut":
            return [
                imap_module_directory
                / "tests/swe/lut/imap_swe_esa-lut_20250301_v000.csv"
            ]
        elif descriptor == "raw":
            return []
        else:
            raise ValueError(f"Unknown descriptor: {descriptor}")

    mock_get_file_paths.side_effect = get_file_paths_side_effect
    science_input = ScienceInput(l1a_cdf_filepath.name)
    inflight_anc = AncillaryInput(
        "imap_swe_l1b-in-flight-cal_20240510_20260716_v000.csv"
    )
    eu_anc = AncillaryInput("imap_swe_eu-conversion_20240510_v000.csv")
    dependencies = ProcessingInputCollection(science_input, inflight_anc, eu_anc)
    l1b_dataset = swe_l1b(dependencies)[0]
    l1b_dataset.attrs["Data_version"] = "000"
    l2_dataset = swe_l2(l1b_dataset)

    # Write L2 to CDF
    l2_dataset.attrs["Data_version"] = "002"
    l2_cdf_filepath = write_cdf(l2_dataset)
    assert l2_cdf_filepath.name == "imap_swe_l2_sci_20240510_v002.cdf"

    # --------14.6 sec spin period validation--------
    bin_flux_val = l2_binned_flux_14sec_validation_df.values[:, 1:].reshape(
        6, 24, 30, 7
    )
    bin_flux_data = l2_dataset["flux"].data

    bin_psd_val = l2_binned_psd_14sec_validation_df.values[:, 1:].reshape(6, 24, 30, 7)
    bin_psd_data = l2_dataset["phase_space_density"].data

    # Since L2 processed data stores nan in bins where there is no data
    # but validation data stores 0.0 in those same bins,
    # we need to find non-nan data to compare.
    nan_mask = ~np.isnan(bin_flux_data)
    non_nan_data = bin_flux_data[nan_mask]
    non_nan_val = bin_flux_val[nan_mask]
    np.testing.assert_allclose(non_nan_data, non_nan_val, rtol=1e-6)

    # Test that binned PSD data matches validation data
    nan_mask = ~np.isnan(bin_psd_data)
    non_nan_data = bin_psd_data[nan_mask]
    non_nan_val = bin_psd_val[nan_mask]
    np.testing.assert_allclose(non_nan_data, non_nan_val, rtol=1e-6)
