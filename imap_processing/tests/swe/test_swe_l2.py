from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l2.swe_l2 import (
    calculate_flux,
    calculate_phase_space_density,
    find_angle_bin_indices,
    get_particle_energy,
    put_data_into_angle_bins,
    swe_l2,
)
from imap_processing.swe.utils import swe_constants
from imap_processing.swe.utils.swe_utils import (
    read_lookup_table,
)


def test_get_particle_energy():
    """Test get_particle_energy function."""
    all_energy = get_particle_energy()
    expected_energy = (
        read_lookup_table()["esa_v"].values * swe_constants.ENERGY_CONVERSION_FACTOR
    )
    np.testing.assert_array_equal(all_energy["energy"], expected_energy)


@patch(
    "imap_processing.swe.utils.swe_constants.GEOMETRIC_FACTORS",
    new=np.full(swe_constants.N_CEMS, 1),
)
@patch(
    "imap_processing.swe.l2.swe_l2.get_particle_energy",
    return_value=pd.DataFrame(
        {
            "table_index": np.repeat([0, 1], 720),
            "e_step": np.tile(np.arange(720), 2),
            "esa_v": np.repeat([1, 2], 720),
            "energy": np.repeat([1, 2], 720),
        }
    ),
)
def test_calculate_phase_space_density(patch_get_particle_energy):
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
            "acq_duration": (
                ["epoch", "cycle"],
                np.full((total_sweeps, swe_constants.N_QUARTER_CYCLES), 80.0),
            ),
            "esa_table_num": (
                ["epoch", "cycle"],
                np.repeat([0, 1], swe_constants.N_QUARTER_CYCLES).reshape(
                    total_sweeps, swe_constants.N_QUARTER_CYCLES
                ),
            ),
        }
    )
    phase_space_density_ds = calculate_phase_space_density(l1b_dataset)
    assert phase_space_density_ds["phase_space_density"].shape == (
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
    np.testing.assert_array_equal(
        phase_space_density_ds["phase_space_density"][0].data, expected_density
    )

    # Test that second sweep has correct values, similar to first sweep,
    # but with energy 2.
    expected_calculated_density = (2 * 1) / (
        1 * swe_constants.VELOCITY_CONVERSION_FACTOR * 2**2
    )
    expected_density = np.full(
        (
            swe_constants.N_ESA_STEPS,
            swe_constants.N_ANGLE_SECTORS,
            swe_constants.N_CEMS,
        ),
        expected_calculated_density,
    )
    np.testing.assert_array_equal(
        phase_space_density_ds["phase_space_density"][1].data, expected_density
    )
    assert type(phase_space_density_ds) == xr.Dataset


def test_calculate_flux():
    """Test calculate_flux function."""
    # Create a dummy l1b dataset
    total_sweeps = 2
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
            "acq_duration": (
                ["epoch", "cycle"],
                np.full((total_sweeps, swe_constants.N_QUARTER_CYCLES), 80.0),
            ),
            "esa_table_num": (
                ["epoch", "cycle"],
                np.repeat([0, 1], swe_constants.N_QUARTER_CYCLES).reshape(
                    total_sweeps, swe_constants.N_QUARTER_CYCLES
                ),
            ),
        }
    )

    flux = calculate_flux(l1b_dataset)
    assert flux.shape == (
        total_sweeps,
        swe_constants.N_ESA_STEPS,
        swe_constants.N_ANGLE_SECTORS,
        swe_constants.N_CEMS,
    )
    assert type(flux) == np.ndarray


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
    "imap_processing.swe.l1b.swe_l1b_science.read_in_flight_cal_data",
    return_value=pd.DataFrame(
        {
            "met_time": [453050300, 453077900],
            "cem1": [1, 1],
            "cem2": [1, 1],
            "cem3": [1, 1],
            "cem4": [1, 1],
            "cem5": [1, 1],
            "cem6": [1, 1],
            "cem7": [1, 1],
        }
    ),
)
@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_swe_l2(mock_read_in_flight_cal_data, use_fake_spin_data_for_time):
    """Test L2 processing."""
    data_start_time = 453051293.099714
    data_end_time = 453070000.0
    use_fake_spin_data_for_time(data_start_time, data_end_time)

    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")

    l1b_dataset = swe_l1b(l1a_datasets[0], "002")
    l2_dataset = swe_l2(l1b_dataset[0], "002")

    assert type(l2_dataset) == xr.Dataset
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

    # Write L2 to CDF
    l2_cdf_filepath = write_cdf(l2_dataset)
    assert l2_cdf_filepath.name == "imap_swe_l2_sci_20240510_v002.cdf"
