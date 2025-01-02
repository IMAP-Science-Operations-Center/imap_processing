from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.swe.l1a.swe_l1a import swe_l1a
from imap_processing.swe.l1b.swe_l1b import swe_l1b
from imap_processing.swe.l2.swe_l2 import (
    ENERGY_CONVERSION_FACTOR,
    VELOCITY_CONVERSION_FACTOR,
    calculate_flux,
    calculate_phase_space_density,
    get_particle_energy,
    swe_l2,
)
from imap_processing.swe.utils.swe_utils import read_lookup_table


def test_get_particle_energy():
    """Test get_particle_energy function."""
    all_energy = get_particle_energy()
    expected_energy = read_lookup_table()["esa_v"].values * ENERGY_CONVERSION_FACTOR
    np.testing.assert_array_equal(all_energy["energy"], expected_energy)


@patch("imap_processing.swe.l2.swe_l2.GEOMETRIC_FACTORS", new=np.full(7, 1))
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
                np.full((total_sweeps, 24, 30, 7), 1),
            ),
            "acq_duration": (["epoch", "cycle"], np.full((total_sweeps, 4), 80.0)),
            "esa_table_num": (
                ["epoch", "cycle"],
                np.repeat([0, 1], 4).reshape(total_sweeps, 4),
            ),
        }
    )
    phase_space_density_ds = calculate_phase_space_density(l1b_dataset)
    assert phase_space_density_ds["phase_space_density"].shape == (
        total_sweeps,
        24,
        30,
        7,
    )

    # Test that first sweep has correct values. In patch,
    #   1. we have set GEOMETRIC_FACTORS to 1.
    #   2. we have set energy to 1.
    #   3. we have set science_data to 1.
    # Using this in the formula, we calculate expected density value.
    expected_calculated_density = (2 * 1) / (1 * VELOCITY_CONVERSION_FACTOR * 1**2)
    expected_density = np.full((24, 30, 7), expected_calculated_density)
    np.testing.assert_array_equal(
        phase_space_density_ds["phase_space_density"][0].data, expected_density
    )

    # Test that second sweep has correct values, similar to first sweep,
    # but with energy 2.
    expected_calculated_density = (2 * 1) / (1 * VELOCITY_CONVERSION_FACTOR * 2**2)
    expected_density = np.full((24, 30, 7), expected_calculated_density)
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
                np.full((total_sweeps, 24, 30, 7), 1),
            ),
            "acq_duration": (["epoch", "cycle"], np.full((total_sweeps, 4), 80.0)),
            "esa_table_num": (
                ["epoch", "cycle"],
                np.repeat([0, 1], 4).reshape(total_sweeps, 4),
            ),
        }
    )

    flux = calculate_flux(l1b_dataset)
    assert flux.shape == (total_sweeps, 24, 30, 7)
    assert type(flux) == np.ndarray


@pytest.mark.usefixtures("use_fake_spin_data_for_time")
def test_swe_l2(use_fake_spin_data_for_time):
    """Test L2 processing."""
    data_start_time = 453051293.099714
    data_end_time = 453066734
    use_fake_spin_data_for_time(data_start_time, data_end_time)

    test_data_path = "tests/swe/l0_data/2024051010_SWE_SCIENCE_packet.bin"
    l1a_datasets = swe_l1a(imap_module_directory / test_data_path, "002")

    l1b_dataset = swe_l1b(l1a_datasets, "002")
    l2_dataset = swe_l2(l1b_dataset, "002")

    assert type(l2_dataset) == xr.Dataset
    assert l2_dataset["spin_phase"].shape == (6, 24, 30, 7)
