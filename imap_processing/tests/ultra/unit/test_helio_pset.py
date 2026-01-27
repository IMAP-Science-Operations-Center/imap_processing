from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.tests.conftest import _download_external_data
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1c.helio_pset import calculate_helio_pset

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.mark.skip(reason="Long running test for validation purposes.")
def test_validate_exposure_time_and_sensitivities(
    ancillary_files, rates_dataset, imap_ena_sim_metakernel, aux_dataset
):
    """Validates exposure time and sensitivities for ebin 0."""
    sens_filename = "SENS-IMAP_ULTRA_90-IMAP_DPS-HELIO-nside32-ebin0.csv"
    exposure_filename = "Exposures-IMAP_ULTRA_90-IMAP_DPS-HELIO-nside32-ebin0.csv"
    de_filename = "imap_ultra_l1b_90sensor-de_20000101-repoint00000_v000.cdf"
    test_data = [
        (sens_filename, "ultra/data/l1/"),
        (exposure_filename, "ultra/data/l1/"),
        (de_filename, "ultra/data/l1/"),
    ]
    _download_external_data(test_data)
    l1b_de = TEST_PATH / de_filename
    l1b_de = load_cdf(l1b_de)
    sensitivities_ebin_0 = pd.read_csv(TEST_PATH / sens_filename)
    exposure_factor_ebin_0 = pd.read_csv(TEST_PATH / exposure_filename)

    test_deadtimes = (
        pd.read_csv(TEST_PATH / "test_p0_ebin0_deadtimes.csv", header=None)
        .to_numpy()
        .squeeze()
    )
    npix = 12288  # nside 32
    # Create a minimal dataset to pass to the function
    dataset = xr.Dataset(
        {
            "spin_number": (["epoch"], np.array([1, 2, 3])),
        }
    )
    dataset.attrs["Repointing"] = "repoint00000"

    pointing_range_met = (472374890.0, 582378000.0)
    # Create mock spin data that has 5525 nominal spins
    # Create DataFrame
    nspins = 5522
    nominal_spin_seconds = 15.0
    spin_data = pd.DataFrame(
        {
            "spin_start_met": np.linspace(
                pointing_range_met[0], pointing_range_met[1], nspins
            ),
            "spin_period_sec": np.full(nspins, nominal_spin_seconds),
            "spin_phase_valid": np.ones(nspins),
            "spin_period_valid": np.ones(nspins),
        }
    )
    with (
        # Mock the pointing times
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.get_pointing_times_from_id",
            return_value=pointing_range_met,
        ),
        # Mock deadtimes to be all ones
        mock.patch(
            "imap_processing.ultra.l1c.ultra_l1c_pset_bins."
            "get_deadtime_ratios_by_spin_phase",
            return_value=xr.DataArray(test_deadtimes, dims="spin_phase_step"),
        ),
        # Mock spin data to match nominal spins in a pointing period
        mock.patch(
            "imap_processing.ultra.l1c.ultra_l1c_pset_bins.get_spin_data",
            return_value=spin_data,
        ),
        # Mock background rates to be constant 0.1
        mock.patch(
            "imap_processing.ultra.l1c.helio_pset.get_spacecraft_background_rates",
            return_value=np.ones((46, npix)),
        ),
        # Mock culling mask (no culling)
        mock.patch("imap_processing.ultra.l1c.helio_pset.compute_culling_mask"),
    ):
        pset = calculate_helio_pset(
            l1b_de,
            dataset,
            rates_dataset,
            aux_dataset,
            "imap_ultra_l1c_90sensor-heliopset",
            ancillary_files,
            90,
            UltraConstants.TOFXPH_SPECIES_GROUPS["proton"],
        )

    # Validate exposure times for ebin 0
    exposure_times = pset["exposure_factor"][0, 0, :].values
    expected_exposure_times = exposure_factor_ebin_0["P0"].to_numpy()
    np.testing.assert_allclose(
        exposure_times,
        expected_exposure_times,
        atol=95,  # TODO This is due to the helio index map differences
        err_msg="Exposure times do not match expected values for ebin 0.",
    )
    # Validate sensitivities for ebin 0
    sensitivity = pset["sensitivity"][0, :].values
    expected_sensitivity = sensitivities_ebin_0["Sensitivity (cm2)"].to_numpy()
    np.testing.assert_allclose(
        sensitivity,
        expected_sensitivity,
        atol=0.0006,  # TODO This is due to the helio index map differences
        err_msg="Sensitivities times do not match expected values for ebin 0.",
    )
