import numpy as np
import pandas as pd
import pytest

from imap_processing import imap_module_directory
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1c.make_helio_maps import make_helio_index_maps

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.mark.external_test_data
def test_make_helio_index_maps(imap_ena_sim_metakernel, use_fake_repoint_data_for_time):
    """Test make_helio_index_maps."""
    start_et = 797949123.371627
    ds = make_helio_index_maps(
        nside=32,
        spin_duration=15.0,
        start_et=start_et,
        num_steps=720,
        instrument_frame=SpiceFrame.IMAP_ULTRA_90,
        compute_bsf=False,
    )

    # Load expected data
    expected_index = pd.read_csv(
        TEST_PATH / "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-index.csv",
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_theta = pd.read_csv(
        TEST_PATH / "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-theta.csv",
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_phi = pd.read_csv(
        TEST_PATH / "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-phi.csv",
        header=None,
        skiprows=1,
    ).to_numpy()

    # Skip ra and dec cols
    expected_index_all_steps = expected_index[:, 2:]
    expected_theta_all_steps = expected_theta[:, 2:]
    expected_phi_all_steps = expected_phi[:, 2:]

    # Replace nans with zero
    expected_index_all_steps = np.nan_to_num(expected_index_all_steps, nan=0)

    # Get outputs
    index_all_steps = ds.index[:, 0, :].values.T
    theta_all_steps = ds.theta[:, 0, :].values.T
    phi_all_steps = ds.phi[:, 0, :].values.T

    # Test index mismatch percentage
    mismatch_count = np.sum(index_all_steps != expected_index_all_steps)
    mismatch_pct = 100 * mismatch_count / index_all_steps.size
    assert mismatch_pct < 0.05

    both_valid_mask = (expected_index_all_steps != 0) & (index_all_steps != 0)

    np.testing.assert_allclose(
        theta_all_steps[both_valid_mask],
        expected_theta_all_steps[both_valid_mask],
        rtol=1e-4,
        atol=0.1,
    )

    np.testing.assert_allclose(
        phi_all_steps[both_valid_mask],
        expected_phi_all_steps[both_valid_mask],
        rtol=1e-4,
        atol=0.2,
    )
