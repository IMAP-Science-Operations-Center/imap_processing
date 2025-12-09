import numpy as np
import pandas as pd
import pytest
import spiceypy as sp

from imap_processing import imap_module_directory
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.tests.conftest import _download_external_data
from imap_processing.ultra.l1c.make_helio_index_maps import make_helio_index_maps

TEST_PATH = imap_module_directory / "tests" / "ultra" / "data" / "l1"


@pytest.fixture
def helio_index_kernels(furnish_kernels, _download_kernels):
    kernels = [
        "imap_sclk_0000.tsc",
        "naif0012.tls",
        "imap_spk_demo.bsp",
        "sim_1yr_imap_attitude.bc",
        "imap_001.tf",
        "de440s.bsp",
        "imap_science_draft.tf",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    with furnish_kernels(kernels) as k:
        yield k


# @pytest.mark.skip(reason="Long running test for validation purposes.")
def test_make_helio_index_maps(
    helio_index_kernels, use_fake_repoint_data_for_time, spice_test_data_path
):
    """Test make_helio_index_maps."""
    # Get coverage window
    ck_kernel, _, _, _ = sp.kdata(1, "ck")
    ck_cover = sp.ckcov(
        ck_kernel, SpiceFrame.IMAP_DPS.value, True, "INTERVAL", 0, "TDB"
    )
    et_start, et_end = sp.wnfetd(ck_cover, 0)

    ds = make_helio_index_maps(
        nside=32,
        spin_duration=15.0,
        start_et=et_start,
        num_steps=720,
        instrument_frame=SpiceFrame.IMAP_ULTRA_90,
        compute_bsf=False,
    )
    index_file = "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-index.csv"
    theta_file = "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-theta.csv"
    phi_file = "IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-phi.csv"
    test_data = [
        (index_file, "ultra/data/l1/"),
        (theta_file, "ultra/data/l1/"),
        (phi_file, "ultra/data/l1/"),
    ]
    _download_external_data(test_data)
    # Load expected data
    expected_index = pd.read_csv(
        TEST_PATH / index_file,
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_theta = pd.read_csv(
        TEST_PATH / theta_file,
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_phi = pd.read_csv(
        TEST_PATH / phi_file,
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
    assert mismatch_pct < 0.02

    both_valid_mask = (expected_index_all_steps != 0) & (index_all_steps != 0)

    np.testing.assert_allclose(
        theta_all_steps[both_valid_mask],
        expected_theta_all_steps[both_valid_mask],
        rtol=1e-4,
    )

    np.testing.assert_allclose(
        phi_all_steps[both_valid_mask],
        expected_phi_all_steps[both_valid_mask],
        rtol=1e-4,
        atol=0.05,
    )


# @pytest.mark.skip(reason="Long running test for validation purposes.")
def test_make_helio_index_maps_45(helio_index_kernels, use_fake_repoint_data_for_time):
    """Test make_helio_index_maps."""
    ck_kernel, _, _, _ = sp.kdata(1, "ck")
    ck_cover = sp.ckcov(
        ck_kernel, SpiceFrame.IMAP_DPS.value, True, "INTERVAL", 0, "TDB"
    )
    et_start, et_end = sp.wnfetd(ck_cover, 0)
    ds = make_helio_index_maps(
        nside=32,
        spin_duration=15.0,
        start_et=et_start,
        num_steps=720,
        instrument_frame=SpiceFrame.IMAP_ULTRA_45,
        compute_bsf=False,
    )

    index_file = "IMAP_ULTRA_45-HELIO-IMAP_DPS-nside32-steps720-ebin0-index.csv"
    theta_file = "IMAP_ULTRA_45-HELIO-IMAP_DPS-nside32-steps720-ebin0-theta.csv"
    phi_file = "IMAP_ULTRA_45-HELIO-IMAP_DPS-nside32-steps720-ebin0-phi.csv"
    test_data = [
        (index_file, "ultra/data/l1/"),
        (theta_file, "ultra/data/l1/"),
        (phi_file, "ultra/data/l1/"),
    ]
    _download_external_data(test_data)
    # Load expected data
    expected_index = pd.read_csv(
        TEST_PATH / index_file,
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_theta = pd.read_csv(
        TEST_PATH / theta_file,
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_phi = pd.read_csv(
        TEST_PATH / phi_file,
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
    assert mismatch_pct < 0.02

    both_valid_mask = (expected_index_all_steps != 0) & (index_all_steps != 0)

    np.testing.assert_allclose(
        theta_all_steps[both_valid_mask],
        expected_theta_all_steps[both_valid_mask],
        rtol=1e-4,
        atol=0.05,
    )

    np.testing.assert_allclose(
        phi_all_steps[both_valid_mask],
        expected_phi_all_steps[both_valid_mask],
        rtol=1e-4,
        atol=0.05,
    )
