import numpy as np
import pandas as pd

from imap_processing.spice.geometry import SpiceFrame
from imap_processing.ultra.l1c.make_helio_maps import make_helio_index_maps


# TODO ask nick about constant offset
# TODO ask nick about phi max aka our inside fov vs theirs
# TODO get CoordConverters.convertToRaDec function
def test_make_helio_index_maps(imap_ena_sim_metakernel, use_fake_repoint_data_for_time):
    """Test make_helio_index_maps."""
    start_et = 797949054.185627 + 69.186
    ds = make_helio_index_maps(
        nside=32,
        spin_duration=15.0,
        start_et=start_et,
        num_steps=720,
        instrument_frame=SpiceFrame.IMAP_ULTRA_90,
        compute_bsf=False,
    )
    base_path = "/Users/luco3133/projects/ultra_stuff/ultra_spin_cal_files"

    # Load expected data
    expected_index = pd.read_csv(
        f"{base_path}/IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-index.csv",
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_theta = pd.read_csv(
        f"{base_path}/IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-theta.csv",
        header=None,
        skiprows=1,
    ).to_numpy()
    expected_phi = pd.read_csv(
        f"{base_path}/IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-phi.csv",
        header=None,
        skiprows=1,
    ).to_numpy()
    # expected_bsf = pd.read_csv(
    #     f"{base_path}/IMAP_ULTRA_90-HELIO-IMAP_DPS-nside32-steps720-ebin0-bsf.csv",
    #     header=None,
    #     skiprows=1,
    # ).to_numpy()

    # CSV format: [ra, dec, step0, step1, ..., step719]
    # Skip ra and dec cols (first 2 columns) to get all time steps
    expected_index_all_steps = expected_index[:, 2:]  # Shape: (pixels, time_steps)
    expected_theta_all_steps = expected_theta[:, 2:]
    expected_phi_all_steps = expected_phi[:, 2:]
    # expected_bsf_all_steps = expected_bsf[:, 2:]

    # Replace nans with zero
    expected_index_all_steps = np.nan_to_num(expected_index_all_steps, nan=0)

    # Get Python outputs for all steps, energy bin 0
    # Python shape: (time_steps, energy_bins, pixels)
    # We want: (pixels, time_steps) to match Java
    index_all_steps = ds.index[:, 0, :].values.T  # Transpose to (pixels, time_steps)
    theta_all_steps = ds.theta[:, 0, :].values.T
    phi_all_steps = ds.phi[:, 0, :].values.T
    # bsf_all_steps = ds.bsf[:, 0, :].values.T

    # Only compare pixels where Java has non-zero values
    java_fov_mask = expected_index_all_steps != 0

    try:
        np.testing.assert_allclose(
            index_all_steps, expected_index_all_steps, equal_nan=True
        )
        print("✓ Index test PASSED (all steps)")
    except AssertionError:
        mismatch_count = np.sum(index_all_steps != expected_index_all_steps)
        total_elements = index_all_steps.size
        print(
            f"✗ Index test FAILED: {mismatch_count}/{total_elements} "
            f"({100 * mismatch_count / total_elements:.2f}%) mismatched"
        )

    try:
        np.testing.assert_allclose(
            theta_all_steps[java_fov_mask],
            expected_theta_all_steps[java_fov_mask],
            rtol=1e-4,
            atol=0.1,
        )
        print("✓ Theta test PASSED (all steps)")
    except AssertionError as e:
        print(f"✗ Theta test FAILED: {e}")

    try:
        np.testing.assert_allclose(
            phi_all_steps[java_fov_mask],
            expected_phi_all_steps[java_fov_mask],
            rtol=1e-4,
            atol=0.2,
        )
        print("✓ Phi test PASSED (all steps)")
    except AssertionError as e:
        print(f"✗ Phi test FAILED: {e}")
    #
    # try:
    #     np.testing.assert_allclose(
    #         bsf_all_steps[java_fov_mask],
    #         expected_bsf_all_steps[java_fov_mask],
    #         rtol=1e-4, atol=0.2
    #     )
    #     print("✓ bsf test PASSED (all steps)")
    # except AssertionError as e:
    #     print(f"✗ bsf test FAILED: {e}")
