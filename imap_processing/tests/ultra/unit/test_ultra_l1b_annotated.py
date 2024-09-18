"""Tests coverage for ultra_l1b_annotated.py"""

import numpy as np
import pytest
import spiceypy as spice

from imap_processing.ultra.l1b.ultra_l1b_annotated import get_particle_velocity


@pytest.fixture()
def kernels(spice_test_data_path):
    """List SPICE kernels."""
    required_kernels = [
        "imap_science_0001.tf",
        "imap_sclk_0000.tsc",
        "sim_1yr_imap_attitude.bc",
        "imap_wkcp.tf",
        "naif0012.tls",
        "sim_1yr_imap_pointing_frame.bc",
    ]
    kernels = [str(spice_test_data_path / kernel) for kernel in required_kernels]

    return kernels


def test_get_particle_velocity(spice_test_data_path, kernels):
    """Tests get_particle_velocity function."""
    spice.furnsh(kernels)

    id_imap_dps = spice.gipool("FRAME_IMAP_DPS", 0, 1)
    pointing_cover = spice.ckcov(
        str(spice_test_data_path / "sim_1yr_imap_pointing_frame.bc"),
        int(id_imap_dps),
        True,
        "SEGMENT",
        0,
        "TDB",
    )
    # Get start and end time of first interval
    start, _ = spice.wnfetd(pointing_cover, 0)

    times = np.array([start])

    instrument_velocity = np.array([[41.18609, -471.24467, -832.8784]])

    velocity_45 = get_particle_velocity(times, instrument_velocity, "IMAP_ULTRA_45")
    velocity_90 = get_particle_velocity(times, instrument_velocity, "IMAP_ULTRA_90")

    # Compute the magnitude of the velocity vectors in both frames
    magnitude_45 = np.linalg.norm(velocity_45[0])
    magnitude_90 = np.linalg.norm(velocity_90[0])

    assert np.allclose(magnitude_45, magnitude_90, atol=1e-6)
