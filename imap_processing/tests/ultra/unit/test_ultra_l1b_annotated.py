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


def test_get_particle_velocity(kernels):
    """Tests get_particle_velocity function."""
    spice.furnsh(kernels)

    time = np.array([0.0])
    ultra_velocity = np.array([0.0])

    velocity = get_particle_velocity(time, ultra_velocity)

    print("hi")
