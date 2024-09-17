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

    kernels = ['/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/imap_static_kernels_euler.tm',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/naif0012.tls',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/imap_frames_demo_euler.tf',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/imap_ultra_instrument_demo.ti',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/de440.bsp',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/earth_000101_230322_221227.bpc',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/imap_spk_demo.bsp',
               '/Users/lasa6858/Desktop/ultra/ultra_prototype_v1/kernels/imap_sclk_0000.tsc']

    return kernels


def test_get_particle_velocity(kernels):
    """Tests get_particle_velocity function."""
    spice.furnsh(kernels)
    import pickle

    with open("/Users/lasa6858/Desktop/directEvents.pkl", "rb") as file:
        directEvents = pickle.load(file)

    time = directEvents['tdb']
    ultra_velocity = directEvents["vultra"]

    velocity = get_particle_velocity(time, ultra_velocity)

    print("hi")
