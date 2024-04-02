from pathlib import Path

import numpy as np
import pytest
import spiceypy as spice

from tools.spice.spice_examples import (
    build_annotated_events,
    get_attitude_timerange,
    get_particle_velocity,
)
from tools.spice.spice_utils import (
    list_files_with_extensions,
)


@pytest.fixture()
def kernel_directory():
    """Kernel directory."""
    kernel_directory = Path(__file__).parents[1] / "test_data" / "spice"
    return kernel_directory


@pytest.fixture()
def kernels(kernel_directory):
    """Loads all kernels."""
    kernels = list_files_with_extensions(
        kernel_directory, [".tsc", ".tls", ".tf", ".bsp"]
    )
    return kernels


@pytest.fixture()
def direct_events():
    """Loads direct events test data.

    Possible example of future direct_events structure:
    direct_events.keys()
    dict_keys(['eventID', 'tdb', 'xstart', 'ystart', 'xstop', 'ystop',
    'eventtype', 'ph', 'spinphase', 'tof', 'ishead90', 'yf', 'd',
    'energy', 'rvecultra', 'vultra', 'rmagultra', 'rnormultra'])
    """
    direct_events = {
        "vultra": np.array(
            [-5.7718470e08, -8.8437594e08, -9.5545958e08], dtype=np.float32
        ),
        "tdb": 802100000.0,
    }

    return direct_events


def test_get_attitude_timerange(kernels, kernel_directory):
    """Tests get_attitude_timerange function."""

    spice.kclear()

    spice.furnsh(kernels)

    ck_kernel = list_files_with_extensions(kernel_directory, [".ck"])

    expected_error_message = "ID -43 not found in CK file"
    with pytest.raises(ValueError, match=expected_error_message):
        get_attitude_timerange(ck_kernel, -43)

    # Load the C kernel
    spice.furnsh(ck_kernel)

    start, end = get_attitude_timerange(ck_kernel, -43000)

    # Clear all kernels
    spice.kclear()

    assert start, end == (802094470.1857615, 802105267.1857585)


@pytest.mark.xfail(reason="Add de430.bsp to test/test_data/spice")
def test_get_particle_velocity(direct_events, kernels, kernel_directory):
    """Tests the get_particle_velocity function."""

    spice.kclear()

    spice.furnsh(kernels)
    ck_kernel = list_files_with_extensions(kernel_directory, [".ck"])
    spice.furnsh(ck_kernel)

    get_particle_velocity(direct_events)

    spice.kclear()


@pytest.mark.xfail(reason="Add de430.bsp to test/test_data/spice")
def test_build_annotated_events(direct_events, kernels, kernel_directory):
    """Tests the build_annotated_events function."""
    spice.kclear()
    kernels = list_files_with_extensions(
        kernel_directory, [".tsc", ".tls", ".tf", ".bsp", ".ck"]
    )

    build_annotated_events(direct_events, kernels)
