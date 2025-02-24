from pathlib import Path

import numpy as np
import pytest
import spiceypy

from imap_processing import imap_module_directory
from tools.spice.spice_examples import (
    _get_particle_velocity,
    build_annotated_events,
    get_attitude_timerange,
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
    # TODO: ALl kernels able to be downloaded from NAIF are not available
    #  in the test_data/spice directory.
    kernels = list_files_with_extensions(
        kernel_directory, [".tsc", ".tls", ".tf", ".bsp", ".ck"]
    )
    # Some kernels were moved into imap_processing package
    kernels.extend(
        list_files_with_extensions(
            imap_module_directory / "tests/spice/test_data", [".tsc", ".tls"]
        )
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


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_get_attitude_timerange(kernels, kernel_directory):
    """Tests get_attitude_timerange function."""

    with spiceypy.KernelPool(kernels):
        ck_kernel = list_files_with_extensions(kernel_directory, [".ck"])
        start, end = get_attitude_timerange(ck_kernel, -43000)

    assert start, end == (802094470.1857615, 802105267.1857585)


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_get_particle_velocity(direct_events, kernels, kernel_directory):
    """Tests the get_particle_velocity function."""

    with spiceypy.KernelPool(kernels):
        _get_particle_velocity(direct_events)


@pytest.mark.xfail(reason="Download NAIF kernels")
def test_build_annotated_events(direct_events, kernels, kernel_directory):
    """Tests the build_annotated_events function."""
    spiceypy.kclear()
    kernels = list_files_with_extensions(
        kernel_directory, [".tsc", ".tls", ".tf", ".bsp", ".ck"]
    )

    build_annotated_events(direct_events, kernels)
