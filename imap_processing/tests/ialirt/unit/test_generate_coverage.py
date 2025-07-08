"""Test processEphemeris functions."""

import pytest

from imap_processing.ialirt.generate_coverage import generate_coverage


@pytest.mark.external_kernel
def test_generate_coverage(furnish_kernels):
    """
    Test the generate_coverage function.
    """
    kernels = [
        "naif0012.tls",
        "pck00011.tpc",
        "de440s.bsp",
        "imap_spk_demo.bsp",
        "earth_1962_240827_2124_combined.bpc",
    ]
    with furnish_kernels(kernels):
        coverage_dict = generate_coverage("2026 SEP 22 00:00:00")
    # TODO: add assertions and make a plot
    assert coverage_dict is not None
