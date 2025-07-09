"""Test processEphemeris functions."""

from datetime import datetime

import pytest

from imap_processing.ialirt.generate_coverage import generate_coverage


@pytest.mark.external_kernel
def test_generate_coverage(furnish_kernels):
    """
    Test the generate_coverage function.
    """
    # Note: tested this code with the Sun and achieved expected
    # results ~12 hours of coverage from horizon to horizon.
    kernels = [
        "naif0012.tls",
        "pck00011.tpc",
        "de440s.bsp",
        "imap_spk_demo.bsp",
        "earth_1962_240827_2124_combined.bpc",
    ]
    with furnish_kernels(kernels):
        coverage_dict = generate_coverage("2026 SEP 22 00:00:00")

    start = datetime.strptime(coverage_dict["Kiel_time"][0], "%Y-%m-%dT%H:%M:%S.%f")
    end = datetime.strptime(coverage_dict["Kiel_time"][-1], "%Y-%m-%dT%H:%M:%S.%f")

    duration = end - start
    hours = duration.total_seconds() / 3600

    # Coverage duration should be approximately 9 hours.
    assert hours == pytest.approx(9, abs=1)
