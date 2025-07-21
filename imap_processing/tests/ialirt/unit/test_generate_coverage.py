"""Test processEphemeris functions."""

from datetime import datetime

import numpy as np
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
        coverage_dict = generate_coverage("2026-09-22T00:00:00Z")

    start = datetime.strptime(coverage_dict["Kiel_time"][0], "%Y-%m-%dT%H:%M:%S.%f")
    end = datetime.strptime(coverage_dict["Kiel_time"][-1], "%Y-%m-%dT%H:%M:%S.%f")

    duration = end - start
    hours = duration.total_seconds() / 3600

    # Coverage duration should be approximately 9 hours.
    assert hours == pytest.approx(9, abs=1)


@pytest.mark.external_kernel
def test_use_outages(furnish_kernels):
    """
    Test that outages are properly used.
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

    outages = {
        "Kiel": [
            ("2026-09-22T11:50:00.00Z", "2026-09-22T12:10:00Z"),
            ("2026-09-22T13:50:00.00Z", "2026-09-22T14:10:00Z"),
            ("2026-09-23T11:50:00.00Z", "2026-09-23T12:10:00Z"),
        ],
    }

    with furnish_kernels(kernels):
        coverage_dict = generate_coverage("2026-09-22T00:00:00Z", outages)

    expected = np.array(
        [
            "2026-09-22T07:00:00.000",
            "2026-09-22T08:00:00.000",
            "2026-09-22T09:00:00.000",
            "2026-09-22T10:00:00.000",
            "2026-09-22T11:00:00.000",
            "2026-09-22T13:00:00.000",
            "2026-09-22T15:00:00.000",
            "2026-09-22T16:00:00.000",
        ]
    )

    np.testing.assert_array_equal(coverage_dict["Kiel_time"], expected)
