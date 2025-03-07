import numpy as np
import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask


def test_calculate_cullingmask_attitude():
    """Test calculate_cullingmask for attitude culling."""

    spin_numbers = np.array([0, 1])
    energy_bins = np.array([10, 20, 30, 40])
    spin_start_time = np.array([0, 1])

    quality_attitude = np.full(
        spin_numbers.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )
    quality_attitude[1] |= ImapAttitudeUltraFlags.SPINRATE.value

    quality_ena_rates = np.full(
        (len(energy_bins), len(spin_numbers)),
        ImapRatesUltraFlags.NONE.value,
        dtype=np.uint16,
    )

    ds = xr.Dataset(
        {
            "quality_attitude": (("spin_number",), quality_attitude),
            "quality_ena_rates": (
                ("energy_bin_geometric_mean", "spin_number"),
                quality_ena_rates,
            ),
            "spin_start_time": (("spin_number",), spin_start_time),
        },
        coords={
            "spin_number": spin_numbers,
            "energy_bin_geometric_mean": energy_bins,
        },
    )

    result_ds = calculate_cullingmask(
        ds, name="imap_ultra_l1b_45sensor-cullingmask", data_version="v1"
    )

    np.testing.assert_array_equal(result_ds["spin_number"].values, np.array([0]))


def test_calculate_cullingmask_rates():
    """Test calculate_cullingmask for rates culling."""
    spin_numbers = np.array([0, 1, 2, 3])
    energy_bins = np.array([10, 20, 30, 40])
    spin_start_time = np.array([0, 1, 2, 3])

    quality_attitude = np.full(
        spin_numbers.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )

    quality_ena_rates = np.full(
        (len(energy_bins), len(spin_numbers)),
        ImapRatesUltraFlags.NONE.value,
        dtype=np.uint16,
    )

    quality_ena_rates[:, 0] |= ImapRatesUltraFlags.ZEROCOUNTS.value
    quality_ena_rates[0, 1] |= ImapRatesUltraFlags.ZEROCOUNTS.value
    quality_ena_rates[0, 2] |= ImapRatesUltraFlags.HIGHRATES.value
    quality_ena_rates[0, 3] |= (
        ImapRatesUltraFlags.ZEROCOUNTS.value | ImapRatesUltraFlags.HIGHRATES.value
    )

    ds = xr.Dataset(
        {
            "quality_attitude": (("spin_number",), quality_attitude),
            "quality_ena_rates": (
                ("energy_bin_geometric_mean", "spin_number"),
                quality_ena_rates,
            ),
            "spin_start_time": (("spin_number",), spin_start_time),
        },
        coords={
            "spin_number": spin_numbers,
            "energy_bin_geometric_mean": energy_bins,
        },
    )

    result_ds = calculate_cullingmask(
        ds, name="imap_ultra_l1b_45sensor-cullingmask", data_version="v1"
    )

    expected_spins = np.array([0, 1])
    np.testing.assert_array_equal(result_ds["spin_number"].values, expected_spins)
