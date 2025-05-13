import numpy as np
import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask


def test_calculate_badtimes():
    """Test calculate_badtimes."""

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
            "epoch": np.array([0, 1, 2, 3], dtype="datetime64[ns]"),
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

    culling_ds = calculate_cullingmask(ds, name="imap_ultra_l1b_45sensor-badtimes")
    badtimes_ds = calculate_badtimes(
        ds,
        culling_ds["spin_number"].values,
        name="imap_ultra_l1b_45sensor-badtimes",
    )

    assert not np.any(
        np.isin(culling_ds["spin_number"].values, badtimes_ds["spin_number"].values)
    )


def test_calculate_badtimes_empty():
    """Test calculate_badtimes when all spins are culled (empty case)."""

    spin_numbers = np.array([0, 1, 2])
    energy_bins = np.array([10, 20, 30])
    spin_start_time = np.array([0, 1, 2])

    quality_attitude = np.full(
        spin_numbers.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )

    quality_ena_rates = np.full(
        (len(energy_bins), len(spin_numbers)),
        ImapRatesUltraFlags.ZEROCOUNTS.value | ImapRatesUltraFlags.HIGHRATES.value,
        dtype=np.uint16,
    )

    ds = xr.Dataset(
        {
            "epoch": np.array([0, 1, 2], dtype="datetime64[ns]"),
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

    badtimes_ds = calculate_badtimes(
        ds,
        spin_numbers,
        name="imap_ultra_l1b_45sensor-badtimes",
    )

    assert badtimes_ds["spin_number"].values[0] == 4294967295
    assert badtimes_ds["spin_start_time"].values[0] == -1.0e31
    assert badtimes_ds["spin_period"].values[0] == -1.0e31
    assert badtimes_ds["spin_rate"].values[0] == -1.0e31
    assert badtimes_ds["quality_attitude"].values[0] == 65535
    assert np.all(badtimes_ds["ena_rates"].values == -1.0e31)
    assert np.all(badtimes_ds["quality_ena_rates"].values == 65535)
