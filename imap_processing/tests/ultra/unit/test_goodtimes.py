import numpy as np
import xarray as xr

from imap_processing.quality_flags import ImapAttitudeUltraFlags, ImapRatesUltraFlags
from imap_processing.ultra.l1b.goodtimes import calculate_goodtimes


def test_calculate_goodtimes_attitude():
    """Test calculate_goodtimes for attitude culling."""

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
            "epoch": (("spin_number",), np.array([0, 1], dtype="datetime64[ns]")),
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

    result_ds = calculate_goodtimes(ds, name="imap_ultra_l1b_45sensor-goodtimes")

    np.testing.assert_array_equal(result_ds["spin_number"].values, np.array([0, 1]))


def test_calculate_goodtimes_rates():
    """Test calculate_goodtimes for rates culling."""
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

    quality_ena_rates[0, 0] |= ImapRatesUltraFlags.FIRSTSPIN.value
    quality_ena_rates[0, 3] |= ImapRatesUltraFlags.LASTSPIN.value

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
            "epoch": (("spin_number",), np.array([0, 1, 2, 3], dtype="datetime64[ns]")),
            "spin_number": spin_numbers,
            "energy_bin_geometric_mean": energy_bins,
        },
    )

    result_ds = calculate_goodtimes(ds, name="imap_ultra_l1b_45sensor-goodtimes")

    expected_spins = np.array([1, 2])
    np.testing.assert_array_equal(result_ds["spin_number"].values, expected_spins)


def test_calculate_goodtimes_empty():
    """Test calculate_goodtimes when all spins are culled (empty case)."""

    spin_numbers = np.array([0, 1, 2])
    energy_bins = np.array([10, 20, 30])
    spin_start_time = np.array([0, 1, 2])

    quality_attitude = np.full(
        spin_numbers.shape, ImapAttitudeUltraFlags.NONE.value, dtype=np.uint16
    )

    quality_ena_rates = np.full(
        (len(energy_bins), len(spin_numbers)),
        ImapRatesUltraFlags.FIRSTSPIN.value,
        dtype=np.uint16,
    )

    ds = xr.Dataset(
        {
            "epoch": (("spin_number",), np.array([0, 1, 2], dtype="datetime64[ns]")),
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

    goodtimes_ds = calculate_goodtimes(
        ds,
        name="imap_ultra_l1b_45sensor-goodtimes",
    )

    assert goodtimes_ds["spin_number"].values[0] == 4294967295
    assert goodtimes_ds["spin_start_time"].values[0] == -1.0e31
    assert goodtimes_ds["spin_period"].values[0] == -1.0e31
    assert goodtimes_ds["spin_rate"].values[0] == -1.0e31
    assert goodtimes_ds["quality_attitude"].values[0] == 65535
    assert np.all(goodtimes_ds["ena_rates"].values == -1.0e31)
    assert np.all(goodtimes_ds["quality_ena_rates"].values == 65535)
