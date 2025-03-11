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

    culling_ds = calculate_cullingmask(
        ds, name="imap_ultra_l1b_45sensor-badtimes", data_version="v1"
    )
    badtimes_ds = calculate_badtimes(
        ds,
        culling_ds["spin_number"].values,
        name="imap_ultra_l1b_45sensor-badtimes",
        data_version="v1",
    )

    assert not np.any(
        np.isin(culling_ds["spin_number"].values, badtimes_ds["spin_number"].values)
    )
