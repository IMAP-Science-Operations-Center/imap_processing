"""Calculate Pointing Set Grids."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.repoint import get_pointing_times
from imap_processing.spice.time import (
    et_to_met,
    met_to_ttj2000ns,
    sct_to_et,
    ttj2000ns_to_et,
)
from imap_processing.ultra.l1b.ultra_l1b_culling import get_de_rejection_mask
from imap_processing.ultra.l1c.l1c_lookup_utils import (
    calculate_pixels_within_scattering_threshold,
    get_spacecraft_pointing_lookup_tables,
)
from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_efficiencies_and_geometric_function,
    get_helio_adjusted_data,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

logger = logging.getLogger(__name__)


def calculate_helio_pset(
    de_dataset: xr.Dataset,
    goodtimes_dataset: xr.Dataset,
    rates_dataset: xr.Dataset,
    params_dataset: xr.Dataset,
    name: str,
    ancillary_files: dict,
    instrument_id: int,
    species_id: int = 1,
) -> xr.Dataset:
    """
    Create dictionary with defined datatype for Pointing Set Grid Data.

    Parameters
    ----------
    de_dataset : xarray.Dataset
        Dataset containing de data.
    goodtimes_dataset : xarray.Dataset
        Dataset containing goodtimes data.
    rates_dataset : xarray.Dataset
        Dataset containing image rates data.
    params_dataset : xarray.Dataset
        Dataset containing image parameters data.
    name : str
        Name of the dataset.
    ancillary_files : dict
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.
    species_id : int
        Species ID, default of 1 refers to Hydrogen.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    pset_dict: dict[str, np.ndarray] = {}
    # Select only the species we are interested in.
    indices = np.where(de_dataset["species"].values == species_id)[0]
    species_dataset = de_dataset.isel(epoch=indices)
    helio_pset_quality_flags = np.full(
        de_dataset["epoch"].shape, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
    )

    rejected = get_de_rejection_mask(
        species_dataset["quality_scattering"].values,
        species_dataset["quality_outliers"].values,
    )
    de_dataset = species_dataset.isel(epoch=~rejected)

    v_mag_helio_spacecraft = np.linalg.norm(
        species_dataset["velocity_dps_helio"].values, axis=1
    )
    vhat_dps_helio = (
        species_dataset["velocity_dps_helio"].values
        / v_mag_helio_spacecraft[:, np.newaxis]
    )
    intervals, _, energy_bin_geometric_means = build_energy_bins()
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_helio,
        species_dataset["energy_heliosphere"].values,
        intervals,
        nside=128,
    )

    healpix = np.arange(n_pix)

    # Get lookup table for FOR indices by spin phase step
    (
        for_indices_by_spin_phase,
        theta_vals,
        phi_vals,
        ra_and_dec,
        boundary_scale_factors,
    ) = get_spacecraft_pointing_lookup_tables(ancillary_files, instrument_id)
    # Check that the number of rows in the lookup table matches the number of pixels
    if for_indices_by_spin_phase.shape[0] != n_pix:
        logger.warning(
            "The lookup table is expected to have the same number of rows as "
            "the number of HEALPix pixels."
        )

    pixels_below_scattering = calculate_pixels_within_scattering_threshold(
        for_indices_by_spin_phase, theta_vals, phi_vals, ancillary_files, instrument_id
    )
    # Calculate exposure
    constant_exposure = ancillary_files["l1c-90sensor-dps-exposure"]
    df_exposure = pd.read_csv(constant_exposure)
    exposure_time, deadtime_ratios = get_spacecraft_exposure_times(
        df_exposure,
        rates_dataset,
        params_dataset,
        pixels_below_scattering,
        boundary_scale_factors,
    )
    # calculate efficiency and geometric function as a function of energy
    efficiencies, geometric_function = get_efficiencies_and_geometric_function(
        pixels_below_scattering,
        boundary_scale_factors,
        theta_vals,
        phi_vals,
        n_pix,
        ancillary_files,
    )
    # Get midpoint timestamp for pointing.
    # TODO remove sct_to_et conversion
    pointing_start, pointing_stop = get_pointing_times(
        et_to_met(sct_to_et(species_dataset["event_times"].data[0]))
    )
    mid_time = ttj2000ns_to_et(met_to_ttj2000ns((pointing_start + pointing_stop) / 2))
    exposure_time, efficiency, geometric_function = get_helio_adjusted_data(
        mid_time,
        exposure_time,
        geometric_function,
        efficiencies,
        ra_and_dec[:, 0],
        ra_and_dec[:, 1],
    )
    sensitivity = efficiencies * geometric_function

    helio_pset_quality_flags = np.full(
        n_pix, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
    )

    start: float = np.min(de_dataset["event_times"].values)
    end: float = np.max(de_dataset["event_times"].values)

    # Time bins in 30 minute intervals
    time_bins = np.arange(start, end + 1800, 1800)

    # Compute mask for culling the Earth
    compute_culling_mask(
        time_bins,
        6378.1,  # Earth radius
        helio_pset_quality_flags,
    )

    # For ISTP, epoch should be the center of the time bin.
    pset_dict["epoch"] = de_dataset.epoch.data[:1].astype(np.int64)
    pset_dict["counts"] = counts[np.newaxis, ...]
    pset_dict["latitude"] = latitude[np.newaxis, ...]
    pset_dict["longitude"] = longitude[np.newaxis, ...]
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["helio_exposure_factor"] = exposure_time
    pset_dict["pixel_index"] = healpix
    pset_dict["energy_bin_delta"] = np.diff(intervals, axis=1).squeeze()[
        np.newaxis, ...
    ]
    pset_dict["sensitivity"] = sensitivity
    pset_dict["efficiency"] = efficiencies
    pset_dict["geometric_function"] = geometric_function
    pset_dict["dead_time_ratio"] = deadtime_ratios
    pset_dict["spin_phase_step"] = np.arange(len(deadtime_ratios))
    pset_dict["quality_flags"] = helio_pset_quality_flags[np.newaxis, ...]

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
