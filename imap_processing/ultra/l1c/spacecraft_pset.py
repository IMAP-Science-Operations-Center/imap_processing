"""Calculate Pointing Set Grids."""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.ultra.l1b.ultra_l1b_culling import get_de_rejection_mask
from imap_processing.ultra.l1c.l1c_lookup_utils import (
    calculate_pixels_within_scattering_threshold,
    get_spacecraft_pointing_lookup_tables,
)
from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    build_energy_bins,
    get_efficiencies_and_geometric_function,
    get_spacecraft_background_rates,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

logger = logging.getLogger(__name__)


def calculate_spacecraft_pset(
    de_dataset: xr.Dataset,
    extendedspin_dataset: xr.Dataset,
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
    extendedspin_dataset : xarray.Dataset
        Dataset containing extendedspin data.
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

    sensor = parse_filename_like(name)["sensor"][0:2]
    # Select only the species we are interested in.
    indices = np.where(de_dataset["species"].values == species_id)[0]
    species_dataset = de_dataset.isel(epoch=indices)

    # Before we use the de_dataset to calculate the pointing set grid we need to filter.
    rejected = get_de_rejection_mask(
        species_dataset["quality_scattering"].values,
        species_dataset["quality_outliers"].values,
    )
    species_dataset = species_dataset.isel(epoch=~rejected)

    v_mag_dps_spacecraft = np.linalg.norm(
        species_dataset["velocity_dps_sc"].values, axis=1
    )
    vhat_dps_spacecraft = (
        species_dataset["velocity_dps_sc"].values / v_mag_dps_spacecraft[:, np.newaxis]
    )

    intervals, _, energy_bin_geometric_means = build_energy_bins()
    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_spacecraft,
        species_dataset["energy_spacecraft"].values,
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
    # calculate efficiency and geometric function as a function of energy
    efficiencies, geometric_function = get_efficiencies_and_geometric_function(
        pixels_below_scattering,
        boundary_scale_factors,
        theta_vals,
        phi_vals,
        n_pix,
        ancillary_files,
    )
    sensitivity = efficiencies * geometric_function

    # Calculate exposure
    constant_exposure = ancillary_files["l1c-90sensor-dps-exposure"]
    df_exposure = pd.read_csv(constant_exposure)

    exposure_pointing, deadtime_ratios = get_spacecraft_exposure_times(
        df_exposure,
        rates_dataset,
        params_dataset,
        pixels_below_scattering,
        boundary_scale_factors,
    )

    # Calculate background rates
    background_rates = get_spacecraft_background_rates(
        rates_dataset,
        sensor,
        ancillary_files,
        intervals,
        goodtimes_dataset["spin_number"].values,
    )
    spacecraft_pset_quality_flags = np.full(
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
        spacecraft_pset_quality_flags,
    )

    # For ISTP, epoch should be the center of the time bin.
    pset_dict["epoch"] = de_dataset.epoch.data[:1].astype(np.int64)
    pset_dict["counts"] = counts[np.newaxis, ...]
    pset_dict["latitude"] = latitude[np.newaxis, ...]
    pset_dict["longitude"] = longitude[np.newaxis, ...]
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates[np.newaxis, ...]
    pset_dict["exposure_factor"] = exposure_pointing
    pset_dict["pixel_index"] = healpix
    pset_dict["energy_bin_delta"] = np.diff(intervals, axis=1).squeeze()[
        np.newaxis, ...
    ]
    pset_dict["quality_flags"] = spacecraft_pset_quality_flags[np.newaxis, ...]

    pset_dict["sensitivity"] = sensitivity
    pset_dict["efficiency"] = efficiencies
    pset_dict["geometric_function"] = geometric_function
    pset_dict["dead_time_ratio"] = deadtime_ratios
    pset_dict["spin_phase_step"] = np.arange(len(deadtime_ratios))

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
