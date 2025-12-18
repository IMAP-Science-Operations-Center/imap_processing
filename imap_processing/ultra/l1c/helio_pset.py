"""Calculate Pointing Set Grids."""

import logging

import numpy as np
import xarray as xr

from imap_processing.cdf.utils import parse_filename_like
from imap_processing.quality_flags import ImapPSETUltraFlags
from imap_processing.spice.geometry import SpiceFrame
from imap_processing.spice.repoint import get_pointing_times_from_id
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
)
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.ultra_l1b_culling import get_de_rejection_mask
from imap_processing.ultra.l1c.l1c_lookup_utils import (
    build_energy_bins,
    calculate_fwhm_spun_scattering,
)
from imap_processing.ultra.l1c.make_helio_index_maps import (
    make_helio_index_maps_with_nominal_kernels,
)
from imap_processing.ultra.l1c.ultra_l1c_culling import compute_culling_mask
from imap_processing.ultra.l1c.ultra_l1c_pset_bins import (
    get_efficiencies_and_geometric_function,
    get_energy_delta_minus_plus,
    get_spacecraft_background_rates,
    get_spacecraft_exposure_times,
    get_spacecraft_histogram,
)
from imap_processing.ultra.utils.ultra_l1_utils import create_dataset

logger = logging.getLogger(__name__)


def calculate_helio_pset(
    de_dataset: xr.Dataset,
    goodtimes_dataset: xr.Dataset,
    rates_dataset: xr.Dataset,
    aux_dataset: xr.Dataset,
    name: str,
    ancillary_files: dict,
    instrument_id: int,
    species_id: list,
) -> xr.Dataset | None:
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
    aux_dataset : xarray.Dataset
        Dataset containing auxiliary data.
    name : str
        Name of the dataset.
    ancillary_files : dict
        Ancillary files.
    instrument_id : int
        Instrument ID, either 45 or 90.
    species_id : List
        Species ID.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset containing the data.
    """
    # Do not cull events based on scattering thresholds
    reject_scattering = False
    # Do not apply boundary scale factor corrections
    apply_bsf = False
    nside = 32
    num_spin_steps = 720
    sensor_id = int(parse_filename_like(name)["sensor"][0:2])
    pset_dict: dict[str, np.ndarray] = {}
    # Select only the species we are interested in.
    indices = np.where(np.isin(de_dataset["ebin"].values, species_id))[0]
    if indices.size == 0:
        logger.info(f"No data available for {name}")
        return None

    species_dataset = de_dataset.isel(epoch=indices)

    rejected = get_de_rejection_mask(
        species_dataset["quality_scattering"].values,
        species_dataset["quality_outliers"].values,
        reject_scattering,
    )
    species_dataset = species_dataset.isel(epoch=~rejected)

    v_mag_helio_spacecraft = np.linalg.norm(
        species_dataset["velocity_dps_helio"].values, axis=1
    )
    vhat_dps_helio = (
        species_dataset["velocity_dps_helio"].values
        / v_mag_helio_spacecraft[:, np.newaxis]
    )
    # Get the start and stop times of the pointing period
    repoint_id = species_dataset.attrs.get("Repointing", None)
    if repoint_id is None:
        raise ValueError("Repointing ID attribute is missing from the dataset.")
    instrument_frame = (
        SpiceFrame.IMAP_ULTRA_90 if sensor_id == 90 else SpiceFrame.IMAP_ULTRA_45
    )
    pointing_range_met = get_pointing_times_from_id(repoint_id)

    logger.info("Generating helio pointing lookup tables.")

    helio_pointing_ds = make_helio_index_maps_with_nominal_kernels(
        kernel_paths=UltraConstants.SIM_KERNELS_FOR_HELIO_INDEX_MAPS,
        nside=nside,
        spin_duration=15.0,
        num_steps=num_spin_steps,
        instrument_frame=instrument_frame,
        compute_bsf=apply_bsf,
    )
    boundary_scale_factors = helio_pointing_ds.bsf
    theta_vals = helio_pointing_ds.theta
    phi_vals = helio_pointing_ds.phi
    fov_index = helio_pointing_ds.index

    intervals, _, energy_bin_geometric_means = build_energy_bins()

    logger.info("calculating spun FWHM scattering values.")
    pixels_below_scattering, scattering_theta, scattering_phi, scattering_thresholds = (
        calculate_fwhm_spun_scattering(
            fov_index,
            theta_vals,
            phi_vals,
            ancillary_files,
            instrument_id,
            reject_scattering,
        )
    )

    counts, latitude, longitude, n_pix = get_spacecraft_histogram(
        vhat_dps_helio,
        species_dataset["energy_heliosphere"].values,
        intervals,
        nside=nside,
    )
    helio_pset_quality_flags = np.full(
        n_pix, ImapPSETUltraFlags.NONE.value, dtype=np.uint16
    )
    healpix = np.arange(n_pix)

    logger.info("Calculating spacecraft exposure times with deadtime correction.")
    exposure_time, deadtime_ratios = get_spacecraft_exposure_times(
        rates_dataset,
        pixels_below_scattering,
        boundary_scale_factors,
        aux_dataset,
        pointing_range_met,
        n_energy_bins=len(energy_bin_geometric_means),
        sensor_id=sensor_id,
        ancillary_files=ancillary_files,
        apply_bsf=apply_bsf,
    )
    logger.info("Calculating spun efficiencies and geometric function.")
    # calculate efficiency and geometric function as a function of energy
    geometric_function, efficiencies = get_efficiencies_and_geometric_function(
        pixels_below_scattering,
        boundary_scale_factors,
        theta_vals.values,
        phi_vals.values,
        n_pix,
        ancillary_files,
        apply_bsf,
    )

    logger.info("Calculating background rates.")
    # TODO calculate helio background rates
    # Calculate background rates
    background_rates = get_spacecraft_background_rates(
        rates_dataset,
        aux_dataset,
        sensor_id,
        ancillary_files,
        intervals,
        goodtimes_dataset["spin_number"].values,
        nside=nside,
    )

    sensitivity = efficiencies * geometric_function

    start: float = np.min(species_dataset["event_times"].values)
    end: float = np.max(species_dataset["event_times"].values)

    # Convert pointing start and end time to ttj2000ns
    pointing_range_ns = met_to_ttj2000ns(pointing_range_met)
    # use either the pointing end time + 30 mins or the max event time,
    # whichever is smaller.
    end = min(end + 1800, ttj2000ns_to_et(pointing_range_ns[1]))
    # Time bins in 30 minute intervals in et
    time_bins = np.arange(start, end, 1800)

    # Compute mask for culling the Earth
    compute_culling_mask(
        time_bins,
        UltraConstants.DEFAULT_EARTH_CULLING_RADIUS,
        helio_pset_quality_flags,
        nside=nside,
    )
    # Epoch should be the start of the pointing
    pset_dict["epoch"] = np.atleast_1d(pointing_range_ns[0]).astype(np.int64)
    pset_dict["epoch_delta"] = np.atleast_1d(np.diff(pointing_range_ns)).astype(
        np.int64
    )
    pset_dict["counts"] = counts[np.newaxis, ...]
    pset_dict["latitude"] = latitude[np.newaxis, ...]
    pset_dict["longitude"] = longitude[np.newaxis, ...]
    pset_dict["energy_bin_geometric_mean"] = energy_bin_geometric_means
    pset_dict["background_rates"] = background_rates[np.newaxis, ...]
    pset_dict["exposure_factor"] = exposure_time[np.newaxis, ...]
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

    # Convert FWHM to gaussian uncertainty by dividing by 2.355
    # See algorithm documentation (section 3.5.7, third bullet point) for more details
    pset_dict["scatter_theta"] = scattering_theta / 2.355
    pset_dict["scatter_phi"] = scattering_phi / 2.355
    pset_dict["scatter_threshold"] = scattering_thresholds

    # Add the energy delta plus/minus to the dataset
    energy_delta_minus, energy_delta_plus = get_energy_delta_minus_plus()
    pset_dict["energy_delta_minus"] = energy_delta_minus
    pset_dict["energy_delta_plus"] = energy_delta_plus

    dataset = create_dataset(pset_dict, name, "l1c")

    return dataset
