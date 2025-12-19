"""Module to create pointing sets."""

import logging

import astropy_healpix.healpy as hp
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import interpolate

from imap_processing.spice.geometry import (
    cartesian_to_spherical,
)
from imap_processing.spice.spin import (
    get_spin_data,
)
from imap_processing.ultra.constants import UltraConstants
from imap_processing.ultra.l1b.lookup_utils import (
    get_geometric_factor,
    get_image_params,
    load_geometric_factor_tables,
)
from imap_processing.ultra.l1b.ultra_l1b_culling import (
    get_pulses_per_spin,
)
from imap_processing.ultra.l1b.ultra_l1b_extended import (
    get_efficiency,
    get_efficiency_interpolator,
    get_spin_info,
)
from imap_processing.ultra.l1c.l1c_lookup_utils import (
    build_energy_bins,
    get_static_deadtime_ratios,
)

# TODO: add species binning.
FILLVAL_FLOAT32 = -1.0e31

logger = logging.getLogger(__name__)


def get_energy_delta_minus_plus(
    energy_bin_edges: np.ndarray | None = None,
) -> tuple[NDArray, NDArray]:
    """
    Calculate the energy_delta_minus and energy_delta_plus for use in the CDF.

    Parameters
    ----------
    energy_bin_edges : numpy.ndarray, optional
        Array of energy bin edges. If None, default Ultra energy bins are used.

    Returns
    -------
    bins_energy_delta_minus : np.ndarray
        Array of energy_delta_minus values.
    bins_energy_delta_plus : np.ndarray
        Array of energy_delta_plus values.

    Notes
    -----
    Calculates as the following:
    energy_delta_minus=abs(bin_geom_mean - bin_lower)
    energy_delta_plus=abs(bin_upper - bin_geom_mean)
    where bin_upper and bin_lower are the upper and lower bounds of the energy bins
    and bin_geom_mean is the geometric mean of the energy bin.
    """
    bins, _, bin_geom_means = build_energy_bins(energy_bin_edges)
    bins_energy_delta_plus, bins_energy_delta_minus = [], []
    for bin_edges, bin_geom_mean in zip(bins, bin_geom_means, strict=False):
        bins_energy_delta_plus.append(bin_edges[1] - bin_geom_mean)
        bins_energy_delta_minus.append(bin_geom_mean - bin_edges[0])
    return abs(np.array(bins_energy_delta_minus)), abs(np.array(bins_energy_delta_plus))


def get_spacecraft_histogram(
    vhat: tuple[np.ndarray, np.ndarray, np.ndarray],
    energy: np.ndarray,
    energy_bin_edges: list[tuple[float, float]],
    nside: int = 128,
    nested: bool = False,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute a 2D histogram of the particle data using HEALPix binning.

    Parameters
    ----------
    vhat : tuple[np.ndarray, np.ndarray, np.ndarray]
        The x,y,z-components of the unit velocity vector.
    energy : np.ndarray
        The particle energy.
    energy_bin_edges : list[tuple[float, float]]
        Array of energy bin edges.
    nside : int, optional
        The nside parameter of the Healpix tessellation.
        Default is 128.
    nested : bool, optional
        Whether the Healpix tessellation is nested. Default is False.

    Returns
    -------
    hist : np.ndarray
        A 2D histogram array with shape (n_pix, n_energy_bins).
    latitude : np.ndarray
        Array of latitude values.
    longitude : np.ndarray
        Array of longitude values.
    n_pix : int
        Number of healpix pixels.

    Notes
    -----
    The histogram will work properly for overlapping energy bins, i.e.
    the same energy value can fall into multiple bins if the intervals overlap.

    azimuthal angle [0, 360], elevation angle [-90, 90]
    """
    # vhat = direction in which particle is traveling
    # Make negative to see where it came from
    spherical_coords = cartesian_to_spherical(-np.array(vhat), degrees=True)
    az, el = (
        spherical_coords[..., 1],
        spherical_coords[..., 2],
    )

    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Calculate the corresponding longitude (az) latitude (el)
    # center coordinates
    longitude, latitude = hp.pix2ang(nside, np.arange(n_pix), lonlat=True)

    # Get HEALPix pixel indices for each event
    # HEALPix expects latitude in [-90, 90] so we don't need to change elevation
    hpix_idx = hp.ang2pix(nside, az, el, nest=nested, lonlat=True)

    # Initialize histogram: (n_energy_bins, n_HEALPix pixels)
    hist = np.zeros((len(energy_bin_edges), n_pix))

    # Bin data in energy & HEALPix space
    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        mask = (energy >= e_min) & (energy < e_max)
        # Only count the events that fall within the energy bin
        hist[i, :] += np.bincount(hpix_idx[mask], minlength=n_pix).astype(np.float64)

    return hist, latitude, longitude, n_pix


def get_spacecraft_count_rate_uncertainty(hist: NDArray, exposure: NDArray) -> NDArray:
    """
    Calculate the count rate uncertainty for HEALPix-binned data.

    Parameters
    ----------
    hist : NDArray
        A 2D histogram array with shape (n_pix, n_energy_bins).
    exposure : NDArray
        A 2D array of exposure times with shape (n_pix, n_energy_bins).

    Returns
    -------
    count_rate_uncertainty : NDArray
        Rate uncertainty with shape (n_pix, n_energy_bins) (counts/sec).

    Notes
    -----
    These calculations were based on Eqn 15 from the IMAP-Ultra Algorithm Document.
    """
    count_uncertainty = np.sqrt(hist)

    rate_uncertainty = np.zeros_like(hist)
    valid = exposure > 0
    rate_uncertainty[valid] = count_uncertainty[valid] / exposure[valid]

    return rate_uncertainty


def get_deadtime_ratios(sectored_rates_ds: xr.Dataset) -> xr.DataArray:
    """
    Compute the dead time ratio at each sector.

    A reduction in exposure time (duty cycle) is caused by the flight hardware listening
    for coincidence events that never occur, due to singles starts predominantly from UV
    radiation. The static exposure time for a given Pointing should be reduced by this
    spatially dependent exposure time reduction factor (the dead time). Further
    description is available in section 3.4.3 of the IMAP-Ultra Algorithm Document.

    Parameters
    ----------
    sectored_rates_ds : xarray.Dataset
        Dataset containing sector mode image rates data.

    Returns
    -------
    dead_time_ratio : xarray.DataArray
        Dead time correction factor for each sector.
    """
    tint = (24 / 360) * sectored_rates_ds.spin_durations
    # compute the correction factor for each step in each sector
    start_full_cdf = sectored_rates_ds.start_rf + sectored_rates_ds.start_lf
    coin_stop_nd = (
        sectored_rates_ds.coin_tn
        + sectored_rates_ds.coin_bn
        - sectored_rates_ds.stop_tn
        - sectored_rates_ds.stop_bn
    )
    numerator = np.exp(5e-7 * start_full_cdf) * np.exp(8e-7 * coin_stop_nd) * tint
    denominator = (
        tint
        - (sectored_rates_ds.event_active_time + 2 * sectored_rates_ds.start_pos) * 1e-7
    )

    correction_factor = numerator / denominator
    # Compute dead time ratio
    dead_time_ratios = 1 / correction_factor

    return dead_time_ratios


def get_sectored_rates(rates_ds: xr.Dataset) -> xr.Dataset | None:
    """
    Filter rates dataset to only include sector mode data.

    Identify intervals where ULTRA was in sector mode by examining contiguous runs
    of identical spin values. At the normal 15-second spin period, each 24° sector
    takes ~1 second, so a full spin in sector mode consists of 15 sectors.

    Parameters
    ----------
    rates_ds : xarray.Dataset
        Dataset containing image rates data.

    Returns
    -------
    rates : xarray.Dataset or None
        Rates dataset with only the sector mode data.
    """
    spins = rates_ds.spin.values
    # Check if spins are monotonically increasing
    if not np.all(np.diff(spins) >= 0):
        logger.warning("Spin values in rates dataset are not monotonically increasing.")
    # Get the indices where the spin value changes
    spin_change = np.where(np.diff(spins) != 0)[0] + 1
    # add the first and last index
    spin_change = np.concatenate(([0], spin_change, [len(spins)]))
    # Get the length of each spin run
    # e.g. 0,0,0,3,3,3,4,4 -> 3,3,2
    spin_runs = np.diff(spin_change)
    # Find the indices where the spin run length is exactly 15 (sector mode)
    spin_run_inds = np.where(spin_runs == 15)[0]

    if len(spin_run_inds) == 0:
        return None

    # Get the start indices of each sector mode spin
    sector_starts = spin_change[spin_run_inds]
    sectored_mode_mask = np.zeros(len(spins), dtype=bool)
    starts = np.asarray(sector_starts)
    # Create offsets 0..14 and broadcast
    idx = starts[:, None] + np.arange(15)
    sectored_mode_mask[idx] = True
    # Return the sectored rates dataset
    return rates_ds.isel(epoch=sectored_mode_mask)


def get_deadtime_ratios_by_spin_phase(
    sectored_rates: xr.Dataset | None,
    aux_dataset: xr.Dataset,
    spin_steps: int,
    sensor_id: int | None = None,
    ancillary_files: dict | None = None,
) -> xr.DataArray:
    """
    Calculate nominal deadtime ratios at every spin phase step (1ms res).

    Parameters
    ----------
    sectored_rates : xarray.Dataset, optional
        Dataset containing sector mode image rates data.
    aux_dataset : xarray.Dataset
        Auxiliary dataset containing spin information.
    spin_steps : int
        Number of spin phase steps (e.g. 15000 for 1ms resolution).
    sensor_id : int, optional
        Sensor ID, either 45 or 90.
    ancillary_files : dict, optional
        Dictionary containing ancillary files.

    Returns
    -------
    xarray.DataArray
        Nominal deadtime ratios at every spin phase step.
    """
    if sectored_rates is None or sectored_rates.epoch.size == 0:
        logger.warning(
            "No sector mode data found in the parameters dataset. Using "
            "static dead time ratios from an ancillary file."
        )
        if sensor_id is None or ancillary_files is None:
            raise ValueError(
                "sensor_id and ancillary_files must be provided to "
                "get static deadtime ratios."
            )
        spin_phases_centered, deadtime_ratios = get_static_deadtime_ratios(
            sensor_id, ancillary_files
        )
    else:
        num_spin_sectors = 15
        sector_indices = np.arange(len(sectored_rates["epoch"])) % num_spin_sectors
        # Get timestamps at the start of each spin (sector 0)
        spin_start_indices = np.where(sector_indices == 0)[0]
        met_time = sectored_rates["shcoarse"].values[spin_start_indices]
        spin_ds = get_spin_info(aux_dataset, met_time)
        # Repeat the spin duration for each of the 15 sectors.
        # Sectors are all within a spin so each one corresponds to the same spin
        # duration
        sectored_rates["spin_durations"] = (
            "epoch",
            np.repeat(spin_ds.spin_duration.data, num_spin_sectors),
        )
        deadtime_ratios = get_deadtime_ratios(sectored_rates).data
        # The center spin phase is the closest / most accurate spin phase.
        # There are 24 spin phases per sector so the nominal middle sector spin phases
        # would be: array([ 12., 36., ..., 300., 324.]) for 15 sectors.
        # We can assume each sector 0 starts at spin phase 0
        spin_phases_centered = (sector_indices / num_spin_sectors) * 360.0 + 12.0

    # Create a dataset with spin phases and dead time ratios
    deadtime_by_spin_phase = xr.Dataset(
        {"deadtime_ratio": (("spin_phase",), deadtime_ratios)},
        coords={"spin_phase": xr.DataArray(spin_phases_centered, dims="spin_phase")},
    )
    # Sort the dataset by spin phase (ascending order)
    deadtime_by_spin_phase = deadtime_by_spin_phase.sortby("spin_phase")
    # Group by spin phase and calculate the median dead time ratio for each phase
    deadtime_medians = deadtime_by_spin_phase.groupby("spin_phase").median(skipna=True)
    if np.any(np.isnan(deadtime_medians["deadtime_ratio"].values)):
        if not np.any(np.isfinite(deadtime_medians["deadtime_ratio"].values)):
            raise ValueError("All dead time ratios are NaN, cannot interpolate.")
        logger.warning(
            "Dead time ratios contain NaN values, filtering data to only include "
            "finite values."
        )
    deadtime_medians = deadtime_medians.where(
        np.isfinite(deadtime_medians["deadtime_ratio"]), drop=True
    )
    interpolator = interpolate.PchipInterpolator(
        deadtime_medians["spin_phase"].values, deadtime_medians["deadtime_ratio"].values
    )
    # Calculate the nominal spin phases at the supplied resolution and query the pchip
    # interpolator to get the deadtime ratios.
    nominal_spin_phases = np.arange(0, 360, 360 / spin_steps)
    deadtime_ratios = xr.DataArray(
        interpolator(nominal_spin_phases), dims="spin_phase_step"
    )
    return deadtime_ratios


def calculate_exposure_time(
    deadtime_ratios: xr.DataArray,
    valid_spun_pixels: xr.DataArray,
    boundary_scale_factors: xr.DataArray,
    apply_bsf: bool = True,
) -> xr.DataArray:
    """
    Adjust the exposure time at each pixel to account for dead time.

    Parameters
    ----------
    deadtime_ratios : xarray.DataArray
        Deadtime ratios at each spin phase step.
    valid_spun_pixels : xarray.DataArray
        3D Array of pixels valid at each spin phase step. If rejection based on
        scattering was set, then these are the pixels below the FWHM scattering
        threshold and in the field of regard at each spin phase step, and energy
        shape = (spin_phase_steps, energy_bins, n_pix). IF no rejection,
        then these are simply the pixels in the field of regard at each spin phase step
        shape = (spin_phase_steps, 1, n_pix).
    boundary_scale_factors : xr.DataArray
        Boundary scale factors for each pixel at each spin phase.
    apply_bsf : bool, optional
        Whether to apply boundary scale factors. Default is True.

    Returns
    -------
    exposure_pointing: xarray.DataArray
        Adjusted exposure times accounting for dead time.
    """
    # nominal spin phase step.
    nominal_ms_step = 15 / valid_spun_pixels.shape[0]  # time step
    # Query the dead-time ratio and apply the nominal exposure time to pixels in the FOR
    # and below the scattering threshold (if scattering rejection is on).
    # Sum over the first dim of valid_spun_pixels is the spin phase step.
    # This is like spinning the spacecraft by nominal 1 ms steps in the despun frame.
    all_counts = valid_spun_pixels * deadtime_ratios
    if apply_bsf:
        all_counts *= boundary_scale_factors

    counts = all_counts.sum(dim="spin_phase_step")
    # Multiply by the nominal spin step to get the exposure time in ms
    exposure_pointing = counts * nominal_ms_step
    return exposure_pointing


def get_spacecraft_exposure_times(
    rates_dataset: xr.Dataset,
    valid_spun_pixels: xr.DataArray,
    boundary_scale_factors: xr.DataArray,
    aux_dataset: xr.Dataset,
    pointing_range_met: tuple[float, float],
    n_energy_bins: int,
    sensor_id: int | None = None,
    ancillary_files: dict | None = None,
    apply_bsf: bool = True,
) -> tuple[NDArray, NDArray]:
    """
    Compute exposure times for HEALPix pixels.

    Parameters
    ----------
    rates_dataset : xarray.Dataset
        Dataset containing image rates data.
    valid_spun_pixels : xarray.DataArray
        3D Array of pixels valid at each spin phase step. If rejection based on
        scattering was set, then these are the pixels below the FWHM scattering
        threshold and in the field of regard at each spin phase step, and energy
        shape = (spin_phase_steps, energy_bins, n_pix). IF no rejection,
        then these are simply the pixels in the field of regard at each spin phase step
        shape = (spin_phase_steps, 1, n_pix).
    boundary_scale_factors : xarray.DataArray
        Boundary scale factors for each pixel at each spin phase.
    aux_dataset : xarray.Dataset
        Auxiliary dataset containing spin information.
    pointing_range_met : tuple
        Start and stop time of the pointing period in mission elapsed time.
    n_energy_bins : int
        Number of energy bins.
    sensor_id : int, optional
        Sensor ID, either 45 or 90.
    ancillary_files : dict, optional
        Dictionary containing ancillary files.
    apply_bsf : bool, optional
        Whether to apply boundary scale factors. Default is True.

    Returns
    -------
    exposure_pointing : np.ndarray
        Total exposure times of pixels in a
        Healpix tessellation of the sky
        in the pointing (dps) frame.
    nominal_deadtime_ratios : np.ndarray
        Deadtime ratios at each spin phase step (1ms res).
    """
    sectored_rates = get_sectored_rates(rates_dataset)
    # Get the number of steps used in the spun pointing lookup tables
    spin_steps = valid_spun_pixels.shape[0]
    nominal_deadtime_ratios = get_deadtime_ratios_by_spin_phase(
        sectored_rates, aux_dataset, spin_steps, sensor_id, ancillary_files
    )
    # The exposure time will be approximately the same per spin, so to save
    # computation time, calculate the exposure time for a single spin and then scale it
    # by the number of spins in the pointing. For more information, see section 3.4.3
    # of the Ultra Algorithm Document.
    exposure_time = calculate_exposure_time(
        nominal_deadtime_ratios, valid_spun_pixels, boundary_scale_factors, apply_bsf
    )
    # Use the universal spin table to determine the actual number of spins
    nominal_spin_seconds = 15.0
    spin_data = get_spin_data()
    # Filter for spins only in pointing
    spin_data = spin_data[
        (spin_data["spin_start_met"] >= pointing_range_met[0])
        & (spin_data["spin_start_met"] <= pointing_range_met[1])
    ]
    # Get only valid spin data
    valid_mask = (spin_data["spin_phase_valid"].values == 1) & (
        spin_data["spin_period_valid"].values == 1
    )
    n_spins_in_pointing: float = np.sum(
        spin_data[valid_mask].spin_period_sec / nominal_spin_seconds
    )
    logger.info(
        f"Calculated total spins universal spin table. Found {n_spins_in_pointing} "
        f"valid spins."
    )
    # Adjust exposure time by the actual number of valid spins in the pointing
    exposure_pointing_adjusted = n_spins_in_pointing * exposure_time

    if exposure_pointing_adjusted.shape[0] != n_energy_bins:
        exposure_pointing_adjusted = np.repeat(
            exposure_pointing_adjusted,
            n_energy_bins,
            axis=0,
        )
    return exposure_pointing_adjusted.values, nominal_deadtime_ratios.values


def get_efficiencies_and_geometric_function(
    valid_spun_pixels: xr.DataArray,
    boundary_scale_factors: xr.DataArray,
    theta_vals: np.ndarray,
    phi_vals: np.ndarray,
    npix: int,
    ancillary_files: dict,
    apply_bsf: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the geometric factor and efficiency for each pixel and energy bin.

    The results are averaged over all spin phases.

    Parameters
    ----------
    valid_spun_pixels : xarray.DataArray
        3D Array of pixels valid at each spin phase step. If rejection based on
        scattering was set, then these are the pixels below the FWHM scattering
        threshold and in the field of regard at each spin phase step, and energy
        shape = (spin_phase_steps, energy_bins, n_pix). IF no rejection,
        then these are simply the pixels in the field of regard at each spin phase step
        shape = (spin_phase_steps, 1, n_pix).
    boundary_scale_factors : xarray.DataArray
        Boundary scale factors for each pixel at each spin phase.
    theta_vals : np.ndarray
        2D or 3D array of theta values. Shape is either (spin_phase_step, npix)
        or (spin_phase_step, energy_bins, npix) when energy-dependent scattering
        rejection is used.
    phi_vals : np.ndarray
        Array of phi values with the same shape as `theta_vals`, giving the
        corresponding phi for each pixel (and energy, if present).
    npix : int
        Number of HEALPix pixels.
    ancillary_files : dict
        Dictionary containing ancillary files.
    apply_bsf : bool, optional
        Whether to apply boundary scale factors. Default is True.

    Returns
    -------
    gf_averaged : np.ndarray
        Averaged geometric factors across all spin phases.
        Shape = (n_energy_bins, npix).
    eff_averaged : np.ndarray
        Averaged efficiencies across all spin phases.
        Shape = (n_energy_bins, npix).
    """
    # Load callable efficiency interpolator function
    eff_interpolator, theta_min_max, phi_min_max = get_efficiency_interpolator(
        ancillary_files
    )
    # load geometric factor lookup table
    geometric_lookup_table = load_geometric_factor_tables(
        ancillary_files, "l1b-sensor-gf-blades"
    )
    # Get energy bin geometric means
    energy_bin_geometric_means = build_energy_bins()[2]
    energy_bins = len(energy_bin_geometric_means)
    # clip arrays to avoid out of bounds errors
    logger.info(
        "Clipping Theta and Phi values to valid ranges for the efficiency "
        "interpolation. \n"
        f"Theta valid range: {theta_min_max}, Phi valid range: {phi_min_max}. \n "
        f"Found "
        f"{np.sum((theta_vals < theta_min_max[0]) | (theta_vals > theta_min_max[1]))}"
        f" Theta values out of range. \n"
        f"Found "
        f"{np.sum((phi_vals < phi_min_max[0]) | (phi_vals > phi_min_max[1]))}"
        f" Phi values out of range. \n"
        f"Theta min and max values before clipping: "
        f"{theta_vals.min()}, {theta_vals.max()} \n"
        f"Phi min and max values before clipping:"
        f" {phi_vals.min()}, {phi_vals.max()} \n"
    )
    theta_vals_clipped = np.clip(theta_vals, theta_min_max[0], theta_min_max[1])
    phi_vals_clipped = np.clip(phi_vals, phi_min_max[0], phi_min_max[1])
    # Initialize summation arrays for geometric factors and efficiencies
    gf_summation = np.zeros((energy_bins, npix))
    eff_summation = np.zeros((energy_bins, npix))
    sample_count = np.zeros((energy_bins, npix))
    # Loop through spin phases
    spin_steps = valid_spun_pixels.sizes["spin_phase_step"]
    for i in range(spin_steps):
        # Loop through energy bins
        # Get valid pixels at this spin phase
        valid_at_spin = valid_spun_pixels.isel(
            spin_phase_step=i
        )  # shape: (energy, pixel)

        for energy_bin_idx in range(energy_bins):
            # Determine pixel indices based on energy dependence
            if theta_vals.ndim < 3:
                # Energy independent calculations
                # TODO this may cause performance issues. Revisit later.
                pixel_inds = np.where(valid_at_spin.isel(energy=0))[0]
                # Compute gf and eff for these theta/phi pairs
                theta_at_spin = theta_vals[i, :]
                phi_at_spin = phi_vals[i, :]
                theta_at_spin_clipped = theta_vals_clipped[i, :]
                phi_at_spin_clipped = phi_vals_clipped[i, :]
            else:
                # Energy dependent calculations
                pixel_inds = np.where(valid_at_spin.isel(energy=energy_bin_idx))[0]
                # Compute gf and eff for these theta/phi pairs
                theta_at_spin = theta_vals[i, energy_bin_idx, :]
                phi_at_spin = phi_vals[i, energy_bin_idx, :]
                theta_at_spin_clipped = theta_vals_clipped[i, energy_bin_idx, :]
                phi_at_spin_clipped = phi_vals_clipped[i, energy_bin_idx, :]

            if pixel_inds.size == 0:
                continue

            gf_values = get_geometric_factor(
                phi=phi_at_spin[pixel_inds],
                theta=theta_at_spin[pixel_inds],
                quality_flag=np.zeros(len(phi_at_spin[pixel_inds])).astype(np.uint16),
                geometric_factor_tables=geometric_lookup_table,
            )
            energy = energy_bin_geometric_means[energy_bin_idx]
            # Clip energy to calibrated range
            energy_clipped = np.clip(energy, 3.0, 80.0)
            eff_values = get_efficiency(
                np.full(pixel_inds.size, energy_clipped),
                phi_at_spin_clipped[pixel_inds],
                theta_at_spin_clipped[pixel_inds],
                ancillary_files,
                interpolator=eff_interpolator,
            )

            # Accumulate and sum eff and gf values
            bsfs = (
                boundary_scale_factors[pixel_inds, i]
                if apply_bsf
                else np.ones(len(pixel_inds))
            )
            gf_summation[energy_bin_idx, pixel_inds] += gf_values * bsfs
            eff_summation[energy_bin_idx, pixel_inds] += eff_values * bsfs
            sample_count[energy_bin_idx, pixel_inds] += 1

    # return averaged geometric factors and efficiencies across all spin phases
    # These are now energy dependent.
    gf_averaged = np.zeros_like(gf_summation)
    eff_averaged = np.zeros_like(eff_summation)
    np.divide(gf_summation, sample_count, out=gf_averaged, where=sample_count != 0)
    np.divide(eff_summation, sample_count, out=eff_averaged, where=sample_count != 0)
    return gf_averaged, eff_averaged


def get_spacecraft_background_rates(
    rates_dataset: xr.Dataset,
    aux_dataset: xr.Dataset,
    sensor_id: int,
    ancillary_files: dict,
    energy_bin_edges: list[tuple[float, float]],
    goodtimes_spin_number: NDArray,
    nside: int = 128,
) -> NDArray:
    """
    Calculate background rates based on the provided parameters.

    Parameters
    ----------
    rates_dataset : xr.Dataset
        Rates dataset.
    aux_dataset : xr.Dataset
        Auxiliary dataset.
    sensor_id : int
        Sensor ID: either 45 or 90.
    ancillary_files : dict[Path]
        Ancillary files containing the lookup tables.
    energy_bin_edges : list[tuple[float, float]]
        Energy bin edges.
    goodtimes_spin_number : NDArray
        Goodtime spins.
        Ex. imap_ultra_l1b_45sensor-goodtimes[0]["spin_number"]
        This is used to determine the number of pulses per spin.
    nside : int, optional
        The nside parameter of the Healpix tessellation (default is 128).

    Returns
    -------
    background_rates : NDArray of shape (n_energy_bins, n_HEALPix pixels)
        Calculated background rates.

    Notes
    -----
    See Eqn. 3, 8, and 20 in the Algorithm Document for the equation.
    """
    pulses = get_pulses_per_spin(aux_dataset, rates_dataset)
    # Pulses for the pointing.
    etof_min = get_image_params("eTOFMin", f"ultra{sensor_id}", ancillary_files)
    etof_max = get_image_params("eTOFMax", f"ultra{sensor_id}", ancillary_files)
    spin_ds = get_spin_info(aux_dataset, rates_dataset["shcoarse"].values)
    spin_number = spin_ds["spin_number"].values

    # Get dmin for PH (mm).
    dmin_ctof = UltraConstants.DMIN_PH_CTOF

    # Compute number of HEALPix pixels that cover the sphere
    n_pix = hp.nside2npix(nside)

    # Initialize background rate array: (n_energy_bins, n_HEALPix pixels)
    background_rates = np.zeros((len(energy_bin_edges), n_pix))

    # Only select pulses from goodtimes.
    goodtime_mask = np.isin(spin_number, goodtimes_spin_number)
    mean_start_pulses = np.mean(pulses.start_pulses[goodtime_mask])
    mean_stop_pulses = np.mean(pulses.stop_pulses[goodtime_mask])
    mean_coin_pulses = np.mean(pulses.coin_pulses[goodtime_mask])

    for i, (e_min, e_max) in enumerate(energy_bin_edges):
        # Calculate ctof for the energy bin boundaries by combining Eqn. 3 and 8.
        # Compute speed for min and max energy using E = 1/2mv^2 -> v = sqrt(2E/m)
        vmin = np.sqrt(2 * e_min * UltraConstants.KEV_J / UltraConstants.MASS_H)  # m/s
        vmax = np.sqrt(2 * e_max * UltraConstants.KEV_J / UltraConstants.MASS_H)  # m/s
        # Compute cTOF = dmin / v
        # Multiply times 1e-3 to convert to m.
        ctof_min = dmin_ctof * 1e-3 / vmax * 1e-9  # Convert to ns
        ctof_max = dmin_ctof * 1e-3 / vmin * 1e-9  # Convert to ns

        background_rates[i, :] = (
            np.abs(ctof_max - ctof_min)
            * (etof_max - etof_min)
            * mean_start_pulses
            * mean_stop_pulses
            * mean_coin_pulses
        ) / 30.0

    return background_rates
