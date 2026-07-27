"""IMAP-Lo L2 data processing."""

import logging
from typing import cast

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import RectangularSkyMap
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.constants import LoConstants as c  # noqa: N813
from imap_processing.lo.l1c.lo_l1c import compute_pointing_directions
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)
from imap_processing.spice.time import met_to_ttj2000ns, ttj2000ns_to_met

logger = logging.getLogger(__name__)

# The L1B products a map is built from, one set per pointing.
GOODTIMES = "imap_lo_l1b_goodtimes"
BGRATES = "imap_lo_l1b_bgrates"
HISTRATES = "imap_lo_l1b_histrates"

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def lo_l2(
    sci_dependencies: dict, anc_dependencies: list, descriptor: str
) -> list[xr.Dataset]:
    """
    Process IMAP-Lo L1B data into an L2 sky map.

    A map accumulates the histogram counts and exposure of every pointing in
    its window, binned by the sky direction each spin-angle bin was looking in,
    and converts the accumulated counts into an intensity with the instrument's
    geometric factors.

    The inputs are expected to have already been filtered down to the pivot
    angle of the map being made, which is done in pre-processing (see
    ``cli.Lo.pre_processing``) so that the map records only the files it was
    made from as its parents.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of the input datasets, keyed by logical source, each a list
        of datasets covering the pointings of the map window. Must contain
        ``"imap_lo_l1b_goodtimes"``, ``"imap_lo_l1b_bgrates"`` and
        ``"imap_lo_l1b_histrates"``.
    anc_dependencies : list
        List of ancillary file paths. Unused, the calibration constants of the
        map live in ``LoConstants``.
    descriptor : str
        The map descriptor to be produced
        (e.g., "l090-ena-h-sf-nsp-ram-hae-6deg-3mo").

    Returns
    -------
    list[xr.Dataset]
        List containing the processed L2 map.

    Raises
    ------
    NotImplementedError
        If a HEALPix map is requested (only rectangular maps supported for Lo),
        or if the map is of a species other than hydrogen.
    """
    logger.info("Starting IMAP-Lo L2 processing pipeline")

    map_descriptor = MapDescriptor.from_string(descriptor)
    logger.info(f"Processing map for species: {map_descriptor.species}")

    # The geometric factors in LoConstants are hydrogen only.
    if map_descriptor.species != "h":
        raise NotImplementedError(
            f"Cannot make a map of species {map_descriptor.species} for "
            f"{descriptor}. Only hydrogen geometric factors are defined."
        )

    sky_map = map_descriptor.to_empty_map()
    if not isinstance(sky_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Lo")

    pointings = _group_inputs_by_pointing(sci_dependencies)
    logger.info(f"Building {descriptor} from {len(pointings)} pointings")

    shape = (c.N_ESA_LEVELS, sky_map.num_points)
    counts = np.zeros(shape)
    exposure = np.zeros(shape)
    # Background is a rate per ESA level per pointing, so it is accumulated
    # weighted by exposure and divided by the total exposure at the end.
    bg_rate_exposure = np.zeros(shape)
    esa_mode = 0

    for repointing, (goodtimes, bgrates, histrates) in sorted(pointings.items()):
        logger.debug(f"Accumulating {repointing}")
        esa_mode = _get_esa_mode(histrates)
        _accumulate_pointing(
            goodtimes,
            bgrates,
            histrates,
            sky_map,
            map_descriptor,
            counts,
            exposure,
            bg_rate_exposure,
        )

    variables = _calculate_rates_and_intensities(
        counts, exposure, bg_rate_exposure, esa_mode
    )
    dataset = _build_map_dataset(sky_map, variables, esa_mode)

    logger.info("IMAP-Lo L2 processing pipeline completed successfully")
    return [
        sky_map.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=descriptor,
            external_map_dataset=dataset,
        )
    ]


# =============================================================================
# INPUT HANDLING
# =============================================================================


def _group_inputs_by_pointing(sci_dependencies: dict) -> dict[str, tuple]:
    """
    Group the L1B inputs into the (goodtimes, bgrates, histrates) of a pointing.

    Each input product records the pointing it covers in its ``Repointing``
    global attribute. Pointings missing any of the three products cannot be
    mapped and are dropped.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of the input datasets, keyed by logical source.

    Returns
    -------
    dict[str, tuple]
        The (goodtimes, bgrates, histrates) datasets of each pointing, keyed by
        repointing.

    Raises
    ------
    KeyError
        If any of the three required products is missing entirely.
    """
    by_pointing: dict[str, dict[str, xr.Dataset]] = {}
    for logical_source in (GOODTIMES, BGRATES, HISTRATES):
        for dataset in sci_dependencies[logical_source]:
            repointing = dataset.attrs.get("Repointing", "")
            by_pointing.setdefault(repointing, {})[logical_source] = dataset

    pointings = {}
    for repointing, products in by_pointing.items():
        missing = {GOODTIMES, BGRATES, HISTRATES} - set(products)
        if missing:
            logger.warning(f"Dropping {repointing}, it has no {sorted(missing)}")
            continue
        pointings[repointing] = (
            products[GOODTIMES],
            products[BGRATES],
            products[HISTRATES],
        )

    return pointings


def _get_esa_mode(histrates: xr.Dataset) -> int:
    """
    Read the ESA mode of a pointing, defaulting to HiRes.

    Parameters
    ----------
    histrates : xr.Dataset
        The L1B histogram rates of the pointing.

    Returns
    -------
    int
        The ESA mode, 0 for HiRes and 1 for HiThr.
    """
    if "esa_mode" not in histrates:
        return 0
    return int(np.atleast_1d(histrates["esa_mode"].values)[0])


# =============================================================================
# SKY MAP ACCUMULATION
# =============================================================================


def _accumulate_pointing(
    goodtimes: xr.Dataset,
    bgrates: xr.Dataset,
    histrates: xr.Dataset,
    sky_map: RectangularSkyMap,
    map_descriptor: MapDescriptor,
    counts: np.ndarray,
    exposure: np.ndarray,
    bg_rate_exposure: np.ndarray,
) -> None:
    """
    Add one pointing's counts and exposure to the map accumulators.

    Parameters
    ----------
    goodtimes : xr.Dataset
        The L1B goodtimes of the pointing, giving its pivot angle and the
        good-time windows its histograms are accepted within.
    bgrates : xr.Dataset
        The L1B background rates of the pointing, one rate per ESA level.
    histrates : xr.Dataset
        The L1B histogram rates of the pointing, giving the counts and exposure
        of each spin-angle bin.
    sky_map : RectangularSkyMap
        The map being built, used for its pixel grid.
    map_descriptor : MapDescriptor
        The parsed descriptor of the map being made.
    counts : np.ndarray
        Accumulator of shape (esa level, pixel), modified in place.
    exposure : np.ndarray
        Accumulator of shape (esa level, pixel), modified in place.
    bg_rate_exposure : np.ndarray
        Accumulator of shape (esa level, pixel), modified in place.
    """
    species = map_descriptor.species
    pivot_angle = float(np.atleast_1d(goodtimes["pivot"].values)[0])
    gt_start = np.atleast_1d(goodtimes["gt_start_met"].values)
    gt_end = np.atleast_1d(goodtimes["gt_end_met"].values)

    histogram_met = ttj2000ns_to_met(histrates["epoch"].values)
    in_goodtime = np.any(
        (histogram_met[:, np.newaxis] >= gt_start)
        & (histogram_met[:, np.newaxis] <= gt_end),
        axis=1,
    )
    if not in_goodtime.any():
        logger.warning("No histogram epochs fall within the good-time windows.")
        return

    pointing_counts = histrates[f"{species}_counts"].values[in_goodtime].sum(axis=0)
    pointing_exposure = histrates["exposure_time_6deg"].values[in_goodtime].sum(axis=0)
    background_rates = np.atleast_2d(bgrates[f"{species}_background_rates"].values)[0]

    spin_angles = _dps_spin_angles()
    # The whole pointing is projected from the middle of its good times, which
    # is where the despun frame is sampled.
    epoch = met_to_ttj2000ns((gt_start.min() + gt_end.max()) / 2.0)
    az_el = compute_pointing_directions(
        epoch,
        pivot_angle,
        spin_angles=spin_angles,
        off_angles=np.array([0.0]),
        to_frame=cast(SpiceFrame, map_descriptor.map_spice_coord_frame),
    )
    # The single boresight off-angle is squeezed out of the frame transform, so
    # the directions come back as (spin angle, lon/lat).
    az_el = np.asarray(az_el).reshape(spin_angles.size, 2)
    pixels = _pixel_indices(sky_map, az_el[:, 0], az_el[:, 1])

    keep = _spin_phase_mask(spin_angles, pivot_angle, map_descriptor.spin_phase)
    if not keep.any():
        return

    # np.add.at accumulates repeated pixels, which is what happens whenever
    # several spin-angle bins land in the same map pixel.
    np.add.at(counts, (slice(None), pixels[keep]), pointing_counts[:, keep])
    np.add.at(exposure, (slice(None), pixels[keep]), pointing_exposure[:, keep])
    np.add.at(
        bg_rate_exposure,
        (slice(None), pixels[keep]),
        background_rates[:, np.newaxis] * pointing_exposure[:, keep],
    )

    sky_map.min_epoch = min(sky_map.min_epoch, int(met_to_ttj2000ns(gt_start.min())))
    sky_map.max_epoch = max(sky_map.max_epoch, int(met_to_ttj2000ns(gt_end.max())))


def _dps_spin_angles() -> np.ndarray:
    """
    Get the despun-frame azimuth of each histogram spin-angle bin center.

    The L1B histogram spin bins are hardware spin-phase bins referenced to the
    spacecraft spin pulse, NOT the instrument (DPS) spin angle. A bin center is
    converted to the IMAP_DPS azimuth by adding the spacecraft to instrument
    spin-phase offset, exactly as the L1B star-sensor product does.

    Returns
    -------
    np.ndarray
        The IMAP_DPS azimuth [degrees] of each of the histogram spin bins.
    """
    bin_width = 360.0 / c.N_SPIN_ANGLE_BINS
    bin_centers = (np.arange(c.N_SPIN_ANGLE_BINS) + 0.5) * bin_width
    offset = get_spacecraft_to_instrument_spin_phase_offset(SpiceFrame.IMAP_LO) * 360.0
    return np.mod(bin_centers + offset, 360.0)


def _pixel_indices(
    sky_map: RectangularSkyMap, longitude: np.ndarray, latitude: np.ndarray
) -> np.ndarray:
    """
    Get the map pixel each sky direction falls in.

    A rectangular map stores its pixels as a 1D array raveled from the
    (azimuth, elevation) grid, elevation varying fastest.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built.
    longitude : np.ndarray
        Longitudes [degrees] in the map's frame.
    latitude : np.ndarray
        Latitudes [degrees] in the map's frame.

    Returns
    -------
    np.ndarray
        The pixel index of each direction.
    """
    spacing = sky_map.spacing_deg
    num_azimuth, num_elevation = sky_map.binning_grid_shape

    azimuth_index = np.clip(
        (np.mod(longitude, 360.0) // spacing).astype(int), 0, num_azimuth - 1
    )
    elevation_index = np.clip(
        ((latitude + 90.0) // spacing).astype(int), 0, num_elevation - 1
    )
    return azimuth_index * num_elevation + elevation_index


def _spin_phase_mask(
    spin_angles: np.ndarray, pivot_angle: float, spin_phase: str
) -> np.ndarray:
    """
    Get the spin-angle bins belonging on a map of the given spin phase.

    A bin's RAM projection factor is ``sin(pivot) * sin(spin angle)``, positive
    looking into the RAM direction and negative looking away from it.

    Parameters
    ----------
    spin_angles : np.ndarray
        The IMAP_DPS azimuth [degrees] of each spin-angle bin.
    pivot_angle : float
        The pivot angle [degrees] of the pointing.
    spin_phase : str
        The spin phase of the map, "ram", "anti" or "full".

    Returns
    -------
    np.ndarray
        Boolean mask of the bins to keep.

    Raises
    ------
    ValueError
        If the spin phase is not one of "ram", "anti" or "full".
    """
    if spin_phase == "full":
        return np.ones(spin_angles.size, dtype=bool)
    if spin_phase not in ("ram", "anti"):
        raise ValueError(
            f"Invalid spin phase: {spin_phase}. Must be 'ram', 'anti' or 'full'."
        )

    ram_projection = np.sin(np.radians(pivot_angle + c.PIVOT_RAM_OFFSET)) * np.sin(
        np.radians(spin_angles)
    )
    return ram_projection > 0 if spin_phase == "ram" else ram_projection < 0


# =============================================================================
# RATES AND INTENSITIES
# =============================================================================


def _geometric_factors(esa_mode: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the recalibrated geometric factors and their asymmetric bounds.

    Parameters
    ----------
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr. Unused for now, the
        geometric factors are not yet split by ESA mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The geometric factor of each ESA level, and its upper and lower error
        bounds.
    """
    levels = slice(0, c.N_ESA_LEVELS)
    geometric_factor = np.array(c.GEO_FACTOR[levels]) * c.GEO_FACTOR_SCALE
    error = np.array(c.GEO_FACTOR_ERR[levels]) * c.GEO_FACTOR_SCALE

    error_upper = np.hypot(geometric_factor * (c.GEO_FACTOR_SCALE_UPPER - 1.0), error)
    error_lower = np.hypot(geometric_factor * (1.0 - c.GEO_FACTOR_SCALE_LOWER), error)

    return geometric_factor, error_upper, error_lower


def _calculate_rates_and_intensities(
    counts: np.ndarray,
    exposure: np.ndarray,
    bg_rate_exposure: np.ndarray,
    esa_mode: int,
) -> xr.Dataset:
    """
    Turn the accumulated counts and exposure into rates and intensities.

    Every quantity is zero in the pixels that were never exposed.

    Parameters
    ----------
    counts : np.ndarray
        Accumulated counts of shape (esa level, pixel).
    exposure : np.ndarray
        Accumulated exposure time [s] of shape (esa level, pixel).
    bg_rate_exposure : np.ndarray
        Accumulated exposure-weighted background rate, same shape.
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr.

    Returns
    -------
    dict[str, np.ndarray]
        The map variables, each of shape (esa level, pixel).
    """
    energy = np.array(c.ESA_ENERGY[: c.N_ESA_LEVELS])[:, np.newaxis]
    geometric_factor, error_upper, error_lower = _geometric_factors(esa_mode)
    geometric_factor = geometric_factor[:, np.newaxis]
    error_upper = error_upper[:, np.newaxis]
    error_lower = error_lower[:, np.newaxis]

    exposed = exposure > 0

    def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """
        Divide only where the map was exposed, zero elsewhere.

        Parameters
        ----------
        numerator : np.ndarray
            The array being divided.
        denominator : np.ndarray
            The array to divide it by.

        Returns
        -------
        np.ndarray
            The quotient, zero in the pixels that were never exposed.
        """
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(exposure),
            where=exposed,
        )

    count_rate = _divide(counts, exposure)
    # Poisson uncertainty on the counts, propagated to the rate
    count_rate_stat_uncert = _divide(np.sqrt(counts), exposure)

    intensity = _divide(count_rate, geometric_factor * energy)
    intensity_stat_uncert = _divide(count_rate_stat_uncert, geometric_factor * energy)

    # The systematic error is the flux excursion from the recalibrated G-factor
    # bounds: the upper/lower excursions come from the lower/upper G-factor
    # bounds respectively, and the symmetric error is their geometric mean. It
    # is undefined where the lower bound would drive the G-factor non-positive.
    valid = geometric_factor > error_lower
    if not valid.all():
        logger.warning(
            "The geometric factor of ESA levels "
            f"{(np.flatnonzero(~valid[:, 0]) + 1).tolist()} is below its lower "
            f"error bound; their systematic errors are left at zero."
        )
    intensity_sys_err_plus = np.where(
        valid,
        intensity * geometric_factor / (geometric_factor - error_lower) - intensity,
        0.0,
    )
    intensity_sys_err_minus = np.where(
        valid,
        intensity - intensity * geometric_factor / (geometric_factor + error_upper),
        0.0,
    )

    bg_rate = _divide(bg_rate_exposure, exposure)
    bg_rate_stat_uncert = np.sqrt(_divide(bg_rate, exposure))
    bg_intensity = _divide(bg_rate, geometric_factor * energy)
    bg_intensity_stat_uncert = _divide(bg_rate_stat_uncert, geometric_factor * energy)

    return {
        "ena_count": counts,
        "exposure_factor": exposure,
        "ena_count_rate": count_rate,
        "ena_count_rate_stat_uncert": count_rate_stat_uncert,
        "ena_intensity": intensity,
        "ena_intensity_stat_uncert": intensity_stat_uncert,
        "ena_intensity_sys_err": np.sqrt(
            intensity_sys_err_plus * intensity_sys_err_minus
        ),
        "ena_intensity_sys_err_plus": intensity_sys_err_plus,
        "ena_intensity_sys_err_minus": intensity_sys_err_minus,
        "bg_rate": bg_rate,
        "bg_rate_stat_uncert": bg_rate_stat_uncert,
        "bg_intensity": bg_intensity,
        "bg_intensity_stat_uncert": bg_intensity_stat_uncert,
    }


def _build_map_dataset(
    sky_map: RectangularSkyMap, variables: dict[str, np.ndarray], esa_mode: int
) -> xr.Dataset:
    """
    Lay the map variables out on the map's sky grid.

    The variables are handed to the map as 1D pixel arrays, which the map
    rewraps onto its longitude/latitude grid and adds its solid angles to.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built.
    variables : dict[str, np.ndarray]
        The map variables, each of shape (esa level, pixel).
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr, which sets the widths of the
        ESA energy passbands.

    Returns
    -------
    xr.Dataset
        The map variables on the (epoch, energy, longitude, latitude) grid,
        with the energy coordinate and its widths.
    """
    energy = np.array(c.ESA_ENERGY[: c.N_ESA_LEVELS])
    for name, values in variables.items():
        sky_map.data_1d[name] = xr.DataArray(
            values[np.newaxis, ...].astype(np.float32),
            dims=["epoch", "energy", "pixel"],
            coords={"energy": energy},
        )

    dataset = sky_map.to_dataset()

    energy_delta = np.array(c.ESA_ENERGY_DELTA[esa_mode])
    dataset["energy_delta_minus"] = xr.DataArray(energy_delta, dims=["energy"])
    dataset["energy_delta_plus"] = xr.DataArray(energy_delta, dims=["energy"])

    return dataset
