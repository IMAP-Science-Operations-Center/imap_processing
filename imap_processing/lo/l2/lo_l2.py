"""IMAP-Lo L2 data processing."""

import logging

import numpy as np
import xarray as xr

from imap_processing.ena_maps.ena_maps import (
    PointingSet,
    RectangularSkyMap,
    SkyTilingType,
)
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo.constants import LoConstants as c  # noqa: N813
from imap_processing.lo.l1c.lo_l1c import compute_pointing_directions
from imap_processing.spice.geometry import (
    SpiceFrame,
    get_spacecraft_to_instrument_spin_phase_offset,
)
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

logger = logging.getLogger(__name__)

# The descriptors of the L1B products a map is built from, one set per pointing.
REQUIRED_PRODUCTS = ("goodtimes", "bgrates", "histrates")

# The map variables accumulated directly from the pointings, before any rates
# or intensities are derived from them.
ACCUMULATED_VARIABLES = ("ena_count", "exposure_factor", "bg_rate_exposure")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def lo_l2(
    sci_dependencies: dict[int, dict[str, xr.Dataset]],
    anc_dependencies: list,
    descriptor: str,
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
    sci_dependencies : dict[int, dict[str, xr.Dataset]]
        The input datasets covering the pointings of the map window, keyed by
        repointing and then by product descriptor.
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

    pointings = _complete_pointings(sci_dependencies)
    logger.info(f"Building {descriptor} from {len(pointings)} pointings")

    _initialize_accumulators(sky_map)
    esa_mode = 0

    for repointing, (goodtimes, bgrates, histrates) in sorted(pointings.items()):
        logger.debug(f"Accumulating repoint{repointing:05d}")
        esa_mode = _get_esa_mode(histrates)
        _accumulate_pointing(goodtimes, bgrates, histrates, sky_map, map_descriptor)

    variables = _calculate_rates_and_intensities(sky_map, esa_mode)
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


def _complete_pointings(
    sci_dependencies: dict[int, dict[str, xr.Dataset]],
) -> dict[int, tuple]:
    """
    Reduce the grouped inputs to the pointings that can be mapped.

    Parameters
    ----------
    sci_dependencies : dict[int, dict[str, xr.Dataset]]
        The input datasets of each pointing, keyed by repointing and then by
        product descriptor.

    Returns
    -------
    dict[int, tuple]
        The (goodtimes, bgrates, histrates) datasets of each mappable pointing,
        keyed by repointing.

    Raises
    ------
    KeyError
        If any of the three required products is missing entirely.
    """
    found = {product for products in sci_dependencies.values() for product in products}
    missing_products = set(REQUIRED_PRODUCTS) - found
    if missing_products:
        raise KeyError(f"No input files for {sorted(missing_products)}")

    pointings = {}
    for repointing, products in sci_dependencies.items():
        missing = set(REQUIRED_PRODUCTS) - set(products)
        if missing:
            logger.warning(
                f"Dropping repoint{repointing:05d}, it has no {sorted(missing)}"
            )
            continue
        pointings[repointing] = tuple(products[p] for p in REQUIRED_PRODUCTS)

    return pointings


def _esa_energy() -> np.ndarray:
    """
    Get the energy of each ESA level the map is binned in.

    Returns
    -------
    np.ndarray
        The energy [keV] of each ESA level.
    """
    return np.array(c.ESA_ENERGY[: c.N_ESA_LEVELS])


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


class LoSpinAnglePointingSet(PointingSet):
    """
    The spin-angle bins of one pointing, as an in-memory pointing set.

    Lo builds its maps straight from the L1B products of a pointing rather than
    from a written L1C pointing set, so the sky direction of each spin-angle
    bin and the values looking in it are assembled here.

    Parameters
    ----------
    epoch : int
        The time [TTJ2000 ns] the pointing is projected from.
    pivot_angle : float
        The pivot angle [degrees] of the pointing.
    spin_angles : np.ndarray
        The IMAP_DPS azimuth [degrees] of each spin-angle bin.
    values : dict[str, np.ndarray]
        The values of the pointing, each of shape (esa level, spin angle).
    frame : SpiceFrame
        The frame to compute the sky directions in, i.e. the map's frame.
    """

    tiling_type: SkyTilingType = SkyTilingType.RECTANGULAR

    def __init__(
        self,
        epoch: int,
        pivot_angle: float,
        spin_angles: np.ndarray,
        values: dict[str, np.ndarray],
        frame: SpiceFrame,
    ):
        dims = [CoordNames.TIME.value, CoordNames.ENERGY_L2.value, "spin_angle"]
        super().__init__(
            xr.Dataset(
                {
                    name: (dims, value[np.newaxis, ...])  # add epoch axis
                    for name, value in values.items()
                },
                coords={
                    CoordNames.TIME.value: [epoch],
                    CoordNames.ENERGY_L2.value: _esa_energy(),
                },
            ),
            spice_reference_frame=frame,
        )
        self.spatial_coords = ("spin_angle",)

        az_el = compute_pointing_directions(
            epoch,
            pivot_angle,
            spin_angles=spin_angles,
            off_angles=np.array([0.0]),
            to_frame=frame,
        )
        self.az_el_points = xr.DataArray(
            np.asarray(az_el),
            dims=[CoordNames.GENERIC_PIXEL.value, CoordNames.AZ_EL_VECTOR.value],
        )

    @property
    def midpoint_j2000_et(self) -> float:
        """
        The time the pointing is projected from.

        The base class derives this from an ``epoch_delta``; a pointing built
        here is handed the single epoch it is projected from directly.

        Returns
        -------
        float
            The epoch of the pointing set [J2000 ET].
        """
        return float(ttj2000ns_to_et(self.epoch))


def _initialize_accumulators(sky_map: RectangularSkyMap) -> None:
    """
    Seed the map with the empty accumulators each pointing is added into.

    ``project_pset_values_to_map`` creates a map variable the first time it
    projects one, so seeding them is what lets the rest of the pipeline read
    the accumulators unconditionally, however many pointings turn out to be
    usable.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built, modified in place.
    """
    for name in ACCUMULATED_VARIABLES:
        sky_map.data_1d[name] = xr.DataArray(
            np.zeros((1, c.N_ESA_LEVELS, sky_map.num_points)),
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_L2.value,
                CoordNames.GENERIC_PIXEL.value,
            ],
            coords={CoordNames.ENERGY_L2.value: _esa_energy()},
        )


def _accumulate_pointing(
    goodtimes: xr.Dataset,
    bgrates: xr.Dataset,
    histrates: xr.Dataset,
    sky_map: RectangularSkyMap,
    map_descriptor: MapDescriptor,
) -> None:
    """
    Add one pointing's counts and exposure to the map.

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
        The map being built, modified in place.
    map_descriptor : MapDescriptor
        The parsed descriptor of the map being made.
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

    spin_angles = _dps_spin_angles()
    keep = _spin_phase_mask(spin_angles, pivot_angle, map_descriptor.spin_phase)
    if not keep.any():
        return

    pointing_counts = histrates[f"{species}_counts"].values[in_goodtime].sum(axis=0)
    pointing_exposure = histrates["exposure_time_6deg"].values[in_goodtime].sum(axis=0)
    background_rates = np.atleast_2d(bgrates[f"{species}_background_rates"].values)[0]

    # The whole pointing is projected from the middle of its good times, which
    # is where the despun frame is sampled.
    epoch = met_to_ttj2000ns((gt_start.min() + gt_end.max()) / 2.0)
    pointing_set = LoSpinAnglePointingSet(
        epoch,
        pivot_angle,
        spin_angles,
        {
            "ena_count": pointing_counts,
            "exposure_factor": pointing_exposure,
            # Background is a rate per ESA level per pointing, so it is
            # accumulated weighted by exposure and divided by the total
            # exposure at the end.
            "bg_rate_exposure": background_rates[:, np.newaxis] * pointing_exposure,
        },
        sky_map.spice_reference_frame,
    )
    # The projection sums the spin-angle bins that land in the same map pixel,
    # and adds this pointing on top of what the earlier pointings left there.
    sky_map.project_pset_values_to_map(
        pointing_set,
        value_keys=list(ACCUMULATED_VARIABLES),
        pset_valid_mask=keep,
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
    sky_map: RectangularSkyMap, esa_mode: int
) -> dict[str, np.ndarray]:
    """
    Turn the accumulated counts and exposure into rates and intensities.

    Every quantity is zero in the pixels that were never exposed.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map the pointings were projected onto, read for its accumulators.
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr.

    Returns
    -------
    dict[str, np.ndarray]
        The map variables, each of shape (epoch, esa level, pixel).
    """
    counts = sky_map.data_1d["ena_count"].values
    exposure = sky_map.data_1d["exposure_factor"].values
    bg_rate_exposure = sky_map.data_1d["bg_rate_exposure"].values

    energy = _esa_energy()[:, np.newaxis]
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
        The map variables, each of shape (epoch, esa level, pixel).
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr, which sets the widths of the
        ESA energy passbands.

    Returns
    -------
    xr.Dataset
        The map variables on the (epoch, energy, longitude, latitude) grid,
        with the energy coordinate and its widths.
    """
    dims = sky_map.data_1d["ena_count"].dims
    for name, values in variables.items():
        sky_map.data_1d[name] = xr.DataArray(values.astype(np.float32), dims=dims)
    # `bg_rate_exposure` is an accumulator, not a map variable.
    sky_map.data_1d = sky_map.data_1d.drop_vars("bg_rate_exposure")

    dataset = sky_map.to_dataset()

    energy_delta = np.array(c.ESA_ENERGY_DELTA[esa_mode])
    dataset["energy_delta_minus"] = xr.DataArray(energy_delta, dims=["energy"])
    dataset["energy_delta_plus"] = xr.DataArray(energy_delta, dims=["energy"])

    return dataset
