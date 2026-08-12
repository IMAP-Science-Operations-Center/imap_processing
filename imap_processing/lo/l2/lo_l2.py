"""IMAP-Lo L2 data processing."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.ena_maps import (
    PointingSet,
    RectangularSkyMap,
    SkyTilingType,
)
from imap_processing.ena_maps.utils.coordinates import CoordNames
from imap_processing.ena_maps.utils.naming import MapDescriptor
from imap_processing.lo import lo_ancillary
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

# The calibration ancillaries shipped with the package.
ANCILLARY_DATA_DIR = Path(__file__).parent.parent / "ancillary_data"


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
        List of ancillary file paths, read for the efficiency and correction
        factors. The geometric factors and ESA level energies come from the
        ancillaries shipped with the package, in ``ANCILLARY_DATA_DIR``.
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

    # Determine if corrections are needed and prepare oxygen data if required
    (
        _sputtering_correction,
        _bootstrap_correction,
        _flux_correction,
        _o_map_dataset,
        _flux_factors,
        _cg_correction,
    ) = _prepare_corrections(
        map_descriptor, descriptor, sci_dependencies, anc_dependencies
    )

    logger.info("Step 1: Loading ancillary data")
    _efficiency_data = load_efficiency_data(anc_dependencies)

    # Only hydrogen maps are supported end to end for now.
    if map_descriptor.species != "h":
        raise NotImplementedError(
            f"Cannot make a map of species {map_descriptor.species} for "
            f"{descriptor}. Only hydrogen maps are supported."
        )

    sky_map = map_descriptor.to_empty_map()
    if not isinstance(sky_map, RectangularSkyMap):
        raise NotImplementedError("HEALPix map output not supported for Lo")

    pointings = _complete_pointings(sci_dependencies)
    logger.info(f"Building {descriptor} from {len(pointings)} pointings")

    # Every pointing of a map is taken in the same ESA mode, so the last one
    # sets the energy response the whole map is binned in.
    esa_mode = _get_esa_mode(pointings[max(pointings)][2]) if pointings else 0
    calibration = _esa_calibration(map_descriptor.species, esa_mode)

    _initialize_accumulators(sky_map, calibration.energy)

    for repointing, (goodtimes, bgrates, histrates) in sorted(pointings.items()):
        logger.debug(f"Accumulating repoint{repointing:05d}")
        _accumulate_pointing(
            goodtimes,
            bgrates,
            histrates,
            sky_map,
            map_descriptor,
            calibration.energy,
        )

    variables = _calculate_rates_and_intensities(sky_map, calibration)
    dataset = _build_map_dataset(sky_map, variables, calibration)

    logger.info("IMAP-Lo L2 processing pipeline completed successfully")
    return [
        sky_map.build_cdf_dataset(
            instrument="lo",
            level="l2",
            descriptor=descriptor,
            external_map_dataset=dataset,
        )
    ]


def _prepare_corrections(
    map_descriptor: MapDescriptor,
    descriptor: str,
    sci_dependencies: dict,
    anc_dependencies: list,
) -> tuple[bool, bool, bool, xr.Dataset | None, Path | None, bool]:
    """
    Determine what corrections are needed and prepare oxygen dataset if required.

    This helper function encapsulates the logic for determining when sputtering
    and bootstrap corrections should be applied, and handles the creation of
    the oxygen dataset needed for sputtering corrections.

    Parameters
    ----------
    map_descriptor : MapDescriptor
        The parsed map descriptor containing species and data type information.
    descriptor : str
        The original descriptor string for creating the oxygen variant.
    sci_dependencies : dict
        Dictionary of datasets needed for L2 data product creation.
    anc_dependencies : list
        List of ancillary file paths.

    Returns
    -------
    tuple[bool, bool, bool, xr.Dataset | None, Path | None, bool]
        A tuple containing:
        - sputtering_correction: Whether to apply sputtering corrections
        - bootstrap_correction: Whether to apply bootstrap corrections
        - flux_correction: Whether to apply flux corrections
        - o_map_dataset: Oxygen dataset if needed, None otherwise
        - flux_factors: Path to flux factors ancillary file if needed,
         None otherwise
        - cg_correction: Whether to apply CG correction to the dataset.
    """
    # Default values - no corrections needed
    sputtering_correction = False
    bootstrap_correction = False
    flux_correction = False
    o_map_dataset = None
    flux_factors: None | Path = None

    # Sputtering and bootstrap corrections are only applied to hydrogen ENA data
    # Guard against recursion: don't process oxygen for oxygen maps
    if (
        map_descriptor.species == "h"
        and map_descriptor.principal_data == "ena"
        and "-o-" not in descriptor
    ):  # Safety check to prevent infinite recursion
        logger.info("Creating map for oxygen for sputtering corrections")
        o_descriptor = descriptor.replace("-h-", "-o-")
        o_map_dataset = lo_l2(sci_dependencies, anc_dependencies, o_descriptor)[0]
        sputtering_correction = True
        bootstrap_correction = True

    if "raw" not in map_descriptor.principal_data:
        flux_correction = True
        try:
            flux_factors = next(
                x for x in anc_dependencies if "esa-eta-fit-factors" in str(x)
            )
        except StopIteration:
            raise ValueError(
                "No flux correction factor file found in ancillary dependencies"
            ) from None

    cg_correction = True if map_descriptor.frame_descriptor == "hf" else False

    return (
        sputtering_correction,
        bootstrap_correction,
        flux_correction,
        o_map_dataset,
        flux_factors,
        cg_correction,
    )


# =============================================================================
# SETUP AND INITIALIZATION HELPERS
# =============================================================================


def load_efficiency_data(anc_dependencies: list) -> pd.DataFrame:
    """
    Load efficiency factor data from ancillary files.

    Parameters
    ----------
    anc_dependencies : list
        List of ancillary file paths to search for efficiency factor files.

    Returns
    -------
    pd.DataFrame
        Concatenated efficiency factor data from all matching files.
        Returns empty DataFrame if no efficiency files found.
    """
    efficiency_files = [
        anc_file
        for anc_file in anc_dependencies
        if "efficiency-factor" in str(anc_file)
    ]

    if not efficiency_files:
        logger.warning("No efficiency factor files found in ancillary dependencies")
        return pd.DataFrame()

    logger.debug(f"Loading {len(efficiency_files)} efficiency factor files")
    return pd.concat(
        [lo_ancillary.read_ancillary_file(anc_file) for anc_file in efficiency_files],
        ignore_index=True,
    )


def load_sputter_correction_data(
    source_species: str, target_species: str
) -> pd.DataFrame:
    """
    Load sputter correction factors from an ancillary file.

    Parameters
    ----------
    source_species : str
        The species doing the sputtering (e.g. "o" for oxygen).
    target_species : str
        The species being corrected (e.g. "h" for hydrogen).

    Returns
    -------
    pd.DataFrame
        Rows matching the given species pair, sorted ascending by esa_step,
        with columns: source_species, target_species, esa_step,
        sputter_factor, sputter_factor_uncertainty.
    """
    sputter_files = sorted(ANCILLARY_DATA_DIR.glob("*sputter-correction-factors*"))

    if not sputter_files:
        raise ValueError("No sputter correction files found")

    df = pd.concat(
        [lo_ancillary.read_ancillary_file(f) for f in sputter_files],
        ignore_index=True,
    )
    mask = (df["source_species"] == source_species) & (
        df["target_species"] == target_species
    )
    result = df[mask].sort_values("esa_step").reset_index(drop=True)
    return result


def load_bootstrap_correction_data() -> pd.DataFrame:
    """
    Load bootstrap correction factors from an ancillary file.

    Returns
    -------
    pd.DataFrame
        Bootstrap correction factors with columns: esa_step_i, esa_step_k,
        bootstrap_factor. Indices are 1-based ESA step numbers where esa_step_k=8
        refers to the virtual E8 channel.
    """
    bootstrap_files = sorted(ANCILLARY_DATA_DIR.glob("*bootstrap-correction-factors*"))

    if not bootstrap_files:
        raise ValueError("No bootstrap correction factor files found")

    return pd.concat(
        [lo_ancillary.read_ancillary_file(f) for f in bootstrap_files],
        ignore_index=True,
    )


def finalize_dataset(dataset: xr.Dataset, descriptor: str) -> xr.Dataset:
    """
    Add attributes and perform final dataset preparation.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to finalize with attributes.
    descriptor : str
        The descriptor for this map dataset.

    Returns
    -------
    xr.Dataset
        The finalized dataset with all attributes added.
    """
    # Initialize the attribute manager
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-common")
    attr_mgr.add_instrument_variable_attrs(instrument="enamaps", level="l2-rectangular")

    # Add global and variable attributes
    dataset.attrs.update(attr_mgr.get_global_attributes("imap_lo_l2_enamap"))

    # Our global attributes have placeholders for descriptor
    # so iterate through here and fill that in with the map-specific descriptor
    for key in ["Data_type", "Logical_source", "Logical_source_description"]:
        dataset.attrs[key] = dataset.attrs[key].format(descriptor=descriptor)
    for var in dataset.data_vars:
        try:
            dataset[var].attrs = attr_mgr.get_variable_attributes(var)
        except KeyError:
            # If no attributes found, try without schema validation
            try:
                dataset[var].attrs = attr_mgr.get_variable_attributes(
                    var, check_schema=False
                )
            except KeyError:
                logger.warning(f"No attributes found for variable {var}")

    return dataset


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
    energy : np.ndarray
        The energy [keV] of each ESA level.
    """

    tiling_type: SkyTilingType = SkyTilingType.RECTANGULAR

    def __init__(
        self,
        epoch: int,
        pivot_angle: float,
        spin_angles: np.ndarray,
        values: dict[str, np.ndarray],
        frame: SpiceFrame,
        energy: np.ndarray,
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
                    CoordNames.ENERGY_L2.value: energy,
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


def _initialize_accumulators(sky_map: RectangularSkyMap, energy: xr.DataArray) -> None:
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
    energy : xr.DataArray
        The energy [keV] of each ESA level.
    """
    for name in ACCUMULATED_VARIABLES:
        sky_map.data_1d[name] = xr.DataArray(
            np.zeros((1, c.N_ESA_LEVELS, sky_map.num_points)),
            dims=[
                CoordNames.TIME.value,
                CoordNames.ENERGY_L2.value,
                CoordNames.GENERIC_PIXEL.value,
            ],
            coords={CoordNames.ENERGY_L2.value: energy},
        )


def _accumulate_pointing(
    goodtimes: xr.Dataset,
    bgrates: xr.Dataset,
    histrates: xr.Dataset,
    sky_map: RectangularSkyMap,
    map_descriptor: MapDescriptor,
    energy: xr.DataArray,
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
    energy : xr.DataArray
        The energy [keV] of each ESA level.
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
        energy,
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
# GEOMETRIC FACTORS
# =============================================================================


def load_geometric_factor_data(species: str) -> pd.DataFrame:
    """
    Load geometric factor data for the specified species.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").

    Returns
    -------
    pd.DataFrame
        Geometric factor dataframe for the specified species.

    Raises
    ------
    ValueError
        If species is not "h" or "o".
    """
    if species not in ["h", "o"]:
        raise ValueError(
            f"Geometric factors only available for 'h' and 'o', got '{species}'"
        )

    if species == "h":
        gf_file = sorted(ANCILLARY_DATA_DIR.glob("*hydrogen-geometric-factor*"))[-1]
    else:  # species == "o"
        gf_file = sorted(ANCILLARY_DATA_DIR.glob("*oxygen-geometric-factor*"))[-1]

    return lo_ancillary.read_ancillary_file(gf_file)


def reduce_geometric_factor_data(species: str, esa_mode: int) -> pd.DataFrame:
    """
    Get geometric factor data for a specific species and ESA mode.

    This helper function loads geometric factor data, filters by ESA mode, and
    selects the row of each of the 7 energy steps, in ascending step order.

    Parameters
    ----------
    species : str
        The species to load geometric factors for ("h" or "o").
    esa_mode : int
        ESA mode (0 for HiRes, 1 for HiThr).

    Returns
    -------
    pd.DataFrame
        Geometric factor data indexed by Observed_E-Step (1-7), containing all
        columns from the geometric factor CSV file.
    """
    # Load geometric factor data for this species
    gf_data = load_geometric_factor_data(species)

    # Filter for the specific ESA mode
    if "esa_mode" in gf_data.columns:
        gf_data = gf_data[gf_data["esa_mode"] == esa_mode]

    # Lo Instrument team: Use only geometric factors where
    # incident_E-Step == Observed_E-Step
    diagonal = gf_data["incident_E-Step"] == gf_data["Observed_E-Step"]
    gf_data = gf_data[diagonal].set_index("Observed_E-Step")

    # Select the energy steps, in order. Raises if the file is missing one.
    return gf_data.loc[list(range(1, c.N_ESA_LEVELS + 1))]


def _esa_calibration(species: str, esa_mode: int) -> xr.Dataset:
    """
    Get the ESA level calibration one map is built from.

    The ancillary names its two geometric factor uncertainty columns for the
    direction the intensity derived from the factor moves in, which is the
    opposite of the direction the factor itself moves in: intensity goes as
    1/G, so a smaller factor gives a larger intensity. Its ``_unc_plus`` is
    therefore the downward excursion of the factor, and its ``_unc_minus`` the
    upward one.

    Parameters
    ----------
    species : str
        The species of the map ("h" or "o").
    esa_mode : int
        The ESA mode, 0 for HiRes and 1 for HiThr.

    Returns
    -------
    xr.Dataset
        The energies, passband half-widths and geometric factors of every ESA
        level, in ascending level order.
    """
    gf_data = reduce_geometric_factor_data(species, esa_mode).astype(float)

    factor = f"GF_Trpl_{species.upper()}"
    geometric_factor = gf_data[factor].to_numpy()

    arr_by_name = dict(
        energy=gf_data["Cntr_E"].to_numpy(),
        energy_delta_minus=gf_data["Cntr_E_delta_minus"].to_numpy(),
        energy_delta_plus=gf_data["Cntr_E_delta_plus"].to_numpy(),
        geometric_factor=geometric_factor,
        geometric_factor_low=geometric_factor
        - gf_data[f"{factor}_unc_plus"].to_numpy(),
        geometric_factor_high=geometric_factor
        + gf_data[f"{factor}_unc_minus"].to_numpy(),
    )
    data_vars = {
        name: xr.DataArray(arr, dims=[CoordNames.ENERGY_L2.value])
        for name, arr in arr_by_name.items()
    }
    return xr.Dataset(data_vars)


# =============================================================================
# RATES AND INTENSITIES CALCULATIONS
# =============================================================================


def _calculate_rates_and_intensities(
    sky_map: RectangularSkyMap, calibration: xr.Dataset
) -> dict[str, xr.DataArray]:
    """
    Turn the accumulated counts and exposure into rates and intensities.

    Every quantity is zero in the pixels that were never exposed.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map the pointings were projected onto, read for its accumulators.
    calibration : xr.Dataset
        The energy response the map is binned in, read for the energies and
        geometric factors the intensities are derived with.

    Returns
    -------
    dict[str, xr.DataArray]
        The map variables, each of shape (epoch, esa level, pixel).
    """
    counts = sky_map.data_1d["ena_count"]
    exposure = sky_map.data_1d["exposure_factor"]
    bg_rate_exposure = sky_map.data_1d["bg_rate_exposure"]

    energy = calibration["energy"]
    geometric_factor = calibration["geometric_factor"]
    gf_low = calibration["geometric_factor_low"]
    gf_high = calibration["geometric_factor_high"]

    exposed = exposure > 0

    def _divide(numerator: xr.DataArray, denominator: xr.DataArray) -> xr.DataArray:
        """
        Divide only where the map was exposed, zero elsewhere.

        Parameters
        ----------
        numerator : xr.DataArray
            The array being divided.
        denominator : xr.DataArray
            The array to divide it by.

        Returns
        -------
        xr.DataArray
            The quotient, zero in the pixels that were never exposed.
        """
        return (numerator / denominator).where(exposed, 0)

    count_rate = _divide(counts, exposure)
    # Poisson uncertainty on the counts, propagated to the rate
    count_rate_stat_uncert = _divide(np.sqrt(counts), exposure)

    intensity = _divide(count_rate, geometric_factor * energy)
    intensity_stat_uncert = _divide(count_rate_stat_uncert, geometric_factor * energy)

    # The systematic error is the intensity excursion from the recalibrated
    # G-factor bounds, and the symmetric error is the geometric mean of the two.
    # Intensity goes as 1/G, so the lower G-factor bound gives the upper
    # intensity. It is undefined where that bound is not positive.
    valid = gf_low > 0
    if not valid.all():
        logger.warning(
            "The geometric factor of ESA levels "
            f"{(np.flatnonzero(~valid.values) + 1).tolist()} is below its lower "
            f"error bound; their systematic errors are left at zero."
        )
    intensity_upper = _divide(count_rate, gf_low.where(valid, 1.0) * energy)
    intensity_lower = _divide(count_rate, gf_high * energy)
    intensity_sys_err_plus = (intensity_upper - intensity).where(valid, 0.0)
    intensity_sys_err_minus = (intensity - intensity_lower).where(valid, 0.0)

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
    sky_map: RectangularSkyMap,
    variables: dict[str, xr.DataArray],
    calibration: xr.Dataset,
) -> xr.Dataset:
    """
    Lay the map variables out on the map's sky grid.

    The variables are handed to the map as 1D pixel arrays, which the map
    rewraps onto its longitude/latitude grid and adds its solid angles to.

    Parameters
    ----------
    sky_map : RectangularSkyMap
        The map being built.
    variables : dict[str, xr.DataArray]
        The map variables, each of shape (epoch, esa level, pixel).
    calibration : xr.Dataset
        The energy response the map is binned in, read for the widths of the
        ESA energy passbands.

    Returns
    -------
    xr.Dataset
        The map variables on the (epoch, energy, longitude, latitude) grid,
        with the energy coordinate and its widths.
    """
    for name, values in variables.items():
        sky_map.data_1d[name] = values.astype(np.float32)
    # `bg_rate_exposure` is an accumulator, not a map variable.
    sky_map.data_1d = sky_map.data_1d.drop_vars("bg_rate_exposure")

    dataset = sky_map.to_dataset()

    dataset["energy_delta_minus"] = calibration["energy_delta_minus"]
    dataset["energy_delta_plus"] = calibration["energy_delta_plus"]

    return dataset
