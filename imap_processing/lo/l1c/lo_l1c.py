"""IMAP-Lo L1C Data Processing."""

import logging
from dataclasses import Field
from enum import Enum

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ena_maps.utils.corrections import (
    add_spacecraft_position_and_velocity_to_pset,
)
from imap_processing.spice.geometry import (
    SpiceFrame,
    frame_transform_az_el,
)
from imap_processing.spice.repoint import get_pointing_times_from_id
from imap_processing.spice.spin import get_spin_number
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
)

N_ESA_ENERGY_STEPS = 7
N_SPIN_ANGLE_BINS = 3600
N_OFF_ANGLE_BINS = 40
# 1 time, 7 energy steps, 3600 spin angle bins, and 40 off angle bins
PSET_SHAPE = (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
PSET_DIMS = ["epoch", "esa_energy_step", "spin_angle", "off_angle"]
ESA_ENERGY_STEPS: np.ndarray = (
    np.arange(N_ESA_ENERGY_STEPS) + 1  # 1 to 7 inclusive
)
SPIN_ANGLE_BIN_EDGES = np.linspace(0, 360, N_SPIN_ANGLE_BINS + 1)
SPIN_ANGLE_BIN_CENTERS = (SPIN_ANGLE_BIN_EDGES[:-1] + SPIN_ANGLE_BIN_EDGES[1:]) / 2
OFF_ANGLE_BIN_EDGES = np.linspace(-2, 2, N_OFF_ANGLE_BINS + 1)
OFF_ANGLE_BIN_CENTERS = (OFF_ANGLE_BIN_EDGES[:-1] + OFF_ANGLE_BIN_EDGES[1:]) / 2


class FilterType(str, Enum):
    """
    Enum for the filter types used in the PSET counts.

    The filter types are used to filter the L1B Direct Event dataset
    to only include the specified event types.
    """

    TRIPLES = "triples"
    DOUBLES = "doubles"
    HYDROGEN = "h"
    OXYGEN = "o"
    NONE = ""


def lo_l1c(sci_dependencies: dict, anc_dependencies: list) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L1B data into L1C CDF data products.

    Parameters
    ----------
    sci_dependencies : dict
        Dictionary of datasets needed for L1C data product creation in xarray Datasets.
    anc_dependencies : list
        Ancillary files needed for L1C data product creation.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1c")

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1b_de" in sci_dependencies:
        logical_source = "imap_lo_l1c_pset"
        l1b_de = sci_dependencies["imap_lo_l1b_de"]
        l1b_goodtimes_only = filter_goodtimes(
            l1b_de, sci_dependencies["imap_lo_l1b_goodtimes"]
        )

        # Get the pointing times from the repoint ID stored in the l1b_de dataset
        repoint_id = l1b_de.attrs.get("Repointing", None)
        if repoint_id is None:
            raise ValueError(
                "Repointing ID attribute is missing from the L1B DE dataset."
            )
        pointing_start_met, pointing_end_met = get_pointing_times_from_id(repoint_id)

        # Handle case where no good times are found after filtering,
        # which would lead to an empty dataset with zero counts
        if len(l1b_goodtimes_only["epoch"]) == 0:
            logging.warning(
                "No good times found in L1B DE dataset after filtering. "
                "Creating PSET dataset with zero counts and exposure time."
            )

        pset = xr.Dataset(
            coords={"epoch": np.array([met_to_ttj2000ns(pointing_start_met)])},
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

        pset["pivot_angle"] = sci_dependencies["imap_lo_l1b_goodtimes"]["pivot"]
        pset["pivot_angle_de"] = sci_dependencies["imap_lo_l1b_goodtimes"]["pivot_de"]

        # ESA mode needs to be added to L1B DE. Adding try statement
        # to avoid error until it's available in the dataset
        if "esa_mode" not in l1b_de:
            logging.debug(
                "ESA mode not found in L1B DE dataset. \
                Setting to default value of 0 for Hi-Res."
            )
            pset["esa_mode"] = xr.DataArray(
                np.array([0]),
                dims=["epoch"],
                attrs=attr_mgr.get_variable_attributes("esa_mode"),
            )
        else:
            pset["esa_mode"] = xr.DataArray(
                np.array([l1b_de["esa_mode"].values[0]]),
                dims=["epoch"],
                attrs=attr_mgr.get_variable_attributes("esa_mode"),
            )

        pset["pointing_start_met"] = xr.DataArray(
            np.array([pointing_start_met]),
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("pointing_start_met"),
        )
        pset["pointing_end_met"] = xr.DataArray(
            np.array([pointing_end_met]),
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("pointing_end_met"),
        )

        # Get the start and end spin numbers based on the pointing start and end MET
        start_spin_number = get_spin_number(pset["pointing_start_met"].item())
        end_spin_number = get_spin_number(pset["pointing_end_met"].item())
        pset["start_spin_number"] = xr.DataArray(
            [start_spin_number],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("start_spin_number"),
        )
        pset["end_spin_number"] = xr.DataArray(
            [end_spin_number],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("end_spin_number"),
        )

        # Set the counts
        pset["triples_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.TRIPLES
        )
        pset["doubles_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.DOUBLES
        )
        pset["h_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.HYDROGEN)
        pset["o_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.OXYGEN)

        # Set the exposure time from L1B histrates summed over good-time epochs
        pset["exposure_time"] = calculate_exposure_times(
            sci_dependencies["imap_lo_l1b_histrates"],
            sci_dependencies["imap_lo_l1b_goodtimes"],
        )

        # Set backgrounds
        (
            pset["h_background_rates"],
            pset["h_background_rates_stat_uncert"],
            pset["h_background_rates_sys_err"],
        ) = set_background_rates(
            FilterType.HYDROGEN,
            sci_dependencies,
            attr_mgr,
        )

        (
            pset["o_background_rates"],
            pset["o_background_rates_stat_uncert"],
            pset["o_background_rates_sys_err"],
        ) = set_background_rates(
            FilterType.OXYGEN,
            sci_dependencies,
            attr_mgr,
        )

        # Use pointing midpoint time to query DPS kernel in order to avoid potential
        # querying outside of pointing due to rounding errors
        pointing_midpoint_met = (
            pset["pointing_start_met"].item() + pset["pointing_end_met"].item()
        ) / 2
        pointing_midpoint_ttj2000ns = met_to_ttj2000ns(pointing_midpoint_met)
        pset["hae_longitude"], pset["hae_latitude"] = set_pointing_directions(
            pointing_midpoint_ttj2000ns, attr_mgr, pset["pivot_angle"].values[0].item()
        )

    pset.attrs = attr_mgr.get_global_attributes(logical_source)

    pset = pset.assign_coords(
        {
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        }
    )

    # Get the spacecraft position and velocity and direction
    pset = add_spacecraft_position_and_velocity_to_pset(pset)

    # Update the attributes for the spacecraft position and velocity variables
    pset["sc_position"].attrs.update(attr_mgr.get_variable_attributes("sc_position"))
    pset["sc_velocity"].attrs.update(attr_mgr.get_variable_attributes("sc_velocity"))
    pset["label_vector_HAE"] = xr.DataArray(
        np.array(["x HAE", "y HAE", "z HAE"], dtype=str),
        name="label_vector_HAE",
        dims=[" "],
        attrs=attr_mgr.get_variable_attributes("label_vector_HAE", check_schema=False),
    )

    return [pset]


def filter_goodtimes(l1b_de: xr.Dataset, goodtimes_ds: xr.Dataset) -> xr.Dataset:
    """
    Filter the L1B Direct Event dataset to only include good times.

    The good times are read from the L1B goodtimes dataset produced by
    l1b_bgrates_and_goodtimes.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset.
    goodtimes_ds : xarray.Dataset
        L1B goodtimes dataset containing gt_start_met and gt_end_met variables
        that define good time windows in MET seconds.

    Returns
    -------
    xarray.Dataset
        Filtered L1B Direct Event dataset containing only events within good
        time windows.
    """
    epochs = l1b_de["epoch"].values
    gt_starts = met_to_ttj2000ns(goodtimes_ds["gt_start_met"].values)
    gt_ends = met_to_ttj2000ns(goodtimes_ds["gt_end_met"].values)

    # Keep events that fall within any goodtime window
    in_goodtime = np.any(
        (epochs[:, np.newaxis] >= gt_starts) & (epochs[:, np.newaxis] <= gt_ends),
        axis=1,
    )

    return l1b_de.isel(epoch=in_goodtime)


def get_triple_coincidences(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the triple coincidence events from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_triples : xarray.Dataset
        L1B Direct Event dataset with only triple coincidence events.
    """
    triple_types = ["111111", "111100", "111000"]
    triple_idx = np.nonzero(np.isin(de["coincidence_type"], triple_types))[0]
    de_triples = de.isel(epoch=triple_idx)

    return de_triples


def get_double_coincidences(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the double coincidence events from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_doubles : xarray.Dataset
        L1B Direct Event dataset with only double coincidence events.
    """
    double_types = [
        "110100",
        "110000",
        "101101",
        "101100",
        "101000",
        "100100",
        "100101",
        "100000",
        "011100",
        "011000",
        "010100",
        "010101",
        "010000",
        "001100",
        "001101",
        "001000",
    ]
    double_idx = np.nonzero(np.isin(de["coincidence_type"], double_types))[0]
    de_doubles = de.isel(epoch=double_idx)

    return de_doubles


def _get_peak_mask(
    de: xr.Dataset, peak_lows: list[int], peak_highs: list[int]
) -> np.ndarray:
    """
    Get a boolean mask for events within specified peak ranges.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.
    peak_lows : list[int]
        List of low peak values for each TOF.
    peak_highs : list[int]
        List of high peak values for each TOF.

    Returns
    -------
    peak_mask : numpy.ndarray
        Boolean mask indicating events within the specified peak ranges.
    """
    tof0_s = de["tof0"] + 0.5 * de["tof3"]
    tof1_s = de["tof1"] - 0.5 * de["tof3"]

    peak_mask = (
        (tof0_s >= peak_lows[0])
        & (tof0_s <= peak_highs[0])
        & (tof1_s >= peak_lows[1])
        & (tof1_s <= peak_highs[1])
        & (de["tof2"] >= peak_lows[2])
        & (de["tof2"] <= peak_highs[2])
    )

    return peak_mask


def _get_golden_triple_mask(de: xr.Dataset) -> np.ndarray:
    """
    Get a boolean mask for events within the golden triple coincidence types.

    A golden triple coincidence is only one of the possible triples-types, so
    we need to subset it separately from just triples.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    golden_triple_mask : numpy.ndarray
        Boolean mask indicating events within the golden triple coincidence types.
    """
    return de["coincidence_type"] == "111111"


def get_h_species(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the hydrogen species from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_h : xarray.Dataset
        L1B Direct Event dataset with only hydrogen species.
    """
    h_peak_low = [20, 10, 10]
    h_peak_high = [70, 50, 40]

    golden_triple_mask = _get_golden_triple_mask(de)
    h_peak_mask = _get_peak_mask(de, h_peak_low, h_peak_high)

    h_idx = np.nonzero((golden_triple_mask & h_peak_mask).values)[0]

    de_h = de.isel(epoch=h_idx)
    return de_h


def get_o_species(de: xr.Dataset) -> xr.Dataset:
    """
    Get only the oxygen species from the L1B Direct Event dataset.

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.

    Returns
    -------
    de_o : xarray.Dataset
        L1B Direct Event dataset with only oxygen species.
    """
    co_peak_low = [100, 60, 60]
    co_peak_high = [270, 150, 150]

    golden_triple_mask = _get_golden_triple_mask(de)
    o_peak_mask = _get_peak_mask(de, co_peak_low, co_peak_high)
    o_idx = np.nonzero((golden_triple_mask & o_peak_mask).values)[0]

    de_o = de.isel(epoch=o_idx)
    return de_o


def create_pset_counts(
    de: xr.Dataset, filter_type: FilterType = FilterType.NONE
) -> xr.DataArray:
    """
    Create the PSET counts for the L1B Direct Event dataset.

    The counts are created by binning the data into 3600 longitude bins,
    40 latitude bins, and 7 energy bins. The data is filtered to only
    include counts based on the specified filter: "triples", "doubles", "h", or "o".

    Parameters
    ----------
    de : xarray.Dataset
        L1B Direct Event dataset.
    filter_type : FilterType, optional
        The event type to include in the counts.
        Can be "triples", "doubles", "h", or "o".

    Returns
    -------
    counts : xarray.DataArray
        The counts for the specified filter.
    """
    match filter_type:
        case FilterType.TRIPLES:
            de_filtered = get_triple_coincidences(de)
        case FilterType.DOUBLES:
            de_filtered = get_double_coincidences(de)
        case FilterType.HYDROGEN:
            de_filtered = get_h_species(de)
        case FilterType.OXYGEN:
            de_filtered = get_o_species(de)
        case _:
            # if no filter is specified, use all data
            de_filtered = de

    # stack the filtered data into the 3D array
    data = np.column_stack(
        (
            de_filtered["esa_step"],
            de_filtered["spin_bin"],
            de_filtered["off_angle_bin"],
        )
    )
    # Create the histogram with 3600 longitude bins, 40 latitude bins, and 7 energy bins
    lon_edges = np.arange(3601)
    lat_edges = np.arange(41)
    energy_edges = np.arange(1, 9)

    hist, _edges = np.histogramdd(
        data,
        bins=[energy_edges, lon_edges, lat_edges],
    )

    # add a new axis of size 1 for the epoch
    hist = hist[np.newaxis, :, :, :]

    counts = xr.DataArray(
        data=hist.astype(np.int16),
        dims=PSET_DIMS,
    )

    return counts


def calculate_exposure_times(
    histrates_ds: xr.Dataset, goodtimes_ds: xr.Dataset
) -> xr.DataArray:
    """
    Calculate exposure times from L1B histrates summed over good-time epochs.

    Sum exposure_time_6deg from the L1B histrates dataset over epochs that fall
    within good-time windows, then expands to the full PSET grid.

    The 60-bin (6 deg) spin dimension is expanded to 3600 bins (0.1 deg) by dividing
    each value by 60 and repeating, preserving per-bin exposure. The off_angle
    dimension is filled uniformly by dividing by N_OFF_ANGLE_BINS and broadcasting.

    Parameters
    ----------
    histrates_ds : xr.Dataset
        L1B histogram rates dataset containing exposure_time_6deg with
        shape (n_epochs, esa_step, spin_bin_6).
    goodtimes_ds : xr.Dataset
        L1B goodtimes dataset containing gt_start_met and gt_end_met.

    Returns
    -------
    exposure_time : xr.DataArray
        Shape (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS).
    """
    epochs = histrates_ds["epoch"].values
    gt_starts = met_to_ttj2000ns(goodtimes_ds["gt_start_met"].values)
    gt_ends = met_to_ttj2000ns(goodtimes_ds["gt_end_met"].values)

    in_goodtime = np.any(
        (epochs[:, np.newaxis] >= gt_starts) & (epochs[:, np.newaxis] <= gt_ends),
        axis=1,
    )

    if not in_goodtime.any():
        logging.warning(
            "No histrates epochs fall within good-time windows. "
            "Exposure times will be zero."
        )
        return xr.DataArray(data=np.zeros(PSET_SHAPE, dtype=np.float32), dims=PSET_DIMS)

    # Sum exposure_time_6deg over good-time epochs; shape (7, 60)
    exposure_6deg = histrates_ds["exposure_time_6deg"].values[in_goodtime]
    exposure_sum = exposure_6deg.sum(axis=0)  # (7, 60)
    exposure_3600 = np.repeat(exposure_sum / 60.0, 60, axis=1)  # (7, 3600)

    # Distribute uniformly across 40 off-angle bins
    exposure_3d = np.broadcast_to(
        (exposure_3600 / N_OFF_ANGLE_BINS)[:, :, np.newaxis],
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
    ).copy()

    logging.debug(
        f"Calculated exposure times: good epochs={in_goodtime.sum()}, "
        f"total exposure (6deg sum)={exposure_sum.sum():.1f}s"
    )

    exposure_4d = exposure_3d[np.newaxis, :, :, :]
    return xr.DataArray(data=exposure_4d.astype(np.float32), dims=PSET_DIMS)


def create_datasets(
    attr_mgr: ImapCdfAttributes, logical_source: str, data_fields: list[Field]
) -> xr.Dataset:
    """
    Create a dataset using the populated data classes.

    Parameters
    ----------
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.
    logical_source : str
        The logical source of the data product that's being created.
    data_fields : list[dataclasses.Field]
        List of Fields for data classes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all data product fields in xr.DataArray.
    """
    # TODO: Once L1B DE processing is implemented using the spin packet
    #  and relative L1A DE time to calculate the absolute DE time,
    #  this epoch conversion will go away and the time in the DE dataclass
    #  can be used direction
    epoch_converted_time = [1]

    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1c_pset":
        esa_energy_step = xr.DataArray(
            data=ESA_ENERGY_STEPS,
            name="esa_energy_step",
            dims=["esa_energy_step"],
            attrs=attr_mgr.get_variable_attributes("esa_energy_step"),
        )
        esa_energy_step_label = xr.DataArray(
            esa_energy_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )

        spin_angle = xr.DataArray(
            data=SPIN_ANGLE_BIN_CENTERS,
            name="spin_angle",
            dims=["spin_angle"],
            attrs=attr_mgr.get_variable_attributes("spin_angle"),
        )
        spin_angle_label = xr.DataArray(
            spin_angle.values.astype(str),
            name="spin_angle_label",
            dims=["spin_angle_label"],
            attrs=attr_mgr.get_variable_attributes("spin_angle_label"),
        )

        off_angle = xr.DataArray(
            data=OFF_ANGLE_BIN_CENTERS,
            name="off_angle",
            dims=["off_angle"],
            attrs=attr_mgr.get_variable_attributes("off_angle"),
        )
        off_angle_label = xr.DataArray(
            off_angle.values.astype(str),
            name="off_angle_label",
            dims=["off_angle_label"],
            attrs=attr_mgr.get_variable_attributes("off_angle_label"),
        )

        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "esa_energy_step": esa_energy_step,
                "esa_energy_step_label": esa_energy_step_label,
                "spin_angle": spin_angle,
                "spin_angle_label": spin_angle_label,
                "off_angle": off_angle,
                "off_angle_label": off_angle_label,
            },
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    # Loop through the data fields that were pulled from the
    # data class. These should match the field names given
    # to each field in the YAML attribute file
    for data_field in data_fields:
        field = data_field.name.lower()
        # Create a list of all the dimensions using the DEPEND_I keys in the
        # YAML attributes
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]

        # Create a data array for the current field and add it to the dataset
        # TODO: TEMPORARY. need to update to use l1b data once that's available.
        if field in [
            "pointing_start_met",
            "pointing_end_met",
            "esa_mode",
            "pivot_angle",
        ]:
            dataset[field] = xr.DataArray(
                data=[1],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        elif field == "exposure_time":
            dataset[field] = xr.DataArray(
                data=np.ones((1, 7, 3600, 40), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

        elif "rates" in field:
            dataset[field] = xr.DataArray(
                data=np.ones(PSET_SHAPE, dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                data=np.ones(PSET_SHAPE, dtype=np.int16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

    return dataset


def set_background_rates(
    species: FilterType,
    sci_dependencies: dict,
    attr_mgr: ImapCdfAttributes,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Set the background rates for the specified species.

    Background rates and statistical uncertainties are read from the
    ``imap_lo_l1b_bgrates`` dataset in ``sci_dependencies``. Each species
    provides a 1-D array of shape ``(N_ESA_ENERGY_STEPS,)`` that is broadcast
    uniformly across all spin-angle and off-angle bins. If the bgrates dataset
    is absent, all arrays default to zero.

    Parameters
    ----------
    species : FilterType
        The species to set the background rates for. Can be "h" or "o".
    sci_dependencies : dict
        Science dependency datasets. Expected to contain the key
        ``"imap_lo_l1b_bgrates"`` with variables
        ``"{species}_background_rates"`` and ``"{species}_background_variance"``.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the L1C attributes.

    Returns
    -------
    background_rates : tuple[xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple containing:
        - The background rates for the specified species.
        - The statistical uncertainties for the background rates.
        - The systematic errors for the background rates.
    """
    if species not in {FilterType.HYDROGEN, FilterType.OXYGEN}:
        raise ValueError(f"Species must be 'h' or 'o', but got {species.value}.")

    bg_rates: np.ndarray = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
        dtype=np.float16,
    )
    bg_stat_uncert: np.ndarray = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
        dtype=np.float16,
    )
    bg_sys_err: np.ndarray = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
        dtype=np.float16,
    )

    bgrates_ds = sci_dependencies.get("imap_lo_l1b_bgrates")
    if bgrates_ds is not None:
        species_key = species.value
        rate_field = f"{species_key}_background_rates"
        variance_field = f"{species_key}_background_variance"

        if rate_field in bgrates_ds:
            rates_per_esa = bgrates_ds[rate_field].values[
                0
            ]  # shape: (N_ESA_ENERGY_STEPS,)
            bg_rates = np.broadcast_to(
                rates_per_esa[:, np.newaxis, np.newaxis],
                (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
            ).astype(np.float16)

        if variance_field in bgrates_ds:
            var_per_esa = bgrates_ds[variance_field].values[
                0
            ]  # shape: (N_ESA_ENERGY_STEPS,)
            bg_stat_uncert = np.broadcast_to(
                var_per_esa[:, np.newaxis, np.newaxis],
                (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS),
            ).astype(np.float16)

    bg_rates_data = xr.DataArray(
        data=bg_rates[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(f"{species.value}_background_rates"),
    )
    bg_stat_uncert_data = xr.DataArray(
        data=bg_stat_uncert[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_stat_uncert"
        ),
    )
    bg_sys_err_data = xr.DataArray(
        data=bg_sys_err[np.newaxis, :, :, :],
        dims=["epoch", "esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_sys_err"
        ),
    )

    return bg_rates_data, bg_stat_uncert_data, bg_sys_err_data


def set_pointing_directions(
    epoch: float,
    attr_mgr: ImapCdfAttributes,
    pivot_angle: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Set the pointing directions for the given epoch.

    The pointing directions are calculated by transforming Spin and off angles
    to HAE longitude and latitude using SPICE. This returns the HAE longitude and
    latitude as (3600, 40) arrays for each the latitude and longitude.

    Parameters
    ----------
    epoch : float
        The epoch time in TTJ2000ns.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the L1C attributes.
    pivot_angle : float
        The pivot angle in degrees.
        Off-angles are adjusted relative to this pivot angle before transformation.

    Returns
    -------
    hae_longitude : xr.DataArray
        The HAE longitude for each spin and off angle bin.
    hae_latitude : xr.DataArray
        The HAE latitude for each spin and off angle bin.
    """
    et = ttj2000ns_to_et(epoch)
    # create a meshgrid of spin and off angles using the bin centers
    spin, off = np.meshgrid(
        SPIN_ANGLE_BIN_CENTERS, OFF_ANGLE_BIN_CENTERS, indexing="ij"
    )
    # off_angles need to account for the pivot_angle
    off += 90 - pivot_angle
    dps_az_el = np.stack([spin, off], axis=-1)

    # Transform from DPS Az/El to HAE lon/lat
    hae_az_el = frame_transform_az_el(
        et, dps_az_el, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_HAE, degrees=True
    )

    return xr.DataArray(
        data=hae_az_el[np.newaxis, :, :, 0].astype(np.float64),
        dims=["epoch", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes("hae_longitude"),
    ), xr.DataArray(
        data=hae_az_el[np.newaxis, :, :, 1].astype(np.float64),
        dims=["epoch", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes("hae_latitude"),
    )
