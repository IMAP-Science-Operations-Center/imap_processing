"""IMAP-Lo L1C Data Processing."""

import logging
from dataclasses import Field
from enum import Enum

import numpy as np
import xarray as xr
from scipy.stats import binned_statistic_dd

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.lo import lo_ancillary
from imap_processing.lo.l1b.lo_l1b import set_bad_or_goodtimes
from imap_processing.spice.geometry import SpiceFrame, frame_transform_az_el
from imap_processing.spice.repoint import get_pointing_times
from imap_processing.spice.spin import get_spin_number
from imap_processing.spice.time import (
    met_to_ttj2000ns,
    ttj2000ns_to_et,
    ttj2000ns_to_met,
)

N_ESA_ENERGY_STEPS = 7
N_SPIN_ANGLE_BINS = 3600
N_OFF_ANGLE_BINS = 40
# 1 time, 7 energy steps, 3600 spin angle bins, and 40 off angle bins
PSET_SHAPE = (1, N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS)
PSET_DIMS = ["epoch", "esa_energy_step", "spin_angle", "off_angle"]
ESA_ENERGY_STEPS = np.arange(N_ESA_ENERGY_STEPS) + 1  # 1 to 7 inclusive
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
        l1b_goodtimes_only = filter_goodtimes(l1b_de, anc_dependencies)
        # TODO: Need to handle case where no good times are found
        # Set the pointing start and end times based on the first epoch
        pointing_start_met, pointing_end_met = get_pointing_times(
            ttj2000ns_to_met(l1b_goodtimes_only["epoch"][0].item())
        )

        pset = xr.Dataset(
            coords={"epoch": np.array([met_to_ttj2000ns(pointing_start_met)])},
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

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
        pset["start_spin_number"] = xr.DataArray(
            [get_spin_number(pset["pointing_start_met"].item())],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("start_spin_number"),
        )
        pset["end_spin_number"] = xr.DataArray(
            [get_spin_number(pset["pointing_end_met"].item())],
            dims="epoch",
            attrs=attr_mgr.get_variable_attributes("end_spin_number"),
        )

        full_counts = create_pset_counts(l1b_de, FilterType.NONE)

        # Set the counts
        pset["triples_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.TRIPLES
        )
        pset["doubles_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.DOUBLES
        )
        pset["h_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.HYDROGEN)
        pset["o_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.OXYGEN)

        # Set the exposure time
        pset["exposure_time"] = calculate_exposure_times(
            full_counts, l1b_goodtimes_only
        )

        # Set backgrounds
        (
            pset["h_background_rates"],
            pset["h_background_rates_stat_uncert"],
            pset["h_background_rates_sys_err"],
        ) = set_background_rates(
            pset["pointing_start_met"].item(),
            pset["pointing_end_met"].item(),
            FilterType.HYDROGEN,
            anc_dependencies,
            attr_mgr,
        )

        (
            pset["o_background_rates"],
            pset["o_background_rates_stat_uncert"],
            pset["o_background_rates_sys_err"],
        ) = set_background_rates(
            pset["pointing_start_met"].item(),
            pset["pointing_end_met"].item(),
            FilterType.OXYGEN,
            anc_dependencies,
            attr_mgr,
        )

        pset["hae_longitude"], pset["hae_latitude"] = set_pointing_directions(
            pset["epoch"].item()
        )

    pset.attrs = attr_mgr.get_global_attributes(logical_source)

    pset = pset.assign_coords(
        {
            "esa_energy_step": ESA_ENERGY_STEPS,
            "spin_angle": SPIN_ANGLE_BIN_CENTERS,
            "off_angle": OFF_ANGLE_BIN_CENTERS,
        }
    )

    return [pset]


def filter_goodtimes(l1b_de: xr.Dataset, anc_dependencies: list) -> xr.Dataset:
    """
    Filter the L1B Direct Event dataset to only include good times.

    The good times are read from the sweep table ancillary file.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset.

    anc_dependencies : list
        Ancillary files needed for L1C data product creation.

    Returns
    -------
    l1b_de : xarray.Dataset
        Filtered L1B Direct Event dataset.
    """
    # the goodtimes are currently the only ancillary file needed for L1C processing
    goodtimes_table_df = lo_ancillary.read_ancillary_file(
        next(str(s) for s in anc_dependencies if "good-times" in str(s))
    )

    esa_steps = l1b_de["esa_step"].values
    epochs = l1b_de["epoch"].values
    spin_bins = l1b_de["spin_bin"].values

    # Get array of bools for each epoch 1 = good time, 0 not good time
    goodtimes_mask = set_bad_or_goodtimes(
        goodtimes_table_df, epochs, esa_steps, spin_bins
    )

    # Filter the dataset using the mask
    filtered_epochs = l1b_de.sel(epoch=goodtimes_mask.astype(bool))

    return filtered_epochs


def create_pset_counts(
    de: xr.Dataset, filter: FilterType = FilterType.NONE
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
    filter : FilterType, optional
        The event type to include in the counts.
        Can be "triples", "doubles", "h", or "o".

    Returns
    -------
    counts : xarray.DataArray
        The counts for the specified filter.
    """
    filter_options = {
        # triples coincidence types
        FilterType.TRIPLES: ["111111", "111100", "111000"],
        # doubles coincidence types
        FilterType.DOUBLES: [
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
        ],
        # hydrogen species identifier
        FilterType.HYDROGEN: "H",
        # oxygen species identifier
        FilterType.OXYGEN: "O",
    }

    # if the filter string is triples or doubles, filter using the coincidence type
    if filter in {FilterType.TRIPLES, FilterType.DOUBLES}:
        filter_idx = np.where(np.isin(de["coincidence_type"], filter_options[filter]))[
            0
        ]
    # if the filter is h or o, filter using the species
    elif filter in {FilterType.HYDROGEN, FilterType.OXYGEN}:
        filter_idx = np.where(np.isin(de["species"], filter_options[filter]))[0]
    else:
        # if no filter is specified, use all data
        filter_idx = np.arange(len(de["epoch"]))

    # Filter the dataset using the filter index
    de_filtered = de.isel(epoch=filter_idx)

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


def calculate_exposure_times(counts: xr.DataArray, l1b_de: xr.Dataset) -> xr.DataArray:
    """
    Calculate the exposure times for the L1B Direct Event dataset.

    The exposure times are calculated by binning the data into 3600 longitude bins,
    40 latitude bins, and 7 energy bins. If more than one exposure time is in a bin,
    the average is taken.

    Parameters
    ----------
    counts : xarray.DataArray
        An event counts array with dimensions (epoch, lon_bins, lat_bins, energy_bins).
    l1b_de : xarray.Dataset
        L1B Direct Event dataset. This data contains the average spin durations.

    Returns
    -------
    exposure_time : xarray.DataArray
        The exposure times for the L1B Direct Event dataset.
    """
    data = np.column_stack(
        (l1b_de["esa_step"], l1b_de["spin_bin"], l1b_de["off_angle_bin"])
    )

    result = binned_statistic_dd(
        data,
        # exposure time equation from Lo Alg Document 10.1.1.4
        4 * l1b_de["avg_spin_durations"].to_numpy() / 3600,
        statistic="mean",
        # NOTE: The l1b pointing_bin_lon is bin number, not actual angle
        bins=[
            np.arange(N_ESA_ENERGY_STEPS + 1),
            np.arange(N_SPIN_ANGLE_BINS + 1),
            np.arange(N_OFF_ANGLE_BINS + 1),
        ],
    )

    stat = result.statistic[np.newaxis, :, :, :]

    exposure_time = xr.DataArray(
        data=stat.astype(np.float16),
        dims=PSET_DIMS,
    )

    return exposure_time


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
    pointing_start_met: float,
    pointing_end_met: float,
    species: FilterType,
    anc_dependencies: list,
    attr_mgr: ImapCdfAttributes,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Set the background rates for the specified species.

    The background rates are set to a constant value of 0.01 counts/s for all bins.

    Parameters
    ----------
    pointing_start_met : float
        The start MET time of the pointing.
    pointing_end_met : float
        The end MET time of the pointing.
    species : FilterType
        The species to set the background rates for. Can be "h" or "o".
    anc_dependencies : list
        Ancillary files needed for L1C data product creation.
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

    bg_rates = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )
    bg_stat_uncert = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )
    bg_sys_err = np.zeros(
        (N_ESA_ENERGY_STEPS, N_SPIN_ANGLE_BINS, N_OFF_ANGLE_BINS), dtype=np.float16
    )

    # read in the background rates from ancillary file
    if species == FilterType.HYDROGEN:
        background_df = lo_ancillary.read_ancillary_file(
            next(str(s) for s in anc_dependencies if "hydrogen-background" in str(s))
        )
    else:
        background_df = lo_ancillary.read_ancillary_file(
            next(str(s) for s in anc_dependencies if "oxygen-background" in str(s))
        )

    # find to the rows for the current pointing
    pointing_bg_df = background_df[
        (background_df["GoodTime_start"] >= pointing_start_met)
        & (background_df["GoodTime_end"] <= pointing_end_met)
    ]

    # convert the bin start and end resolution from 6 degrees to .1 degrees
    pointing_bg_df["bin_start"] = pointing_bg_df["bin_start"] * 60
    # The last bin end in the file is 0, which means 60 degrees. This is
    # converted to 0.1 degree resolution of 3600
    pointing_bg_df["bin_end"] = pointing_bg_df["bin_end"] * 60
    pointing_bg_df.loc[pointing_bg_df["bin_end"] == 0, "bin_end"] = 3600
    # for each row in the bg ancillary file for this pointing
    for _, row in pointing_bg_df.iterrows():
        bin_start = int(row["bin_start"])
        bin_end = int(row["bin_end"])
        # for each energy step, set the background rate and uncertainty
        for esa_step in range(0, 7):
            value = row[f"E-Step{esa_step + 1}"]
            if row["type"] == "rate":
                bg_rates[esa_step, bin_start:bin_end, :] = value
            elif row["type"] == "sigma":
                bg_sys_err[esa_step, bin_start:bin_end, :] = value
            else:
                raise ValueError("Unknown background type in ancillary file.")
    # set the background rates, uncertainties, and systematic errors
    bg_rates_data = xr.DataArray(
        data=bg_rates,
        dims=["esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(f"{species.value}_background_rates"),
    )
    bg_stat_uncert_data = xr.DataArray(
        data=bg_stat_uncert,
        dims=["esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_stat_uncert"
        ),
    )
    bg_sys_err_data = xr.DataArray(
        data=bg_sys_err,
        dims=["esa_energy_step", "spin_angle", "off_angle"],
        attrs=attr_mgr.get_variable_attributes(
            f"{species.value}_background_rates_sys_err"
        ),
    )

    return bg_rates_data, bg_stat_uncert_data, bg_sys_err_data


def set_pointing_directions(epoch: float) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Set the pointing directions for the given epoch.

    The pointing directions are calculated by transforming Spin and off angles
    to HAE longitude and latitude using SPICE. This returns the HAE longitude and
    latitude as (3600, 40) arrays for each the latitude and longitude.

    Parameters
    ----------
    epoch : float
        The epoch time in TTJ2000ns.

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
    dps_az_el = np.stack([spin, off], axis=-1)

    # Transform from DPS Az/El to HAE lon/lat
    hae_az_el = frame_transform_az_el(
        et, dps_az_el, SpiceFrame.IMAP_DPS, SpiceFrame.IMAP_HAE, degrees=True
    )

    return xr.DataArray(
        data=hae_az_el[:, :, 0].astype(np.float64),
        dims=["spin_angle", "off_angle"],
        # TODO: Add hae_longitude to yaml
        # attrs=attr_mgr.get_variable_attributes(
        #    "hae_longitude"
        # )
    ), xr.DataArray(
        data=hae_az_el[:, :, 1].astype(np.float64),
        dims=["spin_angle", "off_angle"],
        # TODO: Add hae_longitude to yaml
        # attrs=attr_mgr.get_variable_attributes(
        #    "hae_latitude"
        # )
    )
