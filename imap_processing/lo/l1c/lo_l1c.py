"""IMAP-Lo L1C Data Processing."""

from dataclasses import Field
from enum import Enum

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import binned_statistic_dd

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns


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
        pset = initialize_pset(l1b_goodtimes_only, attr_mgr, logical_source)
        full_counts = create_pset_counts(l1b_goodtimes_only)
        pset["triples_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.TRIPLES
        )
        pset["doubles_counts"] = create_pset_counts(
            l1b_goodtimes_only, FilterType.DOUBLES
        )
        pset["h_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.HYDROGEN)
        pset["o_counts"] = create_pset_counts(l1b_goodtimes_only, FilterType.OXYGEN)
        pset["exposure_time"] = calculate_exposure_times(
            full_counts, l1b_goodtimes_only
        )
    pset.attrs = attr_mgr.get_global_attributes(logical_source)
    # TODO: Temp fix before adding attribute variables.
    #  CDF won't open if DEPEND_0 is not deleted currently.
    del pset["epoch"].attrs["DEPEND_0"]

    pset = pset.assign_coords(
        {
            "energy": np.arange(1, 8),
            "longitude": np.arange(3600),
            "latitude": np.arange(40),
        }
    )

    return [pset]


def initialize_pset(
    l1b_de: xr.Dataset, attr_mgr: ImapCdfAttributes, logical_source: str
) -> xr.Dataset:
    """
    Initialize the PSET dataset and set the Epoch.

    The Epoch time is set to the first of the L1B
    Direct Event times. There is one Epoch per PSET file.

    Parameters
    ----------
    l1b_de : xarray.Dataset
        L1B Direct Event dataset.
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the L1C attributes.
    logical_source : str
        The logical source of the pset.

    Returns
    -------
    pset : xarray.Dataset
        Initialized PSET dataset.
    """
    pset = xr.Dataset(
        attrs=attr_mgr.get_global_attributes(logical_source),
    )
    # TODO: Need to create utility to get start of repointing to use
    #  for the pset epoch time. Setting to first DE for now
    pset_epoch = l1b_de["epoch"][0].item()
    pset["epoch"] = xr.DataArray(
        np.array([pset_epoch]),
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    return pset


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
    goodtimes_table_df = pd.read_csv(anc_dependencies[0])

    # convert goodtimes from MET to TTJ2000
    goodtimes_start = met_to_ttj2000ns(goodtimes_table_df["GoodTime_strt"])
    goodtimes_end = met_to_ttj2000ns(goodtimes_table_df["GoodTime_end"])

    # Create a mask for epochs within any of the start/end time ranges
    goodtimes_mask = np.zeros_like(l1b_de["epoch"], dtype=bool)

    # Iterate over the good times and create a mask
    for start, end in zip(goodtimes_start, goodtimes_end):
        goodtimes_mask |= (l1b_de["epoch"] >= start) & (l1b_de["epoch"] < end)

    # Filter the dataset using the mask
    filtered_epochs = l1b_de.sel(epoch=goodtimes_mask)

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
        FilterType.HYDROGEN: "h",
        # oxygen species identifier
        FilterType.OXYGEN: "o",
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
            de_filtered["pointing_bin_lon"],
            de_filtered["pointing_bin_lat"],
        )
    )
    # Create the histogram with 3600 longitude bins, 40 latitude bins, and 7 energy bins
    lon_edges = np.arange(3601)
    lat_edges = np.arange(41)
    energy_edges = np.arange(8)

    hist, edges = np.histogramdd(
        data,
        bins=[energy_edges, lon_edges, lat_edges],
    )

    # add a new axis of size 1 for the epoch
    hist = hist[np.newaxis, :, :, :]

    counts = xr.DataArray(
        data=hist.astype(np.int16),
        dims=["epoch", "energy", "longitude", "latitude"],
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
    # Create bin edges
    lon_edges = np.arange(3601)
    lat_edges = np.arange(41)
    energy_edges = np.arange(8)

    data = np.column_stack(
        (l1b_de["esa_step"], l1b_de["pointing_bin_lon"], l1b_de["pointing_bin_lat"])
    )

    result = binned_statistic_dd(
        data,
        # exposure time equation from Lo Alg Document 10.1.1.4
        4 * l1b_de["avg_spin_durations"].to_numpy() / 3600,
        statistic="mean",
        bins=[energy_edges, lon_edges, lat_edges],
    )

    stat = result.statistic[np.newaxis, :, :, :]

    exposure_time = xr.DataArray(
        data=stat.astype(np.float16),
        dims=["epoch", "energy", "longitude", "latitude"],
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

    # Create a data array for the epoch time
    # TODO: might need to update the attrs to use new YAML file
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1c_pset":
        esa_step = xr.DataArray(
            data=[1, 2, 3, 4, 5, 6, 7],
            name="esa_step",
            dims=["esa_step"],
            attrs=attr_mgr.get_variable_attributes("esa_step"),
        )
        pointing_bins = xr.DataArray(
            data=np.arange(3600),
            name="pointing_bins",
            dims=["pointing_bins"],
            attrs=attr_mgr.get_variable_attributes("pointing_bins"),
        )

        esa_step_label = xr.DataArray(
            esa_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )
        pointing_bins_label = xr.DataArray(
            pointing_bins.values.astype(str),
            name="pointing_bins_label",
            dims=["pointing_bins_label"],
            attrs=attr_mgr.get_variable_attributes("pointing_bins_label"),
        )
        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "pointing_bins": pointing_bins,
                "pointing_bins_label": pointing_bins_label,
                "esa_step": esa_step,
                "esa_step_label": esa_step_label,
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
        if field in ["pointing_start", "pointing_end", "mode", "pivot_angle"]:
            dataset[field] = xr.DataArray(
                data=[1],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        #  The data type will be set in the data class when that's created
        elif field == "exposure_time":
            dataset[field] = xr.DataArray(
                data=np.ones((1, 7), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

        elif "rate" in field:
            dataset[field] = xr.DataArray(
                data=np.ones((1, 3600, 7), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                data=np.ones((1, 3600, 7), dtype=np.int16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

    return dataset
