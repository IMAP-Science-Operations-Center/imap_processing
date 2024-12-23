"""Module for GLOWS Level 2 processing."""

import dataclasses

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.glows.l1b.glows_l1b_data import HistogramL1B
from imap_processing.glows.l2.glows_l2_data import DailyLightcurve, HistogramL2


def glows_l2(input_dataset: xr.Dataset, data_version: str) -> list[xr.Dataset]:
    """
    Will process GLoWS L2 data from L1 data.

    Parameters
    ----------
    input_dataset : xarray.Dataset
        Input L1B dataset.
    data_version : str
        Version for output.

    Returns
    -------
    xarray.Dataset
     Glows L2 Dataset.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("glows")
    cdf_attrs.add_instrument_variable_attrs("glows", "l2")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    split_data = split_data_by_observational_day(input_dataset)
    l2_output = []
    for data in split_data:
        l2 = generate_l2(data)
        l2_output.append(create_l2_dataset(l2, cdf_attrs))

    return l2_output


# TODO: filter good times out
def generate_l2(l1b_dataset: xr.Dataset) -> HistogramL2:
    """
    Generate L2 data from L1B data.

    Returns L2 data in the form of a HistogramL2 dataclass.

    Parameters
    ----------
    l1b_dataset : xarray.Dataset
        Input L1B dataset.

    Returns
    -------
    HistogramL2
        L2 data in the form of a HistogramL2 dataclass.
    """
    # most of the values from L1B are averaged over a day

    good_data = l1b_dataset.isel(
        epoch=return_good_times(l1b_dataset["flags"], np.ones((17,)))
    )
    # todo: bad angle filter
    # TODO: if there are no good times, assign -1 and 0 to outputs
    # TODO filter bad bins out. Needs to happen here while everything is still
    # per-timestamp.

    # one dataset collects multiple epoch values which need to be averaged down into
    # one value.
    all_variables = dataclasses.fields(HistogramL1B)

    daily_lightcurve = DailyLightcurve(good_data)

    # Generate outputs that are passed in directly from L1B
    var_outputs = {
        "total_l1b_inputs": len(good_data["epoch"]),
        "number_of_good_l1b_inputs": len(good_data["epoch"]),
        # TODO replace post-filter
        "identifier": 100,  # TODO: retrieve from spin table
        "start_time": good_data["epoch"].data[0],
        "end_time": good_data["epoch"].data[-1],
        # TODO fill this in
        "bad_time_flag_occurrences": None,
        # Accumulate all the histograms from good times from the day into one
        "daily_lightcurve": daily_lightcurve,
    }

    for field in all_variables:
        var_name = field.name
        if "average" in var_name:
            # This results in a scalar value, so `keepdims=True` ensures we keep the
            # epoch dimension.
            var_outputs[var_name] = (
                l1b_dataset[var_name].mean(dim="epoch", keepdims=True).data
            )

            var_outputs[var_name.replace("average", "std_dev")] = (
                l1b_dataset[var_name].std(dim="epoch", keepdims=True).data
            )

    # l1b stuff is done
    output = HistogramL2(**var_outputs)

    return output


def filter_bad_bins(histograms: NDArray, bin_exclusions: NDArray) -> NDArray:
    """
    Filter out bad bins from the histogram.

    Parameters
    ----------
    histograms : numpy.ndarray
        Histogram data, with shape (n_timestamps, n_bins).
    bin_exclusions : numpy.ndarray
        Array of bin exclusions. This 2d array has a timestamp and bin filter array
        pair. The bin filter array indicates "1" if a bin is to be excluded.

    Returns
    -------
    numpy.ndarray
        Histogram data with bad bins marked with -1.
    """
    # TODO: will need ancillary file imap_glows_exclusions_by_instr_team
    # TODO: complete once unique_block_identifier is implemented
    # file contains timestamp & bin filter array pairs. For the timestamp, the
    # filter should be applied such that 1 excludes the bin.

    # excluded bins can be marked with -1
    return histograms


def split_data_by_observational_day(input_dataset: xr.Dataset) -> list[xr.Dataset]:
    """
    Return L1B data array for an observational day, given start and stop times.

    Parameters
    ----------
    input_dataset : xarray.Dataset
        Input L1B dataset.

    Returns
    -------
    list : xarray.Dataset
        List of L1B datasets, each representing a day of data.
    """
    # TODO: replace this with a query to the spin table to get the observational days

    # Find the range of epoch values within the observational day.
    # This should be replaced with a query to the spin table to get the observational
    # days within the time range of the file and when those observational days
    # start and stop

    day_ends = [
        input_dataset["epoch"].data[0],
        input_dataset["epoch"].data[100],
        input_dataset["epoch"].data[-1],
    ]
    print(day_ends)

    # TODO: this slice is inclusive on the start and end for some reason. When you
    #  replace this slice with the real spin data, figure that out.
    data_by_day = [
        input_dataset.sel(epoch=slice(day_ends[i], day_ends[i + 1]))
        for i in range(len(day_ends) - 1)
    ]
    return data_by_day


def create_l2_dataset(
    histogram_l2: HistogramL2, attrs: ImapCdfAttributes
) -> xr.Dataset:
    """
    Create a xarray dataset from a HistogramL2 dataclass.

    This dataset should include all the CDF attributes.

    Parameters
    ----------
    histogram_l2 : HistogramL2
        L2 data.
    attrs : ImapCdfAttributes
        CDF attributes for GLOWS L2.

    Returns
    -------
    xarray.Dataset
        L2 dataset for output to CDF file.
    """
    # Each L2 file only has one timestamp.
    # TODO: If we want this to point to the start time, we need to set the attribute
    #  variable BIN_LOCATION to 0. Otherwise, we need this to be halfway between start
    #  time and end time.
    time_data = np.array([histogram_l2.start_time], dtype=np.float64)
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=attrs.get_variable_attributes("epoch"),
    )

    bins = xr.DataArray(
        np.arange(histogram_l2.daily_lightcurve.number_of_bins),
        name="bins",
        dims=["bins"],
        attrs=attrs.get_variable_attributes("test"),
    )

    eclipic_data = xr.DataArray(
        np.arange(3),
        name="ecliptic",
        dims=["ecliptic"],
        attrs=attrs.get_variable_attributes("test"),
    )

    output = xr.Dataset(
        coords={"epoch": epoch_time, "bins": bins, "ecliptic": eclipic_data},
        attrs=attrs.get_global_attributes("imap_glows_l2_hist"),
    )

    ecliptic_variables = [
        "spacecraft_location_average",
        "spacecraft_location_std_dev",
        "spacecraft_velocity_average",
        "spacecraft_velocity_std_dev",
    ]
    for key, value in dataclasses.asdict(histogram_l2).items():
        if key in ecliptic_variables:
            output[key] = xr.DataArray(
                value,
                dims=["epoch", "ecliptic"],
                attrs=attrs.get_variable_attributes("test"),
            )

        elif key != "daily_lightcurve":
            if type(value) != np.ndarray:
                value = np.array([value])
            output[key] = xr.DataArray(
                value,
                dims=["epoch"],
                attrs=attrs.get_variable_attributes("test"),
            )

    for key, value in dataclasses.asdict(histogram_l2.daily_lightcurve).items():
        value = np.array([value])
        if key == "number_of_bins":
            # number_of_bins does not have n_bins dimensions.
            output[key] = xr.DataArray(
                value,
                dims=["epoch"],
                attrs=attrs.get_variable_attributes("test"),
            )
        else:
            output[key] = xr.DataArray(
                value,
                dims=["epoch", "bins"],
                attrs=attrs.get_variable_attributes("test"),
            )

    return output


def return_good_times(flags: xr.DataArray, active_flags: NDArray) -> NDArray:
    """
    Return the good times based on the input flags.

    Parameters
    ----------
    flags : xarray.DataArray
        Flags dataset with shape (n_timestamps, n_flags). If a flag is active and set
        to 1, the timestamp is considered good.

    active_flags : numpy.ndarray
        Array of active flags. If the flag is set to 1, it is considered active.

    Returns
    -------
    numpy.ndarray
        An array of indices for good times.
    """
    if len(active_flags) != flags.shape[1]:
        print("Active flags don't matched expected length")

    # A good time is where all the active flags are equal to one.
    # Here, we mask the active indices using active_flags, and then return the times
    # where all the active indices == 1.
    good_times = np.where(np.all(flags[:, active_flags == 1] == 1, axis=1))[0]
    return good_times
