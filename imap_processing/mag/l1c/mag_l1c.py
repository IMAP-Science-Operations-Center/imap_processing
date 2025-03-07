"""MAG L1C processing module."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import ttj2000ns_to_et


def mag_l1c(
    first_input_dataset: xr.Dataset, second_input_dataset: xr.Dataset, version: str
) -> xr.Dataset:
    """
    Will process MAG L1C data from L1A data.

    This requires both the norm and burst data to be passed in.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset to process. This can be either burst or norm data, for
        mago or magi.
    second_input_dataset : xr.Dataset
        The second input dataset to process. This should be burst if first_input_dataset
        was norm, or norm if first_input_dataset was burst. It should match the
        instrument - both inputs should be mago or magi.
    version : str
        The version of the output data.

    Returns
    -------
    output_dataset : xr.Dataset
        L1C data set.
    """
    # TODO: L1C processing involves filling gaps with burst data.
    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]
    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    input_logical_source_2 = second_input_dataset.attrs["Logical_source"]
    if isinstance(second_input_dataset.attrs["Logical_source"], list):
        input_logical_source_2 = second_input_dataset.attrs["Logical_source"][0]

    if "norm" in input_logical_source_1:
        output_dataset = first_input_dataset.copy()
        logical_source = input_logical_source_1.replace("l1b", "l1c")
    elif "norm" in input_logical_source_2:
        output_dataset = second_input_dataset.copy()
        logical_source = input_logical_source_2.replace("l1b", "l1c")
    else:
        raise RuntimeError("Neither input dataset is norm data")

    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_global_attribute("Data_version", version)

    output_dataset.attrs = attribute_manager.get_global_attributes(logical_source)

    print(first_input_dataset.attrs["Logical_source"])
    # TODO: sort first/second input into norm/burst
    process_mag_l1c(first_input_dataset, second_input_dataset)

    return []


def process_mag_l1c(normal_mode_dataset: xr.Dataset, burst_mode_dataset: xr.Dataset):
    # TODO:
    # - determine expected timeline
    # - copy NM data in
    # - write interpolation step
    # - copy output of interpolation into output
    expected_nm_vectors_per_second = []
    expected_timeline = normal_mode_dataset["epoch"].data
    print(ttj2000ns_to_et(normal_mode_dataset["epoch"].data[0]))
    print(ttj2000ns_to_et(burst_mode_dataset["epoch"].data[0]))
    print(
        f"Difference: {normal_mode_dataset['epoch'].data[0] - burst_mode_dataset['epoch'].data[0]}"
    )

    output_dataset = normal_mode_dataset.copy(deep=True)
    output_dataset["sample_interpolated"] = xr.DataArray(
        np.zeros(len(normal_mode_dataset))
    )

    output_dataset

    return normal_mode_dataset


def generate_timeline(epoch_data: np.ndarray, vectors_per_second_attr: str = None):
    """

    Parameters
    ----------
    epoch_data
    vectors_per_second_attr
        format: {start time}:{vectors per second},{start time}:{vectors per second}

    Returns
    -------

    """
    # given a dataarray of epoch values (from normal mode data) find any gaps of larger than 1 second.
    vectors_per_second = None
    full_timeline = np.zeros(0)
    gaps = np.zeros((0,2))
    if vectors_per_second_attr is not None and vectors_per_second_attr != "":
        vecsec_segments = vectors_per_second_attr.split(",")
        end_index = epoch_data.shape[0]
        for vecsec_segment in reversed(vecsec_segments):
            start_time, vecsec = vecsec_segment.split(":")
            start_index = np.where(int(start_time) == epoch_data)[0][0]
            gaps = np.concatenate((gaps, find_gaps(epoch_data[start_index:end_index], int(vecsec))))
            end_index = start_index
    else:
        # TODO: How to handle this case
        gaps = find_gaps(epoch_data, 2) # Assume half second gaps
        # alternatively, I could try and find the average time between vectors

    # When we have our gaps, generate the full timeline
    last_gap = 0
    for gap in gaps:
        gap_start_index = np.where(epoch_data == gap[0])[0]
        gap_end_index = np.where(epoch_data == gap[1])[0]
        if gap_start_index.size != 1 or gap_end_index.size != 1:
            raise ValueError("Gap start or end not found in input timeline")

        full_timeline = np.concatenate((full_timeline, epoch_data[
                                                       last_gap:gap_start_index[
                                                           0]],
                                        generate_missing_timestamps(gap)))
        last_gap = gap_end_index[0]

    full_timeline = np.concatenate((full_timeline, epoch_data[last_gap:]))

    return full_timeline

def find_gaps(timeline_data: np.ndarray, vectors_per_second: int) -> np.ndarray:
    """
    Find gaps in timeline_data that are larger than 1/vectors_per_second.

    Returns timestamps (start_gap, end_gap) where startgap and endgap both
    exist in timeline data.

    Parameters
    ----------
    timeline_data : numpy.ndarray
        Array of timestamps.
    vectors_per_second : int
        Number of vectors expected per second.
    Returns
    -------
    numpy.ndarray
        Array of timestamps of shape (n, 2) containing n gaps with start_gap and
        end_gap. Start_gap and end_gap both correspond to points in timeline_data.
    """
    # Expected difference between timestamps in nanoseconds.
    expected_gap = 1 / vectors_per_second * 1e9

    diffs = abs(timeline_data[:-1] - np.roll(timeline_data, -1)[:-1])
    gap_index = np.where(diffs != expected_gap)[0]
    output = np.zeros((len(gap_index), 2))

    for index, gap in enumerate(gap_index):
        output[index, :] = [timeline_data[gap], timeline_data[gap + 1]]

    # TODO: How should I handle/find gaps at the end?
    return output


def generate_missing_timestamps(gap: np.ndarray):
    """
    Generate a new timeline from input gaps.

    Any gaps specified in gaps will be filled with timestamps that are 0.5 seconds
    apart.

    Parameters
    ----------
    gap : numpy.ndarray
        Array of timestamps of shape (2,) containing n gaps with start_gap and
        end_gap. Start_gap and end_gap both correspond to points in timeline_data.

    Returns
    -------
    full_timeline: numpy.ndarray
        Completed timeline.

    """
    # Generated timestamps should always be 0.5 seconds apart
    #TODO: is this in the configuration file?
    difference_ns = 0.5 * 1e9

    return np.arange(gap[0], gap[1], difference_ns)
