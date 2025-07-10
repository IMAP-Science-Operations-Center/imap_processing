"""MAG L1C processing module."""

import logging
from typing import Optional

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag import imap_mag_sdc_configuration_v001 as configuration
from imap_processing.mag.constants import ModeFlags, VecSec
from imap_processing.mag.l1c.interpolation_methods import InterpolationFunction

logger = logging.getLogger(__name__)


def mag_l1c(
    first_input_dataset: xr.Dataset,
    second_input_dataset: xr.Dataset = None,
) -> xr.Dataset:
    """
    Will process MAG L1C data from L1A data.

    This requires both the norm and burst data to be passed in.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset to process. This can be either burst or norm data, for
        mago or magi.
    second_input_dataset : xr.Dataset, optional
        The second input dataset to process. This should be burst if first_input_dataset
        was norm, or norm if first_input_dataset was burst. It should match the
        instrument - both inputs should be mago or magi.

    Returns
    -------
    output_dataset : xr.Dataset
        L1C data set.
    """
    # TODO:
    # find missing sequences and output them
    # Fix gaps at the beginning of the day by going to previous day's file
    # Fix gaps at the end of the day
    # Allow for one input to be missing
    # Missing burst file - just pass through norm file
    # Missing norm file - go back to previous L1C file to find timestamps, then
    # interpolate the entire day from burst

    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]
    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    sensor = input_logical_source_1[-1:]
    output_logical_source = f"imap_mag_l1c_norm-mag{sensor}"

    normal_mode_dataset, burst_mode_dataset = select_datasets(
        first_input_dataset, second_input_dataset
    )

    interp_function = InterpolationFunction[configuration.L1C_INTERPOLATION_METHOD]
    if normal_mode_dataset and burst_mode_dataset:
        full_interpolated_timeline = process_mag_l1c(
            normal_mode_dataset, burst_mode_dataset, interp_function
        )
    elif normal_mode_dataset is not None:
        full_interpolated_timeline = fill_normal_data(
            normal_mode_dataset, normal_mode_dataset["epoch"].data
        )
    else:
        # TODO: With only burst data, downsample by retrieving the timeline
        raise NotImplementedError

    completed_timeline = remove_missing_data(full_interpolated_timeline)

    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_instrument_variable_attrs("mag", "l1c")
    compression = xr.DataArray(
        np.arange(2),
        name="compression",
        dims=["compression"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_attrs", check_schema=False
        ),
    )

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_attrs", check_schema=False
        ),
    )

    epoch_time = xr.DataArray(
        completed_timeline[:, 0],
        name="epoch",
        dims=["epoch"],
        attrs=attribute_manager.get_variable_attributes("epoch"),
    )

    direction_label = xr.DataArray(
        direction.values.astype(str),
        name="direction_label",
        dims=["direction_label"],
        attrs=attribute_manager.get_variable_attributes(
            "direction_label", check_schema=False
        ),
    )

    compression_label = xr.DataArray(
        compression.values.astype(str),
        name="compression_label",
        dims=["compression_label"],
        attrs=attribute_manager.get_variable_attributes(
            "compression_label", check_schema=False
        ),
    )
    global_attributes = attribute_manager.get_global_attributes(output_logical_source)
    # TODO merge missing sequences? replace?
    global_attributes["missing_sequences"] = ""

    try:
        global_attributes["is_mago"] = normal_mode_dataset.attrs["is_mago"]
        global_attributes["is_active"] = normal_mode_dataset.attrs["is_active"]
        global_attributes["missing_sequences"] = normal_mode_dataset.attrs[
            "missing_sequences"
        ]
    except KeyError as e:
        logger.info(
            f"Key error when assigning global attributes, attribute not found in "
            f"L1B file with logical source "
            f"{normal_mode_dataset.attrs['Logical_source']}: {e}"
        )

    global_attributes["interpolation_method"] = interp_function.name

    output_dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "direction": direction,
            "direction_label": direction_label,
            "compression": compression,
            "compression_label": compression_label,
        },
        attrs=global_attributes,
    )

    output_dataset["vectors"] = xr.DataArray(
        completed_timeline[:, 1:5],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=attribute_manager.get_variable_attributes("vector_attrs"),
    )

    output_dataset["vector_magnitude"] = xr.apply_ufunc(
        lambda x: np.linalg.norm(x[:4]),
        output_dataset["vectors"],
        input_core_dims=[["direction"]],
        output_core_dims=[[]],
        vectorize=True,
    )
    # output_dataset['vector_magnitude'].attrs =
    # attribute_manager.get_variable_attributes("vector_magnitude_attrs")

    output_dataset["compression_flags"] = xr.DataArray(
        completed_timeline[:, 6:8],
        name="compression_flags",
        dims=["epoch", "compression"],
        attrs=attribute_manager.get_variable_attributes("compression_flags_attrs"),
    )

    output_dataset["generated_flag"] = xr.DataArray(
        completed_timeline[:, 5],
        name="generated_flag",
        dims=["epoch"],
        # attrs=attribute_manager.get_variable_attributes("generated_flag_attrs"),
    )

    return output_dataset


def select_datasets(
    first_input_dataset: xr.Dataset, second_input_dataset: Optional[xr.Dataset] = None
) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Given one or two datasets, assign one to norm and one to burst.

    If only one dataset is provided, the other will be marked as None. If two are
    provided, they will be validated to ensure one is norm and one is burst.

    Parameters
    ----------
    first_input_dataset : xr.Dataset
        The first input dataset.
    second_input_dataset : xr.Dataset, optional
        The second input dataset.

    Returns
    -------
    tuple
        Tuple containing norm_mode_dataset, burst_mode_dataset.
    """
    normal_mode_dataset = None
    burst_mode_dataset = None

    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]

    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    if "norm" in input_logical_source_1:
        normal_mode_dataset = first_input_dataset

    if "burst" in input_logical_source_1:
        burst_mode_dataset = first_input_dataset

    if second_input_dataset is None:
        logger.info(
            f"Only one input dataset provided with logical source "
            f"{input_logical_source_1}"
        )
    else:
        input_logical_source_2 = second_input_dataset.attrs["Logical_source"]
        if isinstance(second_input_dataset.attrs["Logical_source"], list):
            input_logical_source_2 = second_input_dataset.attrs["Logical_source"][0]

        if "burst" in input_logical_source_2:
            burst_mode_dataset = second_input_dataset

        elif "norm" in input_logical_source_2:
            normal_mode_dataset = second_input_dataset

        # If there are two inputs, one should be norm and one should be burst
        if normal_mode_dataset is None or burst_mode_dataset is None:
            raise RuntimeError(
                "L1C requires one normal mode and one burst mode input file."
            )

    return normal_mode_dataset, burst_mode_dataset


def process_mag_l1c(
    normal_mode_dataset: xr.Dataset,
    burst_mode_dataset: xr.Dataset,
    interpolation_function: InterpolationFunction,
) -> np.ndarray:
    """
    Create MAG L1C data from L1B datasets.

    This function starts from the normal mode dataset and completes the following steps:
    1. find all the gaps in the dataset
    2. generate a new timeline with the gaps filled
    3. fill the timeline with normal mode data (so, all the non-gap timestamps)
    4. interpolate the gaps using the burst mode data and the method specified in
        interpolation_function.

    It returns an (n, 8) shaped array:
    0 - epoch (timestamp)
    1-4 - vector x, y, z, and range
    5 - generated flag (0 for normal data, 1 for interpolated data, -1 for missing data)
    6-7 - compression flags (is_compressed, compression_width)

    Parameters
    ----------
    normal_mode_dataset : xarray.Dataset
        The normal mode dataset, which acts as a base for the output.
    burst_mode_dataset : xarray.Dataset
        The burst mode dataset, which is used to fill in the gaps in the normal mode.
    interpolation_function : InterpolationFunction
        The interpolation function to use to fill in the gaps.

    Returns
    -------
    np.ndarray
        An (n, 8) shaped array containing the completed timeline.
    """
    norm_epoch = normal_mode_dataset["epoch"].data
    if "vectors_per_second" in normal_mode_dataset.attrs:
        normal_vecsec_dict = vectors_per_second_from_string(
            normal_mode_dataset.attrs["vectors_per_second"]
        )
    else:
        normal_vecsec_dict = None

    output_dataset = normal_mode_dataset.copy(deep=True)
    output_dataset["sample_interpolated"] = xr.DataArray(
        np.zeros(len(normal_mode_dataset))
    )

    gaps = find_all_gaps(norm_epoch, normal_vecsec_dict)

    new_timeline = generate_timeline(norm_epoch, gaps)
    norm_filled = fill_normal_data(normal_mode_dataset, new_timeline)
    interpolated = interpolate_gaps(
        burst_mode_dataset, gaps, norm_filled, interpolation_function
    )

    return interpolated


def fill_normal_data(
    normal_dataset: xr.Dataset, new_timeline: np.ndarray
) -> np.ndarray:
    """
    Fill the new timeline with the normal mode data.

    If the timestamp exists in the normal mode data, it will be filled in the output.

    Parameters
    ----------
    normal_dataset : xr.Dataset
        The normal mode dataset.
    new_timeline : np.ndarray
        A 1D array of timestamps to fill.

    Returns
    -------
    np.ndarray
        An (n, 8) shaped array containing the timeline filled with normal mode data.
        Gaps are marked as -1 in the generated flag column at index 5.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.
    """
    # TODO: fill with FILLVAL?
    filled_timeline: np.ndarray = np.zeros((len(new_timeline), 8))
    filled_timeline[:, 0] = new_timeline
    # Flags, will also indicate any missed timestamps
    filled_timeline[:, 5] = ModeFlags.MISSING.value

    for index, timestamp in enumerate(normal_dataset["epoch"].data):
        timeline_index = np.searchsorted(new_timeline, timestamp)
        filled_timeline[timeline_index, 1:5] = normal_dataset["vectors"].data[index]
        filled_timeline[timeline_index, 5] = ModeFlags.NORM.value
        filled_timeline[timeline_index, 6:8] = normal_dataset["compression_flags"].data[
            index
        ]

    return filled_timeline


def interpolate_gaps(
    burst_dataset: xr.Dataset,
    gaps: np.ndarray,
    filled_norm_timeline: np.ndarray,
    interpolation_function: InterpolationFunction,
) -> np.ndarray:
    """
    Interpolate the gaps in the filled timeline using the burst mode data.

    Returns an array that matches the format of filled_norm_timeline, with gaps filled
    using interpolated burst data.

    Parameters
    ----------
    burst_dataset : xarray.Dataset
        The L1B burst mode dataset.
    gaps : numpy.ndarray
        An array of gaps to fill, with shape (n, 2) where n is the number of gaps.
    filled_norm_timeline : numpy.ndarray
        Timeline filled with normal mode data in the shape (n, 8).
    interpolation_function : InterpolationFunction
        The interpolation function to use to fill in the gaps.

    Returns
    -------
    numpy.ndarray
        An array of shape (n, 8) containing the fully filled timeline.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.
    """
    burst_epochs = burst_dataset["epoch"].data
    # Exclude range values
    burst_vectors = burst_dataset["vectors"].data
    # Default to two vectors per second
    burst_vecsec_dict = {0: VecSec.TWO_VECS_PER_S.value}
    if "vectors_per_second" in burst_dataset.attrs:
        burst_vecsec_dict = vectors_per_second_from_string(
            burst_dataset.attrs["vectors_per_second"]
        )

    for gap in gaps:
        # TODO: we might need a few inputs before or after start/end
        burst_gap_start = (np.abs(burst_epochs - gap[0])).argmin()
        burst_gap_end = (np.abs(burst_epochs - gap[1])).argmin()

        # for the CIC filter, we need 2x normal mode cadence seconds

        norm_rate = VecSec(int(gap[2]))

        # Input rate
        # Find where burst_start is after the start of the timeline
        burst_vecsec_index = (
            np.searchsorted(
                list(burst_vecsec_dict.keys()),
                burst_epochs[burst_gap_start],
                side="right",
            )
            - 1
        )
        burst_rate = VecSec(list(burst_vecsec_dict.values())[burst_vecsec_index])

        required_seconds = (1 / norm_rate.value) * 2
        burst_buffer = int(required_seconds * burst_rate.value)

        burst_start = max(0, burst_gap_start - burst_buffer)
        burst_end = min(len(burst_epochs) - 1, burst_gap_end + burst_buffer)

        gap_timeline = filled_norm_timeline[
            (filled_norm_timeline > gap[0]) & (filled_norm_timeline < gap[1])
        ]
        logger.info(
            f"difference between gap start and burst start: "
            f"{gap_timeline[0] - burst_epochs[burst_start]}"
        )

        short = (gap_timeline >= burst_epochs[burst_start]) & (
            gap_timeline <= burst_epochs[burst_gap_end]
        )
        if len(gap_timeline) != (short).sum():
            print(f"Chopping timeline from {len(gap_timeline)} to {short.sum()}")

        # Limit timestamps to only include the areas with burst data
        gap_timeline = gap_timeline[
            (
                (gap_timeline >= burst_epochs[burst_start])
                & (gap_timeline <= burst_epochs[burst_gap_end])
            )
        ]
        # do not include range
        gap_fill = interpolation_function(
            burst_vectors[burst_start:burst_end, :3],
            burst_epochs[burst_start:burst_end],
            gap_timeline,
            input_rate=burst_rate,
            output_rate=norm_rate,
        )

        # gaps should not have data in timeline, still check it
        for index, timestamp in enumerate(gap_timeline):
            timeline_index = np.searchsorted(filled_norm_timeline[:, 0], timestamp)
            if sum(
                filled_norm_timeline[timeline_index, 1:4]
            ) == 0 and burst_gap_start + index < len(burst_vectors):
                filled_norm_timeline[timeline_index, 1:4] = gap_fill[index]

                filled_norm_timeline[timeline_index, 4] = burst_vectors[
                    burst_gap_start + index, 3
                ]
                filled_norm_timeline[timeline_index, 5] = ModeFlags.BURST.value
                filled_norm_timeline[timeline_index, 6:8] = burst_dataset[
                    "compression_flags"
                ].data[burst_gap_start + index]

    return filled_norm_timeline


def generate_timeline(epoch_data: np.ndarray, gaps: np.ndarray) -> np.ndarray:
    """
    Generate a new timeline from existing, gap-filled timeline and gaps.

    The gaps are generated at a .5 second cadence, regardless of the cadence of the
    existing data.

    Parameters
    ----------
    epoch_data : numpy.ndarray
        The existing timeline data, in the shape (n,).
    gaps : numpy.ndarray
        An array of gaps to fill, with shape (n, 2) where n is the number of gaps.
        The gap is specified as (start, end) where start and end both exist in the
        timeline already.

    Returns
    -------
    numpy.ndarray
        The new timeline, filled with the existing data and the generated gaps.
    """
    full_timeline: np.ndarray = np.zeros(0)

    # When we have our gaps, generate the full timeline
    last_gap = 0
    for gap in gaps:
        gap_start_index = np.where(epoch_data == gap[0])[0]
        gap_end_index = np.where(epoch_data == gap[1])[0]
        if gap_start_index.size != 1 or gap_end_index.size != 1:
            raise ValueError("Gap start or end not found in input timeline")

        full_timeline = np.concatenate(
            (
                full_timeline,
                epoch_data[last_gap : gap_start_index[0]],
                generate_missing_timestamps(gap),
            )
        )
        last_gap = gap_end_index[0]

    full_timeline = np.concatenate((full_timeline, epoch_data[last_gap:]))

    return full_timeline


def find_all_gaps(
    epoch_data: np.ndarray, vecsec_dict: Optional[dict] = None
) -> np.ndarray:
    """
    Find all the gaps in the epoch data.

    If vectors_per_second_attr is provided, it will be used to find the gaps. Otherwise,
    it will assume a nominal 1/2 second gap. A gap is defined as missing data from the
    expected sequence as defined by vectors_per_second_attr.

    Parameters
    ----------
    epoch_data : numpy.ndarray
        The epoch data to find gaps in.
    vecsec_dict : dict, optional
        A dictionary of the form {start: vecsec, start: vecsec} where start is the time
        in nanoseconds and vecsec is the number of vectors per second. This will be
        used to find the gaps. If not provided, a 1/2 second gap is assumed.

    Returns
    -------
    numpy.ndarray
        An array of gaps with shape (n, 3) where n is the number of gaps. The gaps are
        specified as (start, end, vector_rate) where start and end both exist in the
        timeline.
    """
    gaps: np.ndarray = np.zeros((0, 3))
    if vecsec_dict is None:
        # TODO: when we go back to the previous file, also retrieve expected
        #  vectors per second
        # If no vecsec is provided, assume 2 vectors per second
        vecsec_dict = {0: VecSec.TWO_VECS_PER_S.value}

    end_index = epoch_data.shape[0]
    for start_time in reversed(sorted(vecsec_dict.keys())):
        start_index = np.where(start_time == epoch_data)[0][0]
        gaps = np.concatenate(
            (
                find_gaps(
                    epoch_data[start_index : end_index + 1], vecsec_dict[start_time]
                ),
                gaps,
            )
        )
        end_index = start_index

    return gaps


def find_gaps(timeline_data: np.ndarray, vectors_per_second: int) -> np.ndarray:
    """
    Find gaps in timeline_data that are larger than 1/vectors_per_second.

    Returns timestamps (start_gap, end_gap, vectors_per_second) where startgap and
    endgap both exist in timeline data.

    Parameters
    ----------
    timeline_data : numpy.ndarray
        Array of timestamps.
    vectors_per_second : int
        Number of vectors expected per second.

    Returns
    -------
    numpy.ndarray
        Array of timestamps of shape (n, 3) containing n gaps with start_gap and
        end_gap, as well as vectors_per_second. Start_gap and end_gap both correspond
        to points in timeline_data.
    """
    # Expected difference between timestamps in nanoseconds.
    expected_gap = 1 / vectors_per_second * 1e9

    # TODO: timestamps can vary by a few ms. Per Alastair, this can be around 7.5% of
    #  cadence without counting as a "gap".
    diffs = abs(np.diff(timeline_data))
    # 3.5e7 == 7.5% of 0.5s in nanoseconds, a common gap. In the future, this number
    # will be calculated from the expected gap.
    gap_index = np.asarray(diffs - expected_gap > expected_gap * 0.075).nonzero()[0]
    output: np.ndarray = np.zeros((len(gap_index), 3))

    for index, gap in enumerate(gap_index):
        output[index, :] = [
            timeline_data[gap],
            timeline_data[gap + 1],
            vectors_per_second,
        ]

    # TODO: How should I handle/find gaps at the end?
    return output


def generate_missing_timestamps(gap: np.ndarray) -> np.ndarray:
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
    # TODO: is this in the configuration file?
    difference_ns = 0.5 * 1e9

    output: np.ndarray = np.arange(gap[0], gap[1], difference_ns)
    return output


def vectors_per_second_from_string(vecsec_string: str) -> dict:
    """
    Extract the vectors per second from a string into a dictionary.

    Dictionary format: {start_time: vecsec, start_time: vecsec}.

    Parameters
    ----------
    vecsec_string : str
        A string of the form "start:vecsec,start:vecsec" where start is the time in
        nanoseconds and vecsec is the number of vectors per second.

    Returns
    -------
    dict
        A dictionary of the form {start_time: vecsec, start_time: vecsec}.
    """
    vecsec_dict = {}
    vecsec_segments = vecsec_string.split(",")
    for vecsec_segment in vecsec_segments:
        start_time, vecsec = vecsec_segment.split(":")
        vecsec_dict[int(start_time)] = int(vecsec)

    return vecsec_dict


def remove_missing_data(filled_timeline: np.ndarray) -> np.ndarray:
    """
    Remove timestamps with no data from the filled timeline.

    Anywhere that the generated flag is equal to -1, the data will be removed.

    Parameters
    ----------
    filled_timeline : numpy.ndarray
        An (n, 8) shaped array containing the filled timeline.
        Indices: 0 - epoch, 1-4 - vector x, y, z, and range, 5 - generated flag,
        6-7 - compression flags.

    Returns
    -------
    cleaned_array : numpy.ndarray
        The filled timeline with missing data removed.
    """
    cleaned_array: np.ndarray = filled_timeline[filled_timeline[:, 5] != -1]
    return cleaned_array
