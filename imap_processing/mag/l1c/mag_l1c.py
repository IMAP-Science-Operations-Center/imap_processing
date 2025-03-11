"""MAG L1C processing module."""
import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.l1c.interpolation_methods import InterpolationFunction

logger = logging.getLogger(__name__)

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
    # TODO:
    # find missing sequences and output them
    # calculate magnitude
    # add missing interpolation methods

    input_logical_source_1 = first_input_dataset.attrs["Logical_source"]
    if isinstance(first_input_dataset.attrs["Logical_source"], list):
        input_logical_source_1 = first_input_dataset.attrs["Logical_source"][0]

    input_logical_source_2 = second_input_dataset.attrs["Logical_source"]
    if isinstance(second_input_dataset.attrs["Logical_source"], list):
        input_logical_source_2 = second_input_dataset.attrs["Logical_source"][0]

    normal_mode_dataset = None
    burst_mode_dataset = None

    if "norm" in input_logical_source_1 and "burst" in input_logical_source_2:
        normal_mode_dataset = first_input_dataset
        burst_mode_dataset = second_input_dataset
    elif "norm" in input_logical_source_2 and "burst" in input_logical_source_1:
        normal_mode_dataset = second_input_dataset
        burst_mode_dataset = first_input_dataset
    else:
        raise RuntimeError(
            "L1C requires one normal mode and one burst mode input " "file."
        )

    with open(
        Path(__file__).parent.parent / "imap_mag_sdc-configuration_v001.yaml"
    ) as f:
        configuration = yaml.safe_load(f)

    interp_function = InterpolationFunction[configuration["L1C_interpolation_method"]]
    completed_timeline = process_mag_l1c(normal_mode_dataset, burst_mode_dataset, interp_function)

    attribute_manager = ImapCdfAttributes()
    attribute_manager.add_instrument_global_attrs("mag")
    attribute_manager.add_global_attribute("Data_version", version)
    # TODO: pull from L1C instead
    attribute_manager.add_instrument_variable_attrs("mag", "l1b")

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

    # TODO: update with L1C specific attributes
    global_attributes = attribute_manager.get_global_attributes(input_logical_source_1)
    try:
        global_attributes["is_mago"] = normal_mode_dataset.attrs["is_mago"]
        global_attributes["is_active"] = normal_mode_dataset.attrs["is_active"]
        global_attributes["missing_sequences"] = normal_mode_dataset.attrs[
            "missing_sequences"
        ]
    except KeyError as e:
        logger.info(
            f"Key error when assigning global attributes, attribute not found in "
            f"L1A file: {e}"
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

    output_dataset['vectors'] = xr.DataArray(
        completed_timeline[:, 1:5],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=attribute_manager.get_variable_attributes("vector_attrs"),
    )

    output_dataset['vector_magnitude'] = xr.apply_ufunc(
        lambda x: np.linalg.norm(x[:4]), output_dataset['vectors'], input_core_dims=[["direction"]], output_core_dims=[[]],
        vectorize=True)
    # output_dataset['vector_magnitude'].attrs = attribute_manager.get_variable_attributes("vector_magnitude_attrs")

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


def process_mag_l1c(
    normal_mode_dataset: xr.Dataset,
    burst_mode_dataset: xr.Dataset,
    interpolation_function: InterpolationFunction,
):
    norm_epoch = normal_mode_dataset["epoch"].data
    vecsec_attr = normal_mode_dataset.attrs["vectors_per_second"]

    output_dataset = normal_mode_dataset.copy(deep=True)
    output_dataset["sample_interpolated"] = xr.DataArray(
        np.zeros(len(normal_mode_dataset))
    )

    gaps = find_all_gaps(norm_epoch, vecsec_attr)

    new_timeline = generate_timeline(norm_epoch, gaps)
    norm_filled = fill_normal_data(normal_mode_dataset, new_timeline)
    interpolated = interpolate_gaps(
        burst_mode_dataset, gaps, norm_filled, interpolation_function
    )

    return interpolated


def fill_normal_data(normal_dataset, new_timeline):
    # TODO: fill with FILLVAL?
    filled_timeline = np.zeros((len(new_timeline), 8))
    filled_timeline[:, 0] = new_timeline
    # Flags, will also indicate any missed timestamps
    filled_timeline[:, 5] = -1

    for index, timestamp in enumerate(normal_dataset["epoch"].data):
        timeline_index = np.searchsorted(new_timeline, timestamp)
        filled_timeline[timeline_index, 1:5] = normal_dataset["vectors"].data[index]
        filled_timeline[timeline_index, 5] = 0
        filled_timeline[timeline_index, 6:8] = normal_dataset["compression_flags"].data[index]

    return filled_timeline


def interpolate_gaps(burst_dataset, gaps, filled_norm_timeline, interpolation_function):
    burst_epochs = burst_dataset["epoch"].data
    # Exclude range values
    burst_vectors = burst_dataset["vectors"].data

    for gap in gaps:
        # TODO: we might need a few inputs before or after start/end
        burst_start = (np.abs(burst_epochs - gap[0])).argmin()
        burst_end = (np.abs(burst_epochs - gap[1])).argmin()
        gap_timeline = filled_norm_timeline[
            np.nonzero(
                (filled_norm_timeline > gap[0]) & (filled_norm_timeline < gap[1])
            )
        ]
        # do not include range
        gap_fill = interpolation_function(
            burst_vectors[burst_start:burst_end, :3],
            burst_epochs[burst_start:burst_end],
            gap_timeline,
        )

        # gaps should not have data in timeline, still check it
        for index, timestamp in enumerate(gap_timeline):
            timeline_index = np.searchsorted(filled_norm_timeline[:, 0], timestamp)
            if sum(filled_norm_timeline[timeline_index, 1:4]) == 0:
                filled_norm_timeline[timeline_index, 1:4] = gap_fill[index]
                filled_norm_timeline[timeline_index, 4] = burst_vectors[burst_start+index, 3]
                filled_norm_timeline[timeline_index, 5] = 1
                filled_norm_timeline[timeline_index, 6:8] = burst_dataset["compression_flags"].data[burst_start+index]

    return filled_norm_timeline


def generate_timeline(epoch_data: np.ndarray, gaps: np.ndarray):
    """

    Parameters
    ----------
    epoch_data

    Returns
    -------

    """
    # given a dataarray of epoch values (from normal mode data) find any gaps of larger than 1 second.
    full_timeline = np.zeros(0)

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
    epoch_data: np.ndarray, vectors_per_second_attr: str = None
) -> np.ndarray:
    """
    Find all the gaps in the epoch data, given

    Parameters
    ----------
    epoch_data
    vectors_per_second

    Returns
    -------

    """
    gaps = np.zeros((0, 2))
    if vectors_per_second_attr is not None and vectors_per_second_attr != "":
        vecsec_segments = vectors_per_second_attr.split(",")
        end_index = epoch_data.shape[0]
        for vecsec_segment in reversed(vecsec_segments):
            start_time, vecsec = vecsec_segment.split(":")
            start_index = np.where(int(start_time) == epoch_data)[0][0]
            gaps = np.concatenate(
                (find_gaps(epoch_data[start_index : end_index + 1], int(vecsec)), gaps)
            )
            end_index = start_index
    else:
        # TODO: How to handle this case
        gaps = find_gaps(epoch_data, 2)  # Assume half second gaps
        # alternatively, I could try and find the average time between vectors

    return gaps


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
    # TODO: is this in the configuration file?
    difference_ns = 0.5 * 1e9

    return np.arange(gap[0], gap[1], difference_ns)
