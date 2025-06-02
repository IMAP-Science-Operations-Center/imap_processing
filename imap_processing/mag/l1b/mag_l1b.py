"""MAG L1B Processing."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from xarray import Dataset

from imap_processing.ancillary.ancillary_dataset_combiner import MagAncillaryCombiner
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import vectors_per_second_from_string

logger = logging.getLogger(__name__)


def mag_l1b(
    input_dataset: xr.Dataset,
    day_to_process: np.datetime64,
    calibration_dataset: xr.Dataset,
) -> Dataset:
    """
    Will process MAG L1B data from L1A data.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    day_to_process : np.datetime64
        The day to process, used for setting the date in the output dataset.
    calibration_dataset : xr.Dataset
        The calibration dataset containing calibration matrices and timeshift values for
        mago and magi.
        When None, this defaults to the test calibration file.

    Returns
    -------
    output_dataset : xr.Dataset
        The processed dataset.
    """
    if calibration_dataset is None:
        default_file = (
            Path(__file__).parent / "imap_mag_l1b-calibration_20240229_v002.cdf"
        )
        calibration_dataset = MagAncillaryCombiner(
            [default_file], day_to_process + 3
        ).combined_dataset
        logger.info("Using default test calibration file.")

    source = input_dataset.attrs["Logical_source"]
    if isinstance(source, list):
        source = source[0]

    if "raw" in source:
        # Raw files should not be processed in L1B.
        raise ValueError("Raw L1A file passed into L1B. Unable to process.")

    mag_attributes = ImapCdfAttributes()
    mag_attributes.add_instrument_global_attrs("mag")
    mag_attributes.add_instrument_variable_attrs("mag", "l1b")
    source = source.replace("l1a", "l1b")

    if "mago" in source:
        is_mago = True
    elif "magi" in source:
        is_mago = False
    else:
        raise ValueError(
            f"Calibration matrix not found, invalid logical source "
            f"{input_dataset.attrs['Logical_source']}"
        )

    # TODO: Check validity of time range for calibration
    calibration_matrix, time_shift = retrieve_matrix_from_l1b_calibration(
        calibration_dataset, day_to_process, is_mago
    )
    print(f"Using calibration matrix: {calibration_matrix}")

    output_dataset = mag_l1b_processing(
        input_dataset, calibration_matrix, time_shift, mag_attributes, source
    )

    return output_dataset


def mag_l1b_processing(
    input_dataset: xr.Dataset,
    calibration_matrix: xr.DataArray,
    time_shift: xr.DataArray,
    mag_attributes: ImapCdfAttributes,
    logical_source: str,
) -> xr.Dataset:
    """
    Will process MAG L1B data from L1A data.

    MAG L1B is almost identical to L1A, with only the vectors and attributes getting
    updated. All non-vector variables are the same.

    This step rescales the vector data according to the compression width, and then
    multiplies the vector according to the calibration matrix for a given range. It
    also shifts the timestamps by the values defined in calibration_dataset.

    Parameters
    ----------
    input_dataset : xr.Dataset
        The input dataset to process.
    calibration_matrix : xr.DataArray
        The calibration dataset containing calibration matrices and timeshift values for
        mago and magi.
    time_shift : xr.DataArray
        The time shift to apply for the given sensor. This should be one value and is
        in seconds.
    mag_attributes : ImapCdfAttributes
        Attribute class for output CDF containing MAG L1B attributes.
    logical_source : str
        The logical source of the input dataset, used for setting the output dataset
        attributes.

    Returns
    -------
    output_dataset : xr.Dataset
        L1b dataset.
    """
    dims = [["direction"], ["compression"]]
    new_dims = [["direction"], ["compression"]]

    # Calculate vectors
    l1b_fields = xr.apply_ufunc(
        update_vector,
        input_dataset["vectors"],
        input_dataset["compression_flags"],
        input_core_dims=dims,
        output_core_dims=new_dims,
        vectorize=True,
        keep_attrs=True,
        kwargs={"calibration_matrix": calibration_matrix},
    )

    # Calculate shifted time
    epoch_time = shift_time(input_dataset["epoch"], time_shift)

    # Update attributes and assemble dataset
    epoch_time.attrs = mag_attributes.get_variable_attributes("epoch")

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=mag_attributes.get_variable_attributes(
            "direction_attrs", check_schema=False
        ),
    )

    compression = xr.DataArray(
        np.arange(2),
        name="compression",
        dims=["compression"],
        attrs=mag_attributes.get_variable_attributes(
            "compression_attrs", check_schema=False
        ),
    )

    direction_label = xr.DataArray(
        direction.values.astype(str),
        name="direction_label",
        dims=["direction_label"],
        attrs=mag_attributes.get_variable_attributes(
            "direction_label", check_schema=False
        ),
    )

    compression_label = xr.DataArray(
        compression.values.astype(str),
        name="compression_label",
        dims=["compression_label"],
        attrs=mag_attributes.get_variable_attributes(
            "compression_label", check_schema=False
        ),
    )

    global_attributes = mag_attributes.get_global_attributes(logical_source)
    try:
        global_attributes["is_mago"] = input_dataset.attrs["is_mago"]
        global_attributes["is_active"] = input_dataset.attrs["is_active"]
        global_attributes["vectors_per_second"] = timeshift_vectors_per_second(
            input_dataset.attrs["vectors_per_second"], time_shift
        )
        global_attributes["missing_sequences"] = input_dataset.attrs[
            "missing_sequences"
        ]
    except KeyError as e:
        logger.info(
            f"Key error when assigning global attributes, attribute not found in "
            f"L1A file: {e}"
        )

    output_dataset = xr.Dataset(
        coords={
            "epoch": epoch_time,
            "direction": direction,
            "compression": compression,
            "direction_label": direction_label,
            "compression_label": compression_label,
        },
        attrs=global_attributes,
    )
    # Fill the output with data
    output_dataset["vectors"] = xr.DataArray(
        l1b_fields[0].data,
        name="vectors",
        dims=["epoch", "direction"],
        attrs=mag_attributes.get_variable_attributes("vector_attrs"),
    )

    output_dataset["compression_flags"] = xr.DataArray(
        input_dataset["compression_flags"].data,
        name="compression_flags",
        dims=["epoch", "compression"],
        attrs=mag_attributes.get_variable_attributes("compression_flags_attrs"),
    )
    return output_dataset


def retrieve_matrix_from_l1b_calibration(
    calibration_dataset: xr.Dataset, day: np.datetime64, is_mago: bool = True
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Retrieve the calibration matrix and time shift from the calibration dataset.

    Parameters
    ----------
    calibration_dataset : xarray.Dataset
        The calibration dataset containing the calibration matrices and time shift.
    day : np.datetime64
        The day to process, used for retrieving the correct day of data out of the
        calibration.
    is_mago : bool
        Whether the calibration is for mago or magi. If True, it retrieves the mago
        calibration matrix and time shift. If False, it retrieves the magi calibration
        matrix and time shift.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray]
        The calibration matrix and time shift. These can be passed directly into
        update_vector, calibrate_vector, and shift_time.
    """
    print(f"Finding data for day {day}")
    if is_mago:
        calibration_matrix = calibration_dataset.sel(epoch=day)["MFOTOURFO"]
        time_shift = calibration_dataset.sel(epoch=day)["OTS"]
    else:
        calibration_matrix = calibration_dataset.sel(epoch=day)["MFITOURFI"]
        time_shift = calibration_dataset.sel(epoch=day)["ITS"]

    return calibration_matrix, time_shift


def update_vector(
    input_vector: np.ndarray,
    input_compression: np.ndarray,
    calibration_matrix: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply calibration and compression scaling to vector.

    This calls, in sequence, calibrate_vector and rescale_vector to apply L1B processing
    to the input vector.

    Parameters
    ----------
    input_vector : numpy.ndarray
        One input vector to update, looking like (x, y, z, range).
    input_compression : numpy.ndarray
        Compression flags corresponding to the vector, looking like (is_compressed,
        compression_width).
    calibration_matrix : xr.DataArray
        DataArray containing the full set of calibration matrices, for each range.
        Size is ((3, 3, 4)).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Updated vector and the same compression flags.
    """
    vector = rescale_vector(input_vector, input_compression)
    cal_vector = calibrate_vector(vector, calibration_matrix)
    return cal_vector, input_compression


def rescale_vector(
    input_vector: np.ndarray, compression_flags: np.ndarray
) -> np.ndarray:
    """
    Rescale vector based on compression flags.

    If the first value of compression_flags is zero, this just returns the input_vector
    unchanged. Otherwise, the vector is scaled using the compression width, which is
    the second part of compression_flags.

    The vector is scaled using the following equation:
    M = 2 ^ (16-width)
    output_vector = input_vector * M

    Therefore, for a 16 bit width, the same vector is returned.

    Parameters
    ----------
    input_vector : numpy.ndarray
        One input vector to update, looking like (x, y, z, range).
    compression_flags : numpy.ndarray
        Compression flags corresponding to the vector, looking like (is_compressed,
        compression_width).

    Returns
    -------
    output_vector : numpy.ndarray
        Updated vector.
    """
    output_vector: np.ndarray = input_vector.astype(np.float64)

    if compression_flags[0]:
        factor = np.float_power(2, (16 - compression_flags[1]))
        output_vector[:3] = input_vector.astype(np.float64)[:3] * factor

    return output_vector


def calibrate_vector(
    input_vector: np.ndarray, calibration_matrix: xr.DataArray = None
) -> np.ndarray:
    """
    Apply calibration matrix to input vector.

    The calibration matrix contains 3x3 matrices for each range. To calibrate the input
    vector, we take the range (which is the fourth value of the vector) to determine
    the correct calibration matrix. We then multiply the input vector by that matrix.

    Parameters
    ----------
    input_vector : numpy.ndarray
        The input vector to calibrate [x, y, z, range].
    calibration_matrix : xr.DataArray
        The full set of calibration matrices, for each range. Size is ((3, 3, 4)).

    Returns
    -------
    updated_vector : numpy.ndarray
        Calibrated vector.
    """
    updated_vector: np.ndarray = input_vector.copy().astype(np.float64)
    if input_vector[3] % 1 != 0:
        raise ValueError("Range must be an integer.")

    range = int(input_vector[3])
    x_y_z = updated_vector[:3]
    updated_vector[:3] = np.dot(calibration_matrix.values[:, :, range], x_y_z)

    return updated_vector


def shift_time(epoch_times: xr.DataArray, time_shift: xr.DataArray) -> xr.DataArray:
    """
    Shift epoch times by the provided time_shift calibration value.

    Sometimes the time values calculated from the sensor vary slightly from the "actual"
    time the data was captured. To correct for this, the MAG team provides time shift
    values in the calibration file. This function applies the time shift to the epoch
    times.

    The time shift is provided in seconds. A positive shift is adding time, while a
    negative shift subtracts it (so the values move backwards.)

    This may mean vectors shift out of the specific day that is being processed. To
    manage this, all MAG L0, L1A, L1B, and L1C science data files contain an extra 30
    minute buffer on either side (so the data ranges from
    midnight - 30 minutes to midnight + 24 hours + 30 minutes.)
    The extra buffer is removed at L1D and L2 so those science files are exactly 24
    hours long.

    For more information please refer to the algorithm document.

    Parameters
    ----------
    epoch_times : xr.DataArray
        The input epoch times, in TT J2000 ns.
    time_shift : xr.DataArray
        The time shift to apply for the given sensor. This should be one value and is
        in seconds.

    Returns
    -------
    shifted_times : xr.DataArray
        The shifted epoch times, equal to epoch_times with time_shift added to each
        value.
    """
    if time_shift.size != 1:
        raise ValueError("Time shift must be a single value.")
    # Time shift is in seconds
    time_shift_ns = time_shift.data * 1e9

    return epoch_times + time_shift_ns


def timeshift_vectors_per_second(
    vectors_per_second: str, time_shift: xr.DataArray
) -> str:
    """
    Shift the vectors per second attribute by the time shift value.

    This ensures that the vectors per second attribute is aligned with the epoch values
    if the time is shifted.

    Parameters
    ----------
    vectors_per_second : str
        The vectors per second attribute from the input dataset, in the format
         "timestamp:rate,timestamp:rate".
    time_shift : xr.DataArray
        The time shift to apply for the given sensor. This should be one value and is
        in seconds.

    Returns
    -------
    str
        The updated vectors per second attribute.
    """
    time_shift_ns = time_shift.data * 1e9

    vecsec = vectors_per_second_from_string(vectors_per_second)
    new_vecsec = ""
    for time, rate in vecsec.items():
        new_time = time + time_shift_ns
        new_vecsec += f"{new_time.astype(np.int64)}:{rate},"

    return new_vecsec[:-1]
