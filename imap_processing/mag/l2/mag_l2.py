"""Module to run MAG L2 processing."""

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag import imap_mag_sdc_configuration_v001 as configuration
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1b.mag_l1b import calibrate_vector
from imap_processing.mag.l2.mag_l2_data import MagL2


def mag_l2(
    calibration_datasets: list[xr.Dataset],
    offsets_dataset: xr.Dataset,
    input_data: xr.Dataset,
) -> list[xr.Dataset]:
    """
    Complete MAG L2 processing.

    Processing uses 4 data input sources:
    1. Calibration dataset
        These calibration files are ancillary files and may require multiple files to
        cover the entire timespan. They are not expected to change often. They are used
        to provide rotation matrices to correct the frame of the vectors. The same
        file(s) are used for both burst and norm calculations.
    2. Offsets dataset
        This is one, hand-created file which must correspond exactly to an L1B
        (for burst) or L1C (for norm) data file. For each vector, this file includes
        offsets, timedelta, and quality flags. The offsets are added to the vectors,
        the timedelta is used to correct the epoch time, and the quality flags are
        directly passed into the output file.
    3. Input data
        This is the L1B or L1C data file. It is used to provide the vectors and epoch
        time. It should always be MAGo in the nominal case, but it is possible that we
        will switch permanently to using MAGi (in the case of sensor failure, for
        example.) The offsets dataset and the input
        data are tightly related, so the input data filename is actually retrieved from
        the offset dataset to ensure they always match.
    4. sdc-configuration
        This is a local configuration file for changes we never expect to make in
        flight. This is in the IMAP local repo because changes to these settings will
        require other code updates to validate the changes. In L2, the only setting used
        is "always_output_mago", which indicates whether we should always output MAGo.
        Note that if this ever is set to False, we will need to update the dependency
        system to set MAGi files as an upstream dependency.

    Input data can be burst or normal mode, but MUST match the file in offset_dataset.
    TODO: retrieve the file from offset_dataset in cli.py.
    Calibration dataset is the same for all runs.

    MAGi data is not used unless we indicate it.

    Parameters
    ----------
    calibration_datasets : list[xr.Dataset]
        Calibration ancillary file inputs.
    offsets_dataset : xr.Dataset
        Offset ancillary file input.
    input_data : xr.Dataset
        Input data from MAG L1C or L1B.

    Returns
    -------
    list[xr.Dataset]
        List of xarray datasets ready to write to CDF file. Expected to be four outputs
        for different frames.
    """
    # TODO we may need to combine multiple calibration datasets into one timeline.

    always_output_mago = configuration.ALWAYS_OUTPUT_MAGO

    # TODO Check that the input file matches the offsets file
    if not np.array_equal(input_data["epoch"].data, offsets_dataset["epoch"].data):
        raise ValueError("Input file and offsets file must have the same timestamps.")

    calibration_matrix = retrieve_matrix_from_l2_calibration(
        calibration_datasets, always_output_mago
    )

    vectors = np.apply_along_axis(
        func1d=calibrate_vector,
        axis=1,
        arr=input_data["vectors"].data,
        calibration_matrix=calibration_matrix,
    )

    basic_test_data = MagL2(
        vectors[:, :3],  # level 2 vectors don't include range
        input_data["epoch"].data,
        input_data["vectors"].data[:, 3],
        {},
        np.zeros(len(input_data["epoch"].data)),
        np.zeros(len(input_data["epoch"].data)),
        DataMode.NORM,
        offsets=offsets_dataset["offsets"].data,
        timedelta=offsets_dataset["timedeltas"].data,
    )
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    # temporarily point to l1c
    attributes.add_instrument_variable_attrs("mag", "l1c")
    return [basic_test_data.generate_dataset(attributes)]


def retrieve_matrix_from_l2_calibration(
    calibration_datasets: list[xr.Dataset], use_mago: bool = True
) -> xr.DataArray:
    """
    Get the calibration matrix for the file.

    Parameters
    ----------
    calibration_datasets : list[xr.Dataset]
        Ancillary file inputs for calibration.
    use_mago : bool
        Use the MAGo calibration matrix. Default is True.

    Returns
    -------
    np.ndarray
        Calibration matrix in the shape (3, 3, 4) to rotate vectors.
    """
    # TODO: allow for multiple inputs
    if isinstance(calibration_datasets, list):
        calibration_dataset = calibration_datasets[0]
        if len(calibration_datasets) > 1:
            raise NotImplementedError
    else:
        calibration_dataset = calibration_datasets

    if use_mago:
        calibration_data = calibration_dataset["URFTOORFO"]
    else:
        calibration_data = calibration_dataset["URFTOORFI"]

    # TODO will need to combine multiple files here
    # TODO: Check validity of the calibration file?

    return calibration_data
