import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.mag.constants import DataMode
from imap_processing.mag.l1d.mag_l1d_data import MagL1d
from imap_processing.mag.l2.mag_l2 import retrieve_matrix_from_l2_calibration
from imap_processing.mag.l2.mag_l2_data import MagL2L1dBase


def mag_l1d(
    calibration_dataset: xr.Dataset,
    input_mago: xr.Dataset,
    input_magi: xr.Dataset,
    day_to_process: np.datetime64,
    data_mode: DataMode = DataMode.NORM,
) -> list[xr.Dataset]:

    day: np.datetime64 = day_to_process.astype("datetime64[D]")

    calibration_matrix_mago = retrieve_matrix_from_l2_calibration(
        calibration_dataset, day, use_mago=True
    )

    calibration_matrix_magi = retrieve_matrix_from_l2_calibration(
        calibration_dataset, day, use_mago=False
    )

    # Only the first 3 components are used for L1d
    mago_vectors = input_mago["vectors"].data[:, :3]
    magi_vectors = input_magi["vectors"].data[:, :3]

    l1d = MagL1d(
        vectors=mago_vectors,
        epoch=input_mago["epoch"].data,
        range= input_mago["vectors"].data[:, 3],
        global_attributes={},
        quality_flags=np.zeros(len(input_mago["epoch"].data)),
        quality_bitmask=np.zeros(len(input_mago["epoch"].data)),
        data_mode=data_mode,
        magi_vectors=magi_vectors,
        magi_range=input_magi["vectors"].data[:, 3],
        offsets=calibration_dataset["offsets"].data,
        mago_calibration=calibration_matrix_mago,
        magi_calibration=calibration_matrix_magi,
    )

    # TODO: L1D attributes
    attributes = ImapCdfAttributes()
    attributes.add_instrument_global_attrs("mag")
    attributes.add_instrument_variable_attrs("mag", "l2")

    output_dataset = l1d.generate_dataset(
        attributes, day_to_process
    )
    return [output_dataset]
