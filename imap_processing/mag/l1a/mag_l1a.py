"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import J2000_EPOCH, met_to_j2000ns
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0 import decom_mag
from imap_processing.mag.l0.mag_l0_data import MagL0
from imap_processing.mag.l1a.mag_l1a_data import MagL1a, TimeTuple

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath: Path, data_version: str) -> list[Path]:
    """
    Will process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packet_filepath : pathlib.Path
        Packet files for processing.
    data_version : str
        Data version to write to CDF files.

    Returns
    -------
    generated_files : list[pathlib.Path]
        A list of generated filenames.
    """
    packets = decom_mag.decom_packets(packet_filepath)

    norm_data = packets["norm"]
    burst_data = packets["burst"]

    generated_files = process_and_write_data(
        norm_data,
        mag_cdf_attrs.mag_l1a_norm_raw_attrs.output(),
        mag_cdf_attrs.mag_l1a_norm_mago_attrs.output(),
        mag_cdf_attrs.mag_l1a_norm_magi_attrs.output(),
        data_version,
    )
    generated_files += process_and_write_data(
        burst_data,
        mag_cdf_attrs.mag_l1a_burst_raw_attrs.output(),
        mag_cdf_attrs.mag_l1a_burst_mago_attrs.output(),
        mag_cdf_attrs.mag_l1a_burst_magi_attrs.output(),
        data_version,
    )

    return generated_files


def process_and_write_data(
    packet_data: list[MagL0],
    raw_attrs: dict,
    mago_attrs: dict,
    magi_attrs: dict,
    data_version: str,
) -> list[Path]:
    """
    Will process MAG L0 data into L1A, then create and write out CDF files.

    Norm and burst mode descriptors are distinguished with the passed in attrs.

    Parameters
    ----------
    packet_data : list[MagL0]
        List of MagL0 packets to process, containing primary and secondary sensor data.
    raw_attrs : dict
        Attributes for MagL1A raw CDF files.
    mago_attrs : dict
        Attributes for MagL1A MAGo CDF files.
    magi_attrs : dict
        Attributes for MagL1A MAGi CDF files.
    data_version : str
        Data version.

    Returns
    -------
    generated_files : list[pathlib.Path]
        A list of generated filenames.
    """
    if not packet_data:
        return []

    # TODO: Rework attrs to be better
    raw_attrs["Data_version"] = data_version
    magi_attrs["Data_version"] = data_version
    mago_attrs["Data_version"] = data_version

    mag_raw = decom_mag.generate_dataset(packet_data, raw_attrs)

    generated_datasets = [mag_raw]

    l1a = process_packets(packet_data)

    for _, mago in l1a["mago"].items():
        norm_mago_output = generate_dataset(mago, mago_attrs)
        generated_datasets.append(norm_mago_output)

    for _, magi in l1a["magi"].items():
        norm_magi_output = generate_dataset(magi, magi_attrs)
        generated_datasets.append(norm_magi_output)

    return generated_datasets


def process_packets(
    mag_l0_list: list[MagL0],
) -> dict[str, dict[np.datetime64, MagL1a]]:
    """
    Given a list of MagL0 packets, process them into MagO and MagI L1A data classes.

    Parameters
    ----------
    mag_l0_list : list[MagL0]
        List of Mag L0 packets to process.

    Returns
    -------
    packet_dict : dict[str, dict[numpy.datetime64, MagL1a]]
        Dictionary containing two keys: "mago" which points to a dictionary of mago
         MagL1A objects, and "magi" which points to a dictionary of magi MagL1A objects.
         Each dictionary has keys of days and values of MagL1A objects, so each day
         corresponds to one MagL1A object.
    """
    magi = {}
    mago = {}

    for mag_l0 in mag_l0_list:
        if mag_l0.COMPRESSION:
            raise NotImplementedError("Unable to process compressed data")

        primary_start_time = TimeTuple(mag_l0.PRI_COARSETM, mag_l0.PRI_FNTM)
        secondary_start_time = TimeTuple(mag_l0.SEC_COARSETM, mag_l0.SEC_FNTM)

        primary_day = (
            J2000_EPOCH
            + met_to_j2000ns(primary_start_time.to_seconds()).astype("timedelta64[ns]")
        ).astype("datetime64[D]")
        secondary_day = (
            J2000_EPOCH
            + met_to_j2000ns(secondary_start_time.to_seconds()).astype(
                "timedelta64[ns]"
            )
        ).astype("datetime64[D]")

        # seconds of data in this packet is the SUBTYPE plus 1
        seconds_per_packet = mag_l0.PUS_SSUBTYPE + 1

        # now we know the number of secs of data in the packet, and the data rates of
        # each sensor, we can calculate how much data is in this packet and where the
        # byte boundaries are.

        # VECSEC is already decoded in mag_l0
        total_primary_vectors = seconds_per_packet * mag_l0.PRI_VECSEC
        total_secondary_vectors = seconds_per_packet * mag_l0.SEC_VECSEC

        # The raw vectors are of type int8, but the output vectors should be at least
        # int16.
        primary_vectors, secondary_vectors = MagL1a.process_vector_data(
            mag_l0.VECTORS.astype(dtype=np.int32),  # type: ignore[union-attr]
            # TODO Maybe Change, Item "str" of "Union[Any, str]"
            #  has no attribute "astype"
            # this is because mypy expects both to have the attributes
            total_primary_vectors,
            total_secondary_vectors,
        )

        primary_timestamped_vectors = MagL1a.calculate_vector_time(
            primary_vectors, mag_l0.PRI_VECSEC, primary_start_time
        )
        secondary_timestamped_vectors = MagL1a.calculate_vector_time(
            secondary_vectors, mag_l0.SEC_VECSEC, secondary_start_time
        )

        # Sort primary and secondary into MAGo and MAGi by 24 hour chunks
        # TODO: Individual vectors should be sorted by day, not the whole packet
        mago_is_primary = mag_l0.PRI_SENS == 0

        mago_day = primary_day if mago_is_primary else secondary_day
        magi_day = primary_day if not mago_is_primary else secondary_day

        if mago_day not in mago:
            mago[mago_day] = MagL1a(
                True,
                bool(mag_l0.MAGO_ACT),
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if mago_is_primary
                else secondary_timestamped_vectors,
            )
        else:
            mago[mago_day].vectors = np.concatenate(
                [
                    mago[mago_day].vectors,
                    (
                        primary_timestamped_vectors
                        if mago_is_primary
                        else secondary_timestamped_vectors
                    ),
                ]
            )

        if magi_day not in magi:
            magi[magi_day] = MagL1a(
                False,
                bool(mag_l0.MAGI_ACT),
                mag_l0.SHCOARSE,
                primary_timestamped_vectors
                if not mago_is_primary
                else secondary_timestamped_vectors,
            )
        else:
            magi[magi_day].vectors = np.concatenate(
                [
                    magi[magi_day].vectors,
                    (
                        primary_timestamped_vectors
                        if not mago_is_primary
                        else secondary_timestamped_vectors
                    ),
                ]
            )

    return {"mago": mago, "magi": magi}


def generate_dataset(single_file_l1a: MagL1a, dataset_attrs: dict) -> xr.Dataset:
    """
    Generate a Xarray dataset for L1A data to output to CDF files.

    Global_attrs should contain all info about mago/magi and burst/norm distinction, as
     well as any general info in the global attributes.

     Assumes each MagL1a object is a single day of data, so one MagL1a object has one
     CDF file output.

    Parameters
    ----------
    single_file_l1a : MagL1a
        L1A data covering one day to process into a xarray dataset.
    dataset_attrs : dict
        Global attributes for the dataset, as created by mag_attrs.

    Returns
    -------
    dataset : xarray.Dataset
        One xarray dataset with proper CDF attributes and shape containing MAG L1A data.
    """
    # TODO: Just leave time in datetime64 type with vector as dtype object to avoid this
    # Get the timestamp from the end of the vector
    time_data = single_file_l1a.vectors[:, 4].astype(
        np.dtype("datetime64[ns]"), copy=False
    )

    direction = xr.DataArray(
        np.arange(4),
        name="direction",
        dims=["direction"],
        attrs=mag_cdf_attrs.direction_attrs.output(),
    )

    # TODO: Epoch here refers to the start of the sample. Confirm that this is
    # what mag is expecting, and if it is, CATDESC needs to be updated.
    epoch_time = xr.DataArray(
        time_data,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    vectors = xr.DataArray(
        single_file_l1a.vectors[:, :4],
        name="vectors",
        dims=["epoch", "direction"],
        attrs=mag_cdf_attrs.vector_attrs.output(),
    )

    output = xr.Dataset(
        coords={"epoch": epoch_time, "direction": direction},
        attrs=dataset_attrs,
    )

    output["vectors"] = vectors

    # TODO: Put is_mago and active in the header

    return output
