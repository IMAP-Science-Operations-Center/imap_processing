"""Contains code to perform ULTRA L1a cdf generation."""
import dataclasses
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import ULTRA_TOF, decom_ultra_apids

logger = logging.getLogger(__name__)


def initiate_data_arrays(decom_ultra: dict, apid: int):
    """Initiate xarray data arrays.

    Parameters
    ----------
    decom_ultra : dict
        Parsed data.
    apid : int
        Packet APID.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    # Converted time
    time_converted = []
    for time in decom_ultra["SHCOARSE"]:
        time_converted.append(calc_start_time(time))

    epoch_time = xr.DataArray(
        time_converted,
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    if apid != ULTRA_TOF.apid[0]:
        dataset = xr.Dataset(
            coords={"Epoch": epoch_time},
            attrs=ultra_cdf_attrs.ultra_l1a_attrs.output(),
        )
    else:
        row = xr.DataArray(
            # Number of pixel rows
            np.arange(54),
            name="Row",
            dims=["Row"],
            attrs=dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc="ROW",  # TODO: short and long descriptions
                fieldname="ROW",
                var_type="ignore_data",
            ).output(),
        )

        column = xr.DataArray(
            # Number of pixel columns
            np.arange(180),
            name="Column",
            dims=["Column"],
            attrs=dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc="COLUMN",  # TODO: short and long descriptions
                fieldname="COLUMN",
                var_type="ignore_data",
            ).output(),
        )

        dataset = xr.Dataset(
            coords={"Epoch": epoch_time, "Row": row, "Column": column},
            attrs=ultra_cdf_attrs.ultra_l1a_attrs.output(),
        )

    return dataset


def xarray(decom_ultra: dict, apid: int):
    """Create xarray for packet.

    Parameters
    ----------
    decom_ultra : dict
        Parsed data.
    apid : int
        Packet APID.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    dataset = initiate_data_arrays(decom_ultra, apid)

    for key, value in decom_ultra.items():
        # EVENT DATA and FASTDATA_00 have been broken down further
        # (see ultra_utils.py) and are therefore not needed.
        if key in {"EVENTDATA", "FASTDATA_00"}:
            continue
        # Packet headers require support attributes
        elif key in [
            "VERSION",
            "TYPE",
            "SEC_HDR_FLG",
            "PKT_APID",
            "SEQ_FLGS",
            "SRC_SEQ_CTR",
            "PKT_LEN",
        ]:
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_support_attrs,
                catdesc=key,  # TODO: short and long descriptions
                fieldname=key,
            ).output()
            dims = ["Epoch"]
        # AUX enums require string attibutes
        elif key in [
            "SPINPERIODVALID",
            "SPINPHASEVALID",
            "SPINPERIODSOURCE",
            "CATBEDHEATERFLAG",
            "HWMODE",
            "IMCENB",
            "LEFTDEFLECTIONCHARGE",
            "RIGHTDEFLECTIONCHARGE",
        ]:
            attrs = dataclasses.replace(
                ultra_cdf_attrs.string_base,
                catdesc=key,  # TODO: short and long descriptions
                fieldname=key,
            ).output()
            dims = ["Epoch"]
        # TOF packetdata has multiple dimensions
        elif key == "PACKETDATA":
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc=key,  # TODO: short and long descriptions
                fieldname=key,
                label_axis=key,
                depend_1="Row",
                depend_2="Column",
                units="PIXELS",
            ).output()
            dims = ["Epoch", "Row", "Column"]
        # Use metadata with a single dimension for
        # all other data products
        else:
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc=key,  # TODO: short and long descriptions
                fieldname=key,
                label_axis=key,
            ).output()
            dims = ["Epoch"]

        dataset[key] = xr.DataArray(
            value,
            name=key,
            dims=dims,
            attrs=attrs,
        )

    return dataset


def ultra_l1a(packet_file: Path, xtce: Path, output_filepath: Path, apid: int):
    """
    Process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : Path
        Path to the CCSDS data packet file.
    xtce : Path
        Path to the XTCE packet definition file.
    output_filepath : Path
        Full directory and filename for CDF file
    apid : int
        Packet APID.
    """
    decom_ultra = decom_ultra_apids(packet_file, xtce, apid)

    dataset = xarray(decom_ultra, apid)
    write_cdf(dataset, Path(output_filepath))
    logging.info(f"Created CDF file at {output_filepath}")
