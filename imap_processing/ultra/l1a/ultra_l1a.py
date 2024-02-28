"""Contains code to perform ULTRA L1a science processing."""
import dataclasses
import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.cdf.utils import calc_start_time, write_cdf
from imap_processing.ultra import ultra_cdf_attrs
from imap_processing.ultra.l0.decom_ultra import ULTRA_AUX, decom_ultra_apids

logger = logging.getLogger(__name__)


def xarray_aux(decom_ultra_aux: dict):
    """Create xarray for auxiliary packet.

    Parameters
    ----------
    decom_ultra_aux : dict
        Parsed data.

    Returns
    -------
    dataset : xarray.Dataset
        Data in xarray format.
    """
    # Converted time
    time_converted = []
    for time in decom_ultra_aux["SHCOARSE"]:
        time_converted.append(calc_start_time(time))

    epoch_time = xr.DataArray(
        time_converted,
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"Epoch": epoch_time},
        attrs=ultra_cdf_attrs.ultra_l1a_attrs.output(),
    )

    for key, value in decom_ultra_aux.items():
        if key in [
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
        else:
            attrs = dataclasses.replace(
                ultra_cdf_attrs.ultra_metadata_attrs,
                catdesc=key,  # TODO: short and long descriptions
                fieldname=key,
                label_axis=key,
            ).output()

        dataset[key] = xr.DataArray(
            value,
            name=key,
            dims=["Epoch"],
            attrs=attrs,
        )

    return dataset


def ultra_l1a(packet_file: Path, xtce: Path, output_filepath: Path):
    """
    Process ULTRA L0 data into L1A CDF files at cdf_filepath..

    Parameters
    ----------
    packet_file : Path
        Path to the CCSDS data packet file.
    xtce : Path
        Path to the XTCE packet definition file.
    output_filepath : Path
        Full directory and filename for CDF file
    """
    decom_ultra_aux = decom_ultra_apids(packet_file, xtce, ULTRA_AUX.apid[0])

    if decom_ultra_aux:
        dataset = xarray_aux(decom_ultra_aux)
        write_cdf(dataset, Path(output_filepath))
        logging.info(f"Created CDF file at {output_filepath}")
