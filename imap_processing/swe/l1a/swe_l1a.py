"""Contains code to perform SWE L1a processing."""

import logging

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.swe.l1a.swe_science import swe_science
from imap_processing.swe.utils.swe_utils import (
    SWEAPID,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def swe_l1a(packet_file: str, data_version: str) -> xr.Dataset:
    """
    Will process SWE l0 data into l1a data.

    Receive all L0 data file. Based on appId, it
    call its function to process. If appId is science, it requires more work
    than other appId such as appId of housekeeping.

    Parameters
    ----------
    packet_file : str
        Path where the raw packet file is stored.
    data_version : str
        Data version to write to CDF files and the Data_version CDF attribute.
        Should be in the format Vxxx.

    Returns
    -------
    List
        List of xarray.Dataset.
    """
    xtce_document = (
        f"{imap_module_directory}/swe/packet_definitions/swe_packet_definition.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file, xtce_document, use_derived_value=False
    )

    processed_data = []

    if SWEAPID.SWE_SCIENCE in datasets_by_apid:
        logger.info("Processing SWE science data.")
        processed_data.append(
            swe_science(
                l0_dataset=datasets_by_apid[SWEAPID.SWE_SCIENCE],
                data_version=data_version,
            )
        )

    # Process non-science data
    # Define minimal CDF attrs for the non science dataset
    imap_attrs = ImapCdfAttributes()
    imap_attrs.add_instrument_global_attrs("swe")
    imap_attrs.add_global_attribute("Data_version", data_version)
    imap_attrs.add_instrument_variable_attrs("swe", "l1a")
    non_science_attrs = imap_attrs.get_variable_attributes("non_science_attrs")
    epoch_attrs = imap_attrs.get_variable_attributes("epoch", check_schema=False)

    if SWEAPID.SWE_APP_HK in datasets_by_apid:
        logger.info("Processing SWE housekeeping data.")
        hk_ds = datasets_by_apid[SWEAPID.SWE_APP_HK]
        hk_ds.attrs.update(imap_attrs.get_global_attributes("imap_swe_l1a_hk"))
        hk_ds["epoch"].attrs.update(epoch_attrs)
        # Add attrs to HK data variables
        for var_name in hk_ds.data_vars:
            hk_ds[var_name].attrs.update(non_science_attrs)
        processed_data.append(hk_ds)

    if SWEAPID.SWE_CEM_RAW in datasets_by_apid:
        logger.info("Processing SWE CEM raw data.")
        cem_raw_ds = datasets_by_apid[SWEAPID.SWE_CEM_RAW]
        cem_raw_ds.attrs.update(
            imap_attrs.get_global_attributes("imap_swe_l1a_cem-raw")
        )
        cem_raw_ds["epoch"].attrs.update(epoch_attrs)

        # Add attrs to CEM raw data variables
        for var_name in cem_raw_ds.data_vars:
            cem_raw_ds[var_name].attrs.update(non_science_attrs)
        processed_data.append(cem_raw_ds)

    if len(processed_data) == 0:
        logger.info("Data contains unknown APID.")

    return processed_data
