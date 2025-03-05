"""IMAP-HI L1A processing module."""

import logging
from pathlib import Path
from typing import Union

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.l1a.histogram import create_dataset as hist_create_dataset
from imap_processing.hi.l1a.science_direct_event import science_direct_event
from imap_processing.hi.utils import HIAPID
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def hi_l1a(packet_file_path: Union[str, Path], data_version: str) -> list[xr.Dataset]:
    """
    Will process IMAP raw data to l1a.

    Parameters
    ----------
    packet_file_path : str
        Data packet file path.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of processed xarray dataset.
    """
    packet_def_file = (
        imap_module_directory / "hi/packet_definitions/TLM_HI_COMBINED_SCI.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file_path, xtce_packet_definition=packet_def_file
    )

    # Process science to l1a.
    processed_data = []
    for apid in datasets_by_apid:
        try:
            apid_enum = HIAPID(apid)
        except ValueError as err:
            raise RuntimeError(f"Encountered unexpected APID [{apid}]") from err

        logger.info(f"Processing IMAP-Hi data for {apid_enum.name} packets")

        if apid_enum in [HIAPID.H45_SCI_CNT, HIAPID.H90_SCI_CNT]:
            data = hist_create_dataset(datasets_by_apid[apid])
            gattr_key = "imap_hi_l1a_hist_attrs"
        elif apid_enum in [HIAPID.H45_SCI_DE, HIAPID.H90_SCI_DE]:
            data = science_direct_event(datasets_by_apid[apid])
            gattr_key = "imap_hi_l1a_de_attrs"
        elif apid_enum in [HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK]:
            data = datasets_by_apid[apid]
            gattr_key = "imap_hi_l1a_hk_attrs"
        elif apid_enum in [HIAPID.H45_DIAG_FEE, HIAPID.H90_DIAG_FEE]:
            data = datasets_by_apid[apid]
            gattr_key = "imap_hi_l1a_diagfee_attrs"

        # Update dataset global attributes
        attr_mgr = ImapCdfAttributes()
        attr_mgr.add_instrument_global_attrs("hi")
        data.attrs.update(attr_mgr.get_global_attributes(gattr_key))

        # TODO: revisit this
        data.attrs["Data_version"] = data_version

        # set the sensor string in Logical_source
        sensor_str = apid_enum.sensor
        data.attrs["Logical_source"] = data.attrs["Logical_source"].format(
            sensor=sensor_str
        )
        processed_data.append(data)
    return processed_data
