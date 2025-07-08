"""Generate ULTRA L1a CDFs."""

import logging
from typing import Optional

import xarray as xr

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.ultra.l0.decom_ultra import (
    process_ultra_cmd_echo,
    process_ultra_energy_rates,
    process_ultra_energy_spectra,
    process_ultra_events,
    process_ultra_macros_checksum,
    process_ultra_rates,
    process_ultra_tof,
)
from imap_processing.ultra.l0.ultra_utils import (
    ULTRA_AUX,
    ULTRA_CMD_ECHO,
    ULTRA_CMD_TEXT,
    ULTRA_ENERGY_EVENTS,
    ULTRA_ENERGY_RATES,
    ULTRA_ENERGY_SPECTRA,
    ULTRA_EVENTS,
    ULTRA_HK,
    ULTRA_MACROS_CHECKSUM,
    ULTRA_PRI_1_EVENTS,
    ULTRA_PRI_2_EVENTS,
    ULTRA_PRI_3_EVENTS,
    ULTRA_PRI_4_EVENTS,
    ULTRA_RATES,
    ULTRA_TOF,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def ultra_l1a(  # noqa: PLR0912
    packet_file: str, apid_input: Optional[int] = None
) -> list[xr.Dataset]:
    """
    Will process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    apid_input : Optional[int]
        Optional apid.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    xtce = str(
        f"{imap_module_directory}/ultra/packet_definitions/ULTRA_SCI_COMBINED.xml"
    )

    datasets_by_apid = packet_file_to_datasets(packet_file, xtce)

    output_datasets = []

    # This is used for two purposes currently:
    #    For testing purposes to only generate a dataset for a single apid.
    #    Each test dataset is only for a single apid while the rest of the apids
    #    contain zeros. Ideally we would have
    #    test data for all apids and remove this parameter.
    if apid_input is not None:
        apids = [apid_input]
    else:
        apids = list(datasets_by_apid.keys())

    all_event_apids = {
        apid: group.logical_source[i]
        for group in [
            ULTRA_EVENTS,
            ULTRA_ENERGY_EVENTS,
            ULTRA_PRI_1_EVENTS,
            ULTRA_PRI_2_EVENTS,
            ULTRA_PRI_3_EVENTS,
            ULTRA_PRI_4_EVENTS,
        ]
        for i, apid in enumerate(group.apid)
    }

    # Update dataset global attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("ultra")
    attr_mgr.add_instrument_variable_attrs("ultra", "l1a")

    for apid in apids:
        if apid in ULTRA_AUX.apid:
            decom_ultra_dataset = datasets_by_apid[apid]
            gattr_key = ULTRA_AUX.logical_source[ULTRA_AUX.apid.index(apid)]
        elif apid in ULTRA_TOF.apid:
            decom_ultra_dataset = process_ultra_tof(datasets_by_apid[apid])
            gattr_key = ULTRA_TOF.logical_source[ULTRA_TOF.apid.index(apid)]
        elif apid in ULTRA_RATES.apid:
            decom_ultra_dataset = process_ultra_rates(datasets_by_apid[apid])
            decom_ultra_dataset = decom_ultra_dataset.drop_vars("fastdata_00")
            gattr_key = ULTRA_RATES.logical_source[ULTRA_RATES.apid.index(apid)]
        elif apid in ULTRA_ENERGY_RATES.apid:
            decom_ultra_dataset = process_ultra_energy_rates(datasets_by_apid[apid])
            decom_ultra_dataset = decom_ultra_dataset.drop_vars("ratedata")
            gattr_key = ULTRA_ENERGY_RATES.logical_source[
                ULTRA_ENERGY_RATES.apid.index(apid)
            ]
        elif apid in all_event_apids:
            decom_ultra_dataset = process_ultra_events(datasets_by_apid[apid], apid)
            gattr_key = all_event_apids[apid]
            # Add coordinate attributes
            attrs = attr_mgr.get_variable_attributes("event_id")
            decom_ultra_dataset.coords["event_id"].attrs.update(attrs)
        elif apid in ULTRA_ENERGY_SPECTRA.apid:
            decom_ultra_dataset = process_ultra_energy_spectra(datasets_by_apid[apid])
            decom_ultra_dataset = decom_ultra_dataset.drop_vars("compdata")
            gattr_key = ULTRA_ENERGY_SPECTRA.logical_source[
                ULTRA_ENERGY_SPECTRA.apid.index(apid)
            ]
        elif apid in ULTRA_MACROS_CHECKSUM.apid:
            decom_ultra_dataset = process_ultra_macros_checksum(datasets_by_apid[apid])
            gattr_key = ULTRA_MACROS_CHECKSUM.logical_source[
                ULTRA_MACROS_CHECKSUM.apid.index(apid)
            ]
        elif apid in ULTRA_HK.apid:
            decom_ultra_dataset = datasets_by_apid[apid]
            gattr_key = ULTRA_HK.logical_source[ULTRA_HK.apid.index(apid)]
        elif apid in ULTRA_CMD_TEXT.apid:
            decom_ultra_dataset = datasets_by_apid[apid]
            decoded_strings = [
                s.decode("ascii").rstrip("\x00")
                for s in decom_ultra_dataset["text"].values
            ]
            decom_ultra_dataset = decom_ultra_dataset.drop_vars("text")
            decom_ultra_dataset["text"] = xr.DataArray(
                decoded_strings,
                dims=["epoch"],
                coords={"epoch": decom_ultra_dataset["epoch"]},
            )
            gattr_key = ULTRA_CMD_TEXT.logical_source[ULTRA_CMD_TEXT.apid.index(apid)]
        elif apid in ULTRA_CMD_ECHO.apid:
            decom_ultra_dataset = process_ultra_cmd_echo(datasets_by_apid[apid])
            gattr_key = ULTRA_CMD_ECHO.logical_source[ULTRA_CMD_ECHO.apid.index(apid)]
        else:
            logger.error(f"APID {apid} not recognized.")
            continue

        decom_ultra_dataset.attrs.update(attr_mgr.get_global_attributes(gattr_key))

        # Add data variable attributes
        for key in decom_ultra_dataset.data_vars:
            attrs = attr_mgr.get_variable_attributes(key.lower())
            decom_ultra_dataset.data_vars[key].attrs.update(attrs)

        # Add coordinate attributes
        attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
        decom_ultra_dataset.coords["epoch"].attrs.update(attrs)

        output_datasets.append(decom_ultra_dataset)

    return output_datasets
