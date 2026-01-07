"""Generate ULTRA L1a CDFs."""

import logging

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
    ULTRA_EXTOF_HIGH_ANGULAR,
    ULTRA_EXTOF_HIGH_ENERGY,
    ULTRA_EXTOF_HIGH_TIME,
    ULTRA_HK,
    ULTRA_MACROS_CHECKSUM,
    ULTRA_PHXTOF_HIGH_ANGULAR,
    ULTRA_PHXTOF_HIGH_ENERGY,
    ULTRA_PHXTOF_HIGH_TIME,
    ULTRA_PRI_1_EVENTS,
    ULTRA_PRI_2_EVENTS,
    ULTRA_PRI_3_EVENTS,
    ULTRA_PRI_4_EVENTS,
    ULTRA_RATES,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def ultra_l1a(  # noqa: PLR0912
    packet_file: str, apid_input: int | None = None, create_derived_l1b: bool = False
) -> list[xr.Dataset]:
    """
    Will process ULTRA L0 data into L1A CDF files at output_filepath.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    apid_input : Optional[int]
        Optional apid.
    create_derived_l1b : bool
        Whether to create the l1b datasets with derived values.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    xtce = str(
        f"{imap_module_directory}/ultra/packet_definitions/ULTRA_SCI_COMBINED.xml"
    )

    # Keep a list to track the two versions, l1a and l1b with the derived values.
    decommutated_packet_datasets = []
    datasets_by_apid = packet_file_to_datasets(packet_file, xtce)
    decommutated_packet_datasets.append(datasets_by_apid)
    if create_derived_l1b:
        # For the housekeeping products, we can create the l1b at the same time
        # as the l1a since there is no additional processing needed.
        datasets_by_apid = packet_file_to_datasets(
            packet_file, xtce, use_derived_value=True
        )
        decommutated_packet_datasets.append(datasets_by_apid)

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

    all_l1a_image_apids = {
        apid: group
        for group in [
            ULTRA_PHXTOF_HIGH_ANGULAR,
            ULTRA_PHXTOF_HIGH_ENERGY,
            ULTRA_PHXTOF_HIGH_TIME,
            ULTRA_EXTOF_HIGH_ANGULAR,
            ULTRA_EXTOF_HIGH_TIME,
            ULTRA_EXTOF_HIGH_ENERGY,
        ]
        for apid in group.apid
    }

    # Update dataset global attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("ultra")
    attr_mgr.add_instrument_variable_attrs("ultra", "l1a")

    for i, datasets_by_apid in enumerate(decommutated_packet_datasets):
        for apid in apids:
            logger.info(f"Processing APID: {apid}")
            if apid in ULTRA_AUX.apid:
                decom_ultra_dataset = datasets_by_apid[apid]
                gattr_key = ULTRA_AUX.logical_source[ULTRA_AUX.apid.index(apid)]
            elif apid in all_l1a_image_apids:
                packet_props = all_l1a_image_apids[apid]
                # TODO there is a known issue with the decom of some tof packets.
                #   This is being tracked in issue #2557.
                try:
                    decom_ultra_dataset = process_ultra_tof(
                        datasets_by_apid[apid], packet_props
                    )
                    gattr_key = packet_props.logical_source[
                        packet_props.apid.index(apid)
                    ]
                except IndexError as e:
                    logger.error(f"Error processing TOF data for APID {apid}: {e}")
                    continue
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
                # We don't want to process the event l1b datasets since those l1b
                # products need more information
                if i == 1:
                    continue
                decom_ultra_dataset = process_ultra_events(datasets_by_apid[apid], apid)
                gattr_key = all_event_apids[apid]
                # Add coordinate attributes
                attrs = attr_mgr.get_variable_attributes("event_id")
                decom_ultra_dataset.coords["event_id"].attrs.update(attrs)
            elif apid in ULTRA_ENERGY_SPECTRA.apid:
                decom_ultra_dataset = process_ultra_energy_spectra(
                    datasets_by_apid[apid]
                )
                decom_ultra_dataset = decom_ultra_dataset.drop_vars("compdata")
                gattr_key = ULTRA_ENERGY_SPECTRA.logical_source[
                    ULTRA_ENERGY_SPECTRA.apid.index(apid)
                ]
            elif apid in ULTRA_MACROS_CHECKSUM.apid:
                decom_ultra_dataset = process_ultra_macros_checksum(
                    datasets_by_apid[apid]
                )
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
                gattr_key = ULTRA_CMD_TEXT.logical_source[
                    ULTRA_CMD_TEXT.apid.index(apid)
                ]
            elif apid in ULTRA_CMD_ECHO.apid:
                decom_ultra_dataset = process_ultra_cmd_echo(datasets_by_apid[apid])
                gattr_key = ULTRA_CMD_ECHO.logical_source[
                    ULTRA_CMD_ECHO.apid.index(apid)
                ]
            else:
                logger.error(f"APID {apid} not recognized.")
                continue

            decom_ultra_dataset.attrs.update(attr_mgr.get_global_attributes(gattr_key))

            if i == 1:
                # Derived values dataset at l1b
                # We already have the l1a attributes, just update the l1a -> l1b
                # in the metadata.
                decom_ultra_dataset.attrs["Data_type"] = decom_ultra_dataset.attrs[
                    "Data_type"
                ].replace("1A", "1B")
                decom_ultra_dataset.attrs["Logical_source"] = decom_ultra_dataset.attrs[
                    "Logical_source"
                ].replace("l1a", "l1b")
                decom_ultra_dataset.attrs["Logical_source_description"] = (
                    decom_ultra_dataset.attrs["Logical_source_description"].replace(
                        "1A", "1B"
                    )
                )

            # Add data variable attributes
            for key in decom_ultra_dataset.data_vars:
                attrs = attr_mgr.get_variable_attributes(key.lower())
                decom_ultra_dataset.data_vars[key].attrs.update(attrs)
                if i == 1:
                    # For l1b datasets, the FILLVAL and VALIDMIN/MAX may be
                    # different datatypes, so we can't use them directly from l1a.
                    # just remove them for now since we don't really have a need for
                    # for them currently.
                    for attr_key in ["FILLVAL", "VALIDMIN", "VALIDMAX"]:
                        if attr_key in decom_ultra_dataset.data_vars[key].attrs:
                            decom_ultra_dataset.data_vars[key].attrs.pop(attr_key)

            # Add coordinate attributes
            attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
            decom_ultra_dataset.coords["epoch"].attrs.update(attrs)

            output_datasets.append(decom_ultra_dataset)

    return output_datasets
