"""CoDICE L1A processing functions."""

import datetime
import logging
import os

import numpy as np
import xarray as xr
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.codice.codice_l1a_de import l1a_direct_event
from imap_processing.codice.codice_l1a_hi_counters_aggregated import (
    l1a_hi_counters_aggregated,
)
from imap_processing.codice.codice_l1a_hi_counters_singles import (
    l1a_hi_counters_singles,
)
from imap_processing.codice.codice_l1a_hi_omni import l1a_hi_omni
from imap_processing.codice.codice_l1a_hi_priority import l1a_hi_priority
from imap_processing.codice.codice_l1a_hi_sectored import l1a_hi_sectored
from imap_processing.codice.codice_l1a_lo_counters_aggregated import (
    l1a_lo_counters_aggregated,
)
from imap_processing.codice.codice_l1a_lo_counters_singles import (
    l1a_lo_counters_singles,
)
from imap_processing.codice.codice_l1a_lo_priority import l1a_lo_priority
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.constants import CODICEAPID
from imap_processing.codice.utils import process_by_table_id
from imap_processing.quality_flags import ImapCodiceL1aFlags
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def process_l1a(  # noqa: PLR0912
    dependency: ProcessingInputCollection,
) -> list[xr.Dataset]:
    """
    Process L1A data based on descriptor and dependencies.

    Parameters
    ----------
    dependency : ProcessingInputCollection
        Collection of processing inputs required for L1A processing.

    Returns
    -------
    list[xarray.Dataset]
        List of processed L1A datasets generated from available APIDs.
    """
    # Get science data which is L0 packet file
    science_file = dependency.get_file_paths(data_type="l0")[0]
    # TODO get the exact time the FSW changed on january 29 and relabel the xml file
    # On January 29, 2026, the CoDICE flight software was updated to a new version.
    # This update included changes to the packet definitions.
    start_date = datetime.datetime.strptime(
        os.path.basename(science_file).split("_")[4], "%Y%m%d"
    )  # Extract the date from the filename
    path = imap_module_directory / "codice/packet_definitions/"
    if start_date >= datetime.datetime(2026, 1, 29):
        xtce_file = path / "imap_codice_packet-definition_20260129_v001.xml"
    else:
        xtce_file = path / "imap_codice_packet-definition_20250101_v001.xml"

    datasets_by_apid = packet_file_to_datasets(
        science_file,
        xtce_file,
    )
    # Initialize hk dataset
    hk_l1a_ds = None
    datasets = []
    for apid in datasets_by_apid:
        if apid not in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            # Get LUT file. Direct events do not need LUT
            lut_file = dependency.get_file_paths(descriptor="l1a-sci-lut")
            lut_file = lut_file[0]

        if apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
            logger.info("Processing Lo SW Species Counts")
            datasets.append(
                process_by_table_id(datasets_by_apid[apid], lut_file, l1a_lo_species)
            )
        elif apid == CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS:
            datasets.append(
                process_by_table_id(datasets_by_apid[apid], lut_file, l1a_hi_omni)
            )
        elif apid == CODICEAPID.COD_HI_SECT_SPECIES_COUNTS:
            logger.info("Processing Hi Sectored Species Counts")
            datasets.append(
                process_by_table_id(datasets_by_apid[apid], lut_file, l1a_hi_sectored)
            )
        elif apid == CODICEAPID.COD_HI_PHA:
            logger.info("Processing Direct Events for Hi")
            datasets.append(l1a_direct_event(datasets_by_apid[apid], apid=apid))
        elif apid == CODICEAPID.COD_LO_PHA:
            logger.info("Processing Direct Events for Lo")
            datasets.append(l1a_direct_event(datasets_by_apid[apid], apid=apid))
        elif apid in [
            CODICEAPID.COD_LO_SW_PRIORITY_COUNTS,
            CODICEAPID.COD_LO_NSW_PRIORITY_COUNTS,
        ]:
            logger.info(f"Processing {apid} Priority Counts")
            datasets.append(
                process_by_table_id(datasets_by_apid[apid], lut_file, l1a_lo_priority)
            )
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_PRIORITIES:
            logger.info("Processing Hi Priority Counts")
            datasets.append(
                process_by_table_id(datasets_by_apid[apid], lut_file, l1a_hi_priority)
            )
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_AGGREGATED:
            logger.info("Processing Hi Counters aggregated")
            datasets.append(
                process_by_table_id(
                    datasets_by_apid[apid], lut_file, l1a_hi_counters_aggregated
                )
            )
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_SINGLES:
            logger.info("Processing Hi Counters singles")
            datasets.append(
                process_by_table_id(
                    datasets_by_apid[apid], lut_file, l1a_hi_counters_singles
                )
            )
        elif apid == CODICEAPID.COD_LO_INST_COUNTS_AGGREGATED:
            logger.info("Processing Lo Counters aggregated")
            datasets.append(
                process_by_table_id(
                    datasets_by_apid[apid], lut_file, l1a_lo_counters_aggregated
                )
            )
        elif apid == CODICEAPID.COD_LO_INST_COUNTS_SINGLES:
            logger.info("Processing Lo Counters singles")
            datasets.append(
                process_by_table_id(
                    datasets_by_apid[apid], lut_file, l1a_lo_counters_singles
                )
            )
        elif apid == CODICEAPID.COD_NHK:
            logger.info("Processing l1a housekeeping data")
            cdf_attrs = ImapCdfAttributes()
            cdf_attrs.add_instrument_global_attrs("codice")
            hk_l1a_ds = datasets_by_apid[apid]
            hk_l1a_ds.attrs.update(
                cdf_attrs.get_global_attributes("imap_codice_l1a_hskp")
            )
            datasets.append(hk_l1a_ds)

            # l1b processing needs to re-run packet file to datasets to do the
            # housekeeping engineering unit conversions based on the packet definitions
            # We only do this if there are any housekeeping packets that need it so we
            # don't process unnecessarily here.
            logger.info("Processing l1b housekeeping data")
            l1b_ds = packet_file_to_datasets(
                science_file,
                xtce_file,
                use_derived_value=True,
            )[apid]
            l1b_ds.attrs.update(cdf_attrs.get_global_attributes("imap_codice_l1b_hskp"))
            datasets.append(l1b_ds)

    # Add quality flags to each science dataset.
    if hk_l1a_ds is not None:
        flags = calculate_quality_flags(hk_l1a_ds.macro_status.values)
    else:
        flags = None
        logger.warning(
            "No L1A housekeeping data. Unable to calculate quality flags"
            " for science datasets."
        )
    add_quality_flags_to_science_datasets(flags, hk_l1a_ds.epoch.values, datasets)
    return datasets


def calculate_quality_flags(macro_status: np.ndarray) -> np.ndarray:
    """
    Calculate quality flags from the L1A housekeeping macro status.

    Parameters
    ----------
    macro_status : numpy.ndarray
        Housekeeping macro status (uint32), one per housekeeping record.

    Returns
    -------
    flags : numpy.ndarray
        Quality flags (uint8), one per housekeeping record.
    """
    # Extract the macro status array and ensure its uint32
    macro_status = macro_status.astype(np.uint32)
    flags = np.full(macro_status.shape, ImapCodiceL1aFlags.NONE, dtype=np.uint8)

    # Any of bits 7 through 13 set means there is a gain test in progress.
    gain_test_mask = sum(1 << bit for bit in range(7, 14))
    gain_test_active = (macro_status & gain_test_mask) != 0
    flags[gain_test_active] |= np.uint8(ImapCodiceL1aFlags.GAIN_TEST)

    # If either bits 28 or 29 are set, this means there was active
    # mcp current mitigation.
    # TODO: Wait for Joeys instructions on bit 29. Macro status is defined as 28 bits.
    mcp_mask = sum(1 << bit for bit in range(28, 30))
    mcp_active = (macro_status & mcp_mask) != 0
    flags[mcp_active] |= np.uint8(ImapCodiceL1aFlags.MCP_CURRENT)

    logger.info(
        f"Flagged {np.sum(gain_test_active)} epochs for active gain "
        f"testing and {np.sum(mcp_active)} for active MCP current"
        f" mitigation."
    )

    return flags


def add_quality_flags_to_science_datasets(
    hk_quality_flags: np.ndarray | None,
    hk_epochs: np.ndarray,
    datasets: list[xr.Dataset],
) -> None:
    """
    Add quality flags to science datasets.

    Each science epoch's flag is the bitwise OR of the flags of all housekeeping
    records that fall within that epoch's collection period.

    Parameters
    ----------
    hk_quality_flags : numpy.ndarray or None
        Housekeeping quality flags (uint8), one per housekeeping record. If None,
        every science epoch is assigned the NONE flag.
    hk_epochs : numpy.ndarray
        Housekeeping epochs (TT-J2000 ns), one per housekeeping record.
    datasets : list[xarray.Dataset]
        Science datasets to update in place with a ``quality_flags`` variable.
    """
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_variable_attrs("codice", "l1a")
    for dataset in datasets:
        # Start with no flags set.
        default_qf: np.ndarray = np.full(
            len(dataset.epoch), ImapCodiceL1aFlags.NONE, dtype=np.uint8
        )
        if hk_quality_flags is None:
            sci_quality_flags = default_qf
        else:
            # Housekeeping products don't get quality flags.
            if "hskp" in dataset.attrs["Logical_source"]:
                continue
            # HK is sent at 1 Hz and science epochs are ~4 minutes (240 s) apart.
            # A science epoch is flagged if *any* HK record in its collection
            # period is flagged. For science epochs at 240, 480, and 720 s, HK
            # records from 240-479 s map to the epoch at 240s, etc.
            sci_epochs = dataset.epoch.values
            # Close the last collection period by appending a boundary 240 s
            # (in ns) after the last science epoch. HK records past this
            # boundary don't map to any science epoch.
            sci_epochs = np.append(sci_epochs, sci_epochs[-1] + 240 * 1e9)
            # Index of the science epoch that each HK record falls under.
            hk_to_sci_inds = (
                np.asarray(np.searchsorted(sci_epochs, hk_epochs, side="right")) - 1
            )
            # Drop HK records that fall outside every collection period.
            valid = (hk_to_sci_inds >= 0) & (hk_to_sci_inds < len(sci_epochs) - 1)
            # Each science flag is the bitwise OR of the flags of its HK records.
            # Example with science epochs at 240, 480, and 720 s:
            #   HK epochs:       230, 240, 300, 500, 800, 1000
            #   HK flags:          1,   0,   1,   0,   0,    1
            #   hk_to_sci_inds:   -1,   0,   0,   1,   2,    3  (-1 and 3 dropped)
            #   Science flags:   [1, 0, 0]
            np.bitwise_or.at(default_qf, hk_to_sci_inds[valid], hk_quality_flags[valid])
            sci_quality_flags = default_qf

        dataset["quality_flags"] = xr.DataArray(
            data=sci_quality_flags,
            name="quality_flags",
            dims="epoch",
            attrs=cdf_attrs.get_variable_attributes("quality_flags"),
        )
