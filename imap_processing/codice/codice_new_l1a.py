"""CoDICE L1A processing functions."""

import logging

import xarray as xr
from imap_data_access import ProcessingInputCollection

from imap_processing import imap_module_directory
from imap_processing.codice.codice_l1a_de import l1a_direct_event
from imap_processing.codice.codice_l1a_hi_omni import l1a_hi_omni
from imap_processing.codice.codice_l1a_hi_priority import l1a_hi_priority
from imap_processing.codice.codice_l1a_hi_sectored import l1a_hi_sectored
from imap_processing.codice.codice_l1a_lo_angular import l1a_lo_angular
from imap_processing.codice.codice_l1a_lo_priority import l1a_lo_priority
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.utils import (
    CODICEAPID,
)
from imap_processing.utils import packet_file_to_datasets

logger = logging.getLogger(__name__)


def process_l1a(dependency: ProcessingInputCollection) -> list[xr.Dataset]:
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

    xtce_file = (
        imap_module_directory / "codice/packet_definitions/codice_packet_definition.xml"
    )
    # Decom packet
    datasets_by_apid = packet_file_to_datasets(
        science_file,
        xtce_file,
    )

    datasets = []
    for apid in datasets_by_apid:
        if apid not in [CODICEAPID.COD_LO_PHA, CODICEAPID.COD_HI_PHA]:
            # Get LUT file. Direct events do not need LUT
            lut_file = dependency.get_file_paths(descriptor="l1a-sci-lut")
            lut_file = lut_file[0]

        if apid == CODICEAPID.COD_LO_SW_SPECIES_COUNTS:
            logger.info("Processing Lo SW Species Counts")
            datasets.append(l1a_lo_species(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_LO_NSW_SPECIES_COUNTS:
            logger.info("Processing Lo NSW Species Counts")
            datasets.append(l1a_lo_species(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_LO_SW_ANGULAR_COUNTS:
            logger.info("Processing Lo SW Angular Counts")
            datasets.append(l1a_lo_angular(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS:
            logger.info("Processing Lo NSW Angular Counts")
            datasets.append(l1a_lo_angular(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS:
            datasets.append(l1a_hi_omni(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_HI_SECT_SPECIES_COUNTS:
            logger.info("Processing Hi Sectored Species Counts")
            datasets.append(l1a_hi_sectored(datasets_by_apid[apid], lut_file))
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
            datasets.append(l1a_lo_priority(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_PRIORITIES:
            logger.info("Processing Hi Priority Counts")
            datasets.append(l1a_hi_priority(datasets_by_apid[apid], lut_file))
    return datasets
