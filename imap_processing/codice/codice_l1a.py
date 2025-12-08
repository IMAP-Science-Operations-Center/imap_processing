"""CoDICE L1A processing functions."""

import logging

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
from imap_processing.codice.codice_l1a_lo_angular import l1a_lo_angular
from imap_processing.codice.codice_l1a_lo_counters_aggregated import (
    l1a_lo_counters_aggregated,
)
from imap_processing.codice.codice_l1a_lo_counters_singles import (
    l1a_lo_counters_singles,
)
from imap_processing.codice.codice_l1a_lo_priority import l1a_lo_priority
from imap_processing.codice.codice_l1a_lo_species import l1a_lo_species
from imap_processing.codice.utils import (
    CODICEAPID,
)
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
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_AGGREGATED:
            logger.info("Processing Hi Counters aggregated")
            datasets.append(
                l1a_hi_counters_aggregated(datasets_by_apid[apid], lut_file)
            )
        elif apid == CODICEAPID.COD_HI_INST_COUNTS_SINGLES:
            logger.info("Processing Hi Counters singles")
            datasets.append(l1a_hi_counters_singles(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_LO_INST_COUNTS_AGGREGATED:
            logger.info("Processing Lo Counters aggregated")
            datasets.append(
                l1a_lo_counters_aggregated(datasets_by_apid[apid], lut_file)
            )
        elif apid == CODICEAPID.COD_LO_INST_COUNTS_SINGLES:
            logger.info("Processing Lo Counters singles")
            datasets.append(l1a_lo_counters_singles(datasets_by_apid[apid], lut_file))
        elif apid == CODICEAPID.COD_NHK:
            logger.info("Processing l1a housekeeping data")
            cdf_attrs = ImapCdfAttributes()
            cdf_attrs.add_instrument_global_attrs("codice")
            l1a_ds = datasets_by_apid[apid]
            l1a_ds.attrs.update(cdf_attrs.get_global_attributes("imap_codice_l1a_hskp"))
            datasets.append(l1a_ds)

            # l1b processing need to re-run packet file to datasets to do the
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

    return datasets
