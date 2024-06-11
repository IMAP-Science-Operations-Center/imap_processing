"""IMAP-HI L1B processing module."""

import logging
from pathlib import Path

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.hi.hi_cdf_attrs import hi_hk_l1b_global_attrs
from imap_processing.hi.utils import HIAPID
from imap_processing.utils import convert_raw_to_eu

logger = logging.getLogger(__name__)


def hi_l1b(l1a_cdf_path: Path):
    """
    High level IMAP-HI L1B processing function.

    Parameters
    ----------
    l1a_cdf_path : pathlib.Path
        Path to L1A CDF file.

    Returns
    -------
    processed_data : xarray.Dataset
        Processed xarray dataset
    """
    l1a_dataset = load_cdf(l1a_cdf_path)
    packet_enum = HIAPID(l1a_dataset["pkt_apid"].data[0])

    # Housekeeping processing
    if packet_enum in (HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK):
        logger.info(f"Running Hi L1B processing on file: {l1a_cdf_path.name}")
        conversion_table_path = (
            imap_module_directory / "hi" / "l1b" / "hi_eng_unit_convert_table.csv"
        )
        l1b_dataset = convert_raw_to_eu(
            l1a_dataset,
            conversion_table_path=conversion_table_path,
            packet_name=packet_enum.name,
            comment="#",
            converters={"mnemonic": str.lower},
        )
        l1b_dataset.attrs.update(hi_hk_l1b_global_attrs.output())
        return l1b_dataset
    elif packet_enum in (HIAPID.H45_SCI_DE, HIAPID.H90_SCI_DE):
        raise NotImplementedError(
            f"L1B processing not implemented for file: {l1a_cdf_path.name}"
        )
