"""IMAP-HI L1B processing module."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

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
    l1a_cdf_path : Path
    Path to L1A CDF file.

    Returns
    -------
    processed_data : xarray.Dataset
        Processed xarray dataset
    """
    logger.info(f"Running Hi L1B processing on file: {l1a_cdf_path.name}")
    l1a_dataset = load_cdf(l1a_cdf_path)
    logical_source_parts = l1a_dataset["logical_source"].split("_")
    # TODO: apid is not currently stored in all L1A data but should be.
    #    Use apid to determine what L1B processing function to call

    # Housekeeping processing
    if logical_source_parts[-1].endswith("hk"):
        # if packet_enum in (HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK):
        packet_enum = HIAPID(l1a_dataset["pkt_apid"].data[0])
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
    elif logical_source_parts[-1].endswith("de"):
        l1b_dataset = annotate_direct_events(l1a_dataset)


def annotate_direct_events(l1a_dataset):
    """
    Perform Hi L1B processing on direct event data.

    Parameters
    ----------
    l1a_dataset: xarray.Dataset
        L1A direct event data.

    Returns
    -------
    xarray.Dataset
        L1B direct event data.
    """
    coords = dict()
    # epoch contains a single element
    # TODO: What value to use. SPDF says times should indicate the center
    #    value of the time bin.
    coords["epoch"] = xr.DataArray(
        np.empty(1, dtype="datetime64[ns]"),
        name="epoch",
        dims=["epoch"],
        # attrs=ConstantCoordinates.EPOCH,
    )
