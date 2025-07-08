"""
Perform IDEX L2c Processing.

Examples
--------
.. code-block:: python
    from imap_processing.idex.idex_l1a import PacketParser
    from imap_processing.idex.idex_l1b import idex_l1b
    from imap_processing.idex.idex_l2a import idex_l2a
    from imap_processing.idex.idex_l2b import idex_l2b
    from imap_processing.cdf.utils import write_cdf

    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20231214_v001.pkts"
    l1a_data = PacketParser(l0_file)
    l1b_data = idex_l1b(l1a_data)
    l2a_data = idex_l2a(l1b_data)
    l2b_data = idex_l2b(l2a_data)
    write_cdf(l2b_data)
"""

import logging

import xarray as xr

from imap_processing.ena_maps.ena_maps import SkyTilingType
from imap_processing.idex.idex_constants import (
    IDEX_EVENT_REFERENCE_FRAME,
    IDEX_SPACING_DEG,
)
from imap_processing.idex.idex_utils import get_idex_attrs, setup_dataset

logger = logging.getLogger(__name__)


def idex_l2c(l2b_datasets: list[xr.Dataset]) -> xr.Dataset:
    """
    Will process IDEX l2b data to create l2c data products.

    Parameters
    ----------
    l2b_datasets : list[xarray.Dataset]
        IDEX L2b datasets.

    Returns
    -------
    l2b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info("Running IDEX L2C processing")
    # create the attribute manager for this data level
    idex_attrs = get_idex_attrs("l2c")
    # Concat the list of l2b datasets into a single dataset
    # Only concat the variables that have "epoch" as a dimension
    l2b_dataset = xr.concat(
        l2b_datasets, "epoch", data_vars="minimal", coords="minimal"
    )

    arrays_to_copy = [
        "counts_by_charge_map",
        "counts_by_mass_map",
        "rate_by_charge_map",
        "rate_by_mass_map",
        "epoch",
        "impact_day_of_year",
        "impact_charge_bins",
        "mass_bins",
        "charge_labels",
        "mass_labels",
        "rectangular_lon_pixel_label",
        "rectangular_lat_pixel_label",
    ]

    l2c_dataset = setup_dataset(l2b_dataset, arrays_to_copy, idex_attrs)

    # Add map attributes
    map_attrs = {
        "sky_tiling_type": SkyTilingType.RECTANGULAR.value,
        "Spacing_degrees": str(IDEX_SPACING_DEG),
        "Spice_reference_frame": IDEX_EVENT_REFERENCE_FRAME.name,
    } | idex_attrs.get_global_attributes("imap_idex_l2c_sci-rectangular")

    l2c_dataset.attrs.update(map_attrs)
    logger.info("IDEX L2C science data processing completed.")
    return l2c_dataset
