"""Calculate ULTRA L1b."""

import logging
import re

import xarray as xr

from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.de import calculate_de
from imap_processing.ultra.l1b.extendedspin import calculate_extendedspin
from imap_processing.ultra.l1b.goodtimes import calculate_goodtimes

logger = logging.getLogger(__name__)


def ultra_l1b(data_dict: dict, ancillary_files: dict) -> list[xr.Dataset]:
    """
    Will process ULTRA L1A data into L1B CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.
    ancillary_files : dict
        Ancillary files.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.

    Notes
    -----
    General flow:
    1. l1a data products are created (upstream to this code)
    2. l1b de is created here and dropped in s3 kicking off processing again
    3. l1b extended, goodtimes, badtimes created here
    """
    output_datasets = []
    # Account for possibility of having 45 and 90 in dictionary.
    for instrument_id in [45, 90]:
        # Find any de product if it is in the data_dict
        l1a_de_products = [
            name
            for name in data_dict.keys()
            if re.search(rf"^imap_ultra_l1a_{instrument_id}sensor.*-de$", name)
        ]
        # L1b de data will be created if L1a de data is available
        # Including priority de products
        if l1a_de_products:
            l1a_de_product = l1a_de_products[0]
            if len(l1a_de_products) > 1:
                raise ValueError(
                    f"Multiple L1a de products found for instrument {instrument_id}. "
                    f"Expected only one but found {len(l1a_de_products)}: "
                    f"{l1a_de_products}"
                )
            l1b_de_product = l1a_de_product.replace("l1a", "l1b")
            de_dataset = calculate_de(
                data_dict[l1a_de_product],
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-aux"],
                l1b_de_product,
                ancillary_files,
            )
            output_datasets.append(de_dataset)
        # L1b extended data will be created if L1a hk, rates,
        # aux, params, and l1b de data are available
        elif (
            f"imap_ultra_l1b_{instrument_id}sensor-de" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
            and f"imap_ultra_l1a_{instrument_id}sensor-params" in data_dict
            and f"imap_ultra_l1b_{instrument_id}sensor-status" in data_dict
        ):
            extendedspin_dataset = calculate_extendedspin(
                {
                    f"imap_ultra_l1a_{instrument_id}sensor-aux": data_dict[
                        f"imap_ultra_l1a_{instrument_id}sensor-aux"
                    ],
                    f"imap_ultra_l1a_{instrument_id}sensor-params": data_dict[
                        f"imap_ultra_l1a_{instrument_id}sensor-params"
                    ],
                    f"imap_ultra_l1a_{instrument_id}sensor-rates": data_dict[
                        f"imap_ultra_l1a_{instrument_id}sensor-rates"
                    ],
                    f"imap_ultra_l1b_{instrument_id}sensor-de": data_dict[
                        f"imap_ultra_l1b_{instrument_id}sensor-de"
                    ],
                    f"imap_ultra_l1b_{instrument_id}sensor-status": data_dict[
                        f"imap_ultra_l1b_{instrument_id}sensor-status"
                    ],
                },
                f"imap_ultra_l1b_{instrument_id}sensor-extendedspin",
                instrument_id,
            )
            output_datasets.append(extendedspin_dataset)
        elif (
            f"imap_ultra_l1b_{instrument_id}sensor-extendedspin" in data_dict
            and f"imap_ultra_l1b_{instrument_id}sensor-goodtimes" in data_dict
        ):
            badtimes_dataset = calculate_badtimes(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-goodtimes"][
                    "spin_number"
                ].values,
                f"imap_ultra_l1b_{instrument_id}sensor-badtimes",
            )
            output_datasets.append(badtimes_dataset)
        elif f"imap_ultra_l1b_{instrument_id}sensor-extendedspin" in data_dict:
            goodtimes_dataset = calculate_goodtimes(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
                f"imap_ultra_l1b_{instrument_id}sensor-goodtimes",
            )
            output_datasets.append(goodtimes_dataset)
    if not output_datasets:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
