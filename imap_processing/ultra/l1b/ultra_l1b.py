"""Calculate ULTRA L1b."""

import xarray as xr

from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask
from imap_processing.ultra.l1b.de import calculate_de
from imap_processing.ultra.l1b.extendedspin import calculate_extendedspin


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
    3. l1b extended, culling, badtimes created here
    """
    output_datasets = []

    # Account for possibility of having 45 and 90 in dictionary.
    for instrument_id in [45, 90]:
        # L1b de data will be created if L1a de data is available
        if f"imap_ultra_l1a_{instrument_id}sensor-de" in data_dict:
            de_dataset = calculate_de(
                data_dict[f"imap_ultra_l1a_{instrument_id}sensor-de"],
                f"imap_ultra_l1b_{instrument_id}sensor-de",
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
                },
                f"imap_ultra_l1b_{instrument_id}sensor-extendedspin",
                instrument_id,
            )
            output_datasets.append(extendedspin_dataset)
        elif (
            f"imap_ultra_l1b_{instrument_id}sensor-extendedspin" in data_dict
            and f"imap_ultra_l1b_{instrument_id}sensor-cullingmask" in data_dict
        ):
            badtimes_dataset = calculate_badtimes(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"][
                    "spin_number"
                ].values,
                f"imap_ultra_l1b_{instrument_id}sensor-badtimes",
            )
            output_datasets.append(badtimes_dataset)
        elif f"imap_ultra_l1b_{instrument_id}sensor-extendedspin" in data_dict:
            cullingmask_dataset = calculate_cullingmask(
                data_dict[f"imap_ultra_l1b_{instrument_id}sensor-extendedspin"],
                f"imap_ultra_l1b_{instrument_id}sensor-cullingmask",
            )
            output_datasets.append(cullingmask_dataset)
    if not output_datasets:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
