"""Calculate ULTRA L1b."""

import xarray as xr

from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask
from imap_processing.ultra.l1b.de import calculate_de
from imap_processing.ultra.l1b.extendedspin import calculate_extendedspin


def ultra_l1b(data_dict: dict, data_version: str) -> list[xr.Dataset]:
    """
    Will process ULTRA L1A data into L1B CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    output_datasets : list[xarray.Dataset]
        List of xarray.Dataset.
    """
    output_datasets = []
    instrument_id = 45 if any("45" in key for key in data_dict.keys()) else 90

    if f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict:
        extendedspin_dataset = calculate_extendedspin(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-rates"],
            f"imap_ultra_l1b_{instrument_id}sensor-extendedspin",
        )
        # TODO: move these to use ImapCdfAttributes().add_global_attribute()
        extendedspin_dataset.attrs["Data_version"] = data_version

        cullingmask_dataset = calculate_cullingmask(
            extendedspin_dataset, f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"
        )
        cullingmask_dataset.attrs["Data_version"] = data_version

        badtimes_dataset = calculate_badtimes(
            extendedspin_dataset, f"imap_ultra_l1b_{instrument_id}sensor-badtimes"
        )
        badtimes_dataset.attrs["Data_version"] = data_version

        output_datasets.extend(
            [extendedspin_dataset, cullingmask_dataset, badtimes_dataset]
        )
    elif (
        f"imap_ultra_l1a_{instrument_id}sensor-aux" in data_dict
        and f"imap_ultra_l1a_{instrument_id}sensor-de" in data_dict
    ):
        de_dataset = calculate_de(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-de"],
            f"imap_ultra_l1b_{instrument_id}sensor-de",
        )
        de_dataset.attrs["Data_version"] = data_version

        output_datasets.append(de_dataset)
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
