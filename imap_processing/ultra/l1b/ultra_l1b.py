"""Calculate ULTRA L1b."""

from imap_processing.ultra.l1b.badtimes import calculate_badtimes
from imap_processing.ultra.l1b.cullingmask import calculate_cullingmask
from imap_processing.ultra.l1b.de import calculate_de
from imap_processing.ultra.l1b.extendedspin import calculate_extendedspin


def ultra_l1b(data_dict: dict):
    """
    Will process ULTRA L1A data into L1B CDF files at output_filepath.

    Parameters
    ----------
    data_dict : dict
        The data itself and its dependent data.

    Returns
    -------
    list
        List of xarray.Dataset.
    """
    output_datasets = []
    instrument_id = 45 if any("45" in key for key in data_dict.keys()) else 90

    if f"imap_ultra_l1a_{instrument_id}sensor-rates" in data_dict:
        extendedspin_dataset = calculate_extendedspin(
            data_dict[f"imap_ultra_l1a_{instrument_id}sensor-rates"],
            f"imap_ultra_l1b_{instrument_id}sensor-extendedspin",
        )

        cullingmask_dataset = calculate_cullingmask(
            extendedspin_dataset, f"imap_ultra_l1b_{instrument_id}sensor-cullingmask"
        )

        badtimes_dataset = calculate_badtimes(
            extendedspin_dataset, f"imap_ultra_l1b_{instrument_id}sensor-badtimes"
        )

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

        output_datasets.append(de_dataset)
    else:
        raise ValueError("Data dictionary does not contain the expected keys.")

    return output_datasets
