"""Calculates ULTRA L1b."""

import json
import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def create_dataset(data_dict, name):
    """
    Create xarray for L1b data.

    Parameters
    ----------
    data_dict: : dict
        L1b data dictionary.

    Returns
    -------
    dataset : xarray.Dataset
        xarray.Dataset
    """
    # TODO: this will change to actual l1b data dictionary
    # For now we are using it for retrieving the epoch.
    dataset = data_dict["imap_ultra_l1a_45de"]

    # Load metadata from the metadata file
    with open(Path(__file__).parent.parent / "ultra_metadata_example.json") as f:
        metadata = json.loads(f.read())[name]

    epoch = dataset[0].coords["epoch"]

    annotated_de_attrs = metadata["dataset_attrs"]

    dataset = xr.Dataset(
        coords={"epoch": epoch.astype("datetime64[ns]")},
        attrs=annotated_de_attrs,
    )

    dataset["x_front"] = xr.DataArray(
        np.zeros(len(epoch), dtype=np.float32),
        dims=["epoch"],
        attrs=metadata["x_front"],
    )

    return dataset


def ultra_l1b(data_dict: dict):
    """
    Process ULTRA L1A data into L1B CDF files at output_filepath.

    Parameters
    ----------
    data_dict: : dict
        Dependencies dictionary.

    Returns
    -------
    output_datasets : list of xarray.Dataset
        List of xarray.Dataset
    """
    output_datasets = []
    instrument_id = int(re.search(r"45|90", next(iter(data_dict.keys()))).group())

    # TODO: Add the other l1b products here.
    l1b_products = [
        f"imap_ultra_l1b_{instrument_id}annotated-de",
        # f"imap_ultra_l1b_{instrument_id}extended-spin",
        # f"imap_ultra_l1b_{instrument_id}culling-mask",
        # f"imap_ultra_l1b_{instrument_id}badtimes"
    ]

    for name in l1b_products:
        # TODO: perform l1b calculations here.
        # For now we are using the L1A data (data_dict) as a placeholder.
        dataset = create_dataset(data_dict, name)
        output_datasets.append(dataset)

    return output_datasets
