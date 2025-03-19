"""
Perform CoDICE l1b processing.

This module processes CoDICE l1a files and creates L1b data products.

Notes
-----
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1b import process_codice_l1b
dataset = process_codice_l1b(l1a_file)
"""

import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_hskp_dataset(
    l1a_dataset: xr.Dataset, cdf_attrs: ImapCdfAttributes
) -> xr.Dataset:
    """
    Create an ``xarray`` dataset for the housekeeping data.

    The dataset can then be written to a CDF file.

    Parameters
    ----------
    l1a_dataset : xr.Dataset
        The L1a dataset that is being processed.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes for the dataset.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The ``xarray`` dataset containing the science data and supporting metadata.
    """
    epoch = l1a_dataset.coords["epoch"]
    l1b_dataset = xr.Dataset(
        coords={"epoch": epoch},
        attrs=cdf_attrs.get_global_attributes("imap_codice_l1b_hskp"),
    )
    for variable_name in l1a_dataset:
        # Get the data array from the L1a data product
        values = l1a_dataset[variable_name].values

        # Convert data array to "rates"
        # TODO: For SIT-3, just convert value to float. Revisit after SIT-3.
        variable_data_arr = values.astype(float)

        # TODO: Change 'TBD' catdesc and fieldname
        # Once packet definition files are re-generated, can get this info from
        # something like this:
        #    for key, value in (packet.header | packet.data).items():
        #      fieldname = value.short_description
        #      catdesc = value.short_description
        # I am holding off making this change until I acquire updated housekeeping
        # packets/validation data that match the latest telemetry definitions
        attrs = cdf_attrs.get_variable_attributes("codice_support_attrs")
        attrs["CATDESC"] = "TBD"
        attrs["DEPEND_0"] = "epoch"
        attrs["FIELDNAM"] = "TBD"
        attrs["LABLAXIS"] = variable_name

        # Put the new data array into the dataset
        l1b_dataset[variable_name] = xr.DataArray(
            variable_data_arr,
            name=variable_name,
            dims=["epoch"],
            attrs=attrs,
        )

    return l1b_dataset


def create_science_dataset(
    l1a_dataset: xr.Dataset, cdf_attrs: ImapCdfAttributes, dataset_name: str
) -> xr.Dataset:
    """
    Create an ``xarray`` dataset for the science data.

    The dataset can then be written to a CDF file.

    Parameters
    ----------
    l1a_dataset : xr.Dataset
        The L1a dataset that is being processed.
    cdf_attrs : ImapCdfAttributes
        The CDF attributes for the dataset.
    dataset_name : str
        The name that is used to construct the data variable name and reference
        the CDF attributes (e.g. ``imap_codice_l1b_hi_omni``).

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The ``xarray`` dataset containing the science data and supporting metadata.
    """
    # Retrieve the coordinates from the l1a dataset
    epoch = l1a_dataset.coords["epoch"]
    energy = l1a_dataset.coords["energy"]
    energy_label = l1a_dataset.coords["energy_label"]

    # Create empty l1b dataset
    l1b_dataset = xr.Dataset(
        coords={"epoch": epoch, "energy": energy, "energy_label": energy_label},
        attrs=cdf_attrs.get_global_attributes(dataset_name),
    )

    # Get the data variables from l1a dataset
    for variable_name in l1a_dataset:
        if variable_name == "esa_sweep_values":
            values = l1a_dataset["esa_sweep_values"]
            l1b_dataset["esa_sweep_values"] = xr.DataArray(
                values,
                dims=["energy"],
                attrs=cdf_attrs.get_variable_attributes("esa_sweep_attrs"),
            )

        elif variable_name == "acquisition_times":
            values = l1a_dataset["acquisition_times"]
            l1b_dataset["acquisition_times"] = xr.DataArray(
                values,
                dims=["energy"],
                attrs=cdf_attrs.get_variable_attributes("acquisition_times_attrs"),
            )

        else:
            # Get the data array from the L1a data product
            values = l1a_dataset[variable_name].values

            # Convert data array to "rates"
            # TODO: For SIT-3, just convert value to float. Revisit after SIT-3.
            variable_data_arr = values.astype(float)

            # Put the new data array into the dataset
            cdf_attrs_key = (
                f"{dataset_name.split('imap_codice_l1b_')[-1]}-{variable_name}"
            )
            l1b_dataset[variable_name] = xr.DataArray(
                variable_data_arr,
                name=variable_name,
                dims=["epoch", "energy"],
                attrs=cdf_attrs.get_variable_attributes(cdf_attrs_key),
            )

    return l1b_dataset


def process_codice_l1b(file_path: Path, data_version: str) -> xr.Dataset:
    """
    Will process CoDICE l1a data to create l1b data products.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the CoDICE L1a file to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    print(f"\nProcessing {file_path}")

    # Open the l1a file
    l1a_dataset = load_cdf(file_path)
    print(l1a_dataset)

    # Get the L1b CDF attributes
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1b")
    cdf_attrs.add_global_attribute("Data_version", data_version)
    l1b_global_attrs = cdf_attrs.get_global_attributes("imap_codice_l1b_lo-sw-species")

    # Use the dataset name as a way to distinguish between data products
    dataset_name = l1a_dataset.attrs["Logical_source"].replace("_l1a_", "_l1b_")

    # Use the L1a data product as a starting point for L1b
    l1b_dataset = l1a_dataset.copy()

    # Update the global attributes
    l1b_dataset.attrs = l1b_global_attrs

    #

    #
    # if "hskp" in dataset_name:
    #     l1b_dataset = create_hskp_dataset(l1a_dataset, cdf_attrs)
    #
    # else:
    #     l1b_dataset = create_science_dataset(l1a_dataset, cdf_attrs, dataset_name)
    #
    # # Write the dataset to CDF
    # logger.info(f"\nFinal data product:\n{l1b_dataset}\n")

    return l1b_dataset


if __name__ == "__main__":
    from imap_processing import imap_module_directory

    TEST_DATA_PATH = imap_module_directory / "tests" / "codice" / "data"
    file_path = (
        imap_module_directory
        / "codice"
        / "data"
        / "imap"
        / "codice"
        / "l1a"
        / "2024"
        / "11"
        / "imap_codice_l1a_lo-sw-species_20241110_v001.cdf"
    )

    dataset = process_codice_l1b(file_path, "001")

    print(dataset)
