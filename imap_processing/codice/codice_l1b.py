"""
Perform CoDICE l1b processing.

This module processes CoDICE l1a files and creates L1a data products.

Notes
-----
from imap_processing.codice.codice_l0 import decom_packets
from imap_processing.codice.codice_l1b import process_codice_l1b
dataset = process_codice_l1b(l1a_file)
"""

import logging

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Fix ISTP compliance issues (revealed in SKTEditor)


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


def process_codice_l1b(l1a_dataset: xr.Dataset, data_version: str) -> xr.Dataset:
    """
    Will process CoDICE l1a data to create l1b data products.

    Parameters
    ----------
    l1a_dataset : xarray.Dataset
        CoDICE L1a dataset to process.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    l1b_dataset : xarray.Dataset
        The``xarray`` dataset containing the science data and supporting metadata.
    """
    logger.info(f"\nProcessing {l1a_dataset.attrs['Logical_source']}.")

    # Start constructing l1b dataset
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1b")
    cdf_attrs.add_global_attribute("Data_version", data_version)

    dataset_name = (
        l1a_dataset.attrs["Logical_source"].replace("-", "_").replace("l1a", "l1b")
    )

    if "hskp" in dataset_name:
        l1b_dataset = create_hskp_dataset(l1a_dataset, cdf_attrs)

    else:
        l1b_dataset = create_science_dataset(l1a_dataset, cdf_attrs, dataset_name)

    # Write the dataset to CDF
    logger.info(f"\nFinal data product:\n{l1b_dataset}\n")

    return l1b_dataset
