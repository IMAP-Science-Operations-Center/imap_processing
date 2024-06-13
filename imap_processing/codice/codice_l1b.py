"""Perform CoDICE l1b processing.

This module processes CoDICE l1a files and creates L1a data products.

Use
---

    from imap_processing.codice.codice_l0 import decom_packets
    from imap_processing.codice.codice_l1b import process_codice_l1b
    dataset = process_codice_l1b(l1a_file)
"""

import logging
from pathlib import Path

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def process_codice_l1b(file_path: Path, data_version: str) -> xr.Dataaset:
    """Process CoDICE l1a data to create l1b data products.

    Parameters
    ----------
    file_path : pathlib.Path | str
        Path to the CoDICE L1a file to process
    data_version : str
        Version of the data product being created

    Returns
    -------
    dataset : xarray.Dataset
        ``xarray`` dataset containing the science data and supporting metadata
    """
    logger.info(f"\nProcessing {file_path.name} file.")

    # Load the L1a CDF
    l1a_dataset = load_cdf(file_path)

    # Start constructing l1b dataset
    cdf_attrs = ImapCdfAttributes()
    cdf_attrs.add_instrument_global_attrs("codice")
    cdf_attrs.add_instrument_variable_attrs("codice", "l1b")

    dataset_name = (
        l1a_dataset.attrs["Logical_source"].replace("-", "_").replace("l1a", "l1b")
    )

    if "hskp" in dataset_name:
        epoch = l1a_dataset.coords["epoch"]
        l1b_dataset = xr.Dataset(
            coords={"epoch": epoch},
            attrs=cdf_attrs.get_global_attributes(dataset_name),
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
            attrs = cdf_attrs.variable_attributes["codice_support_attrs"]
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

    else:
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

    # Write the dataset to CDF
    logger.info(f"\nFinal data product:\n{l1b_dataset}\n")
    l1b_dataset.attrs["Data_version"] = data_version
    l1b_dataset.attrs["cdf_filename"] = write_cdf(l1b_dataset)
    logger.info(f"\tCreated CDF file: {l1b_dataset.cdf_filename}")

    return l1b_dataset
