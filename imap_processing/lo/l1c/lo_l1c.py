"""IMAP-Lo L1C Data Processing."""

from collections import namedtuple
from dataclasses import Field
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_j2000ns


def lo_l1c(dependencies: dict, data_version: str) -> list[Path]:
    """
    Will process IMAP-Lo L1B data into L1C CDF data products.

    Parameters
    ----------
    dependencies : dict
        Dictionary of datasets needed for L1C data product creation in xarray Datasets.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1c")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # if the dependencies are used to create Annotated Direct Events
    if "imap_lo_l1b_de" in dependencies:
        logical_source = "imap_lo_l1c_pset"
        # TODO: TEMPORARY. Need to update to use the L1C data class once that exists
        #  and I have sample data.
        data_field_tup = namedtuple("data_field_tup", ["name"])
        data_fields = [
            data_field_tup("POINTING_START"),
            data_field_tup("POINTING_END"),
            data_field_tup("MODE"),
            data_field_tup("PIVOT_ANGLE"),
            data_field_tup("TRIPLES_COUNTS"),
            data_field_tup("TRIPLES_RATES"),
            data_field_tup("DOUBLES_COUNTS"),
            data_field_tup("DOUBLES_RATES"),
            data_field_tup("HYDROGEN_COUNTS"),
            data_field_tup("HYDROGEN_RATES"),
            data_field_tup("OXYGEN_COUNTS"),
            data_field_tup("OXYGEN_RATES"),
            data_field_tup("EXPOSURE_TIME"),
        ]

    dataset: list[Path] = create_datasets(attr_mgr, logical_source, data_fields)  # type: ignore[arg-type]
    # TODO Remove once data_fields input is removed from create_datasets
    return dataset


# TODO: This is going to work differently when I sample data.
#  The data_fields input is temporary.
def create_datasets(
    attr_mgr: ImapCdfAttributes, logical_source: str, data_fields: list[Field]
) -> xr.Dataset:
    """
    Create a dataset using the populated data classes.

    Parameters
    ----------
    attr_mgr : ImapCdfAttributes
        Attribute manager used to get the data product field's attributes.
    logical_source : str
        The logical source of the data product that's being created.
    data_fields : list[dataclasses.Field]
        List of Fields for data classes.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all data product fields in xr.DataArray.
    """
    # TODO: Once L1B DE processing is implemented using the spin packet
    #  and relative L1A DE time to calculate the absolute DE time,
    #  this epoch conversion will go away and the time in the DE dataclass
    #  can be used direction
    epoch_converted_time = [met_to_j2000ns(1)]

    # Create a data array for the epoch time
    # TODO: might need to update the attrs to use new YAML file
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    if logical_source == "imap_lo_l1c_pset":
        esa_step = xr.DataArray(
            data=[1, 2, 3, 4, 5, 6, 7],
            name="esa_step",
            dims=["esa_step"],
            attrs=attr_mgr.get_variable_attributes("esa_step"),
        )
        pointing_bins = xr.DataArray(
            data=np.arange(3600),
            name="pointing_bins",
            dims=["pointing_bins"],
            attrs=attr_mgr.get_variable_attributes("pointing_bins"),
        )

        esa_step_label = xr.DataArray(
            esa_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )
        pointing_bins_label = xr.DataArray(
            pointing_bins.values.astype(str),
            name="pointing_bins_label",
            dims=["pointing_bins_label"],
            attrs=attr_mgr.get_variable_attributes("pointing_bins_label"),
        )
        dataset = xr.Dataset(
            coords={
                "epoch": epoch_time,
                "pointing_bins": pointing_bins,
                "pointing_bins_label": pointing_bins_label,
                "esa_step": esa_step,
                "esa_step_label": esa_step_label,
            },
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    # Loop through the data fields that were pulled from the
    # data class. These should match the field names given
    # to each field in the YAML attribute file
    for data_field in data_fields:
        field = data_field.name.lower()
        # Create a list of all the dimensions using the DEPEND_I keys in the
        # YAML attributes
        dims = [
            value
            for key, value in attr_mgr.get_variable_attributes(field).items()
            if "DEPEND" in key
        ]

        # Create a data array for the current field and add it to the dataset
        # TODO: TEMPORARY. need to update to use l1b data once that's available.
        if field in ["pointing_start", "pointing_end", "mode", "pivot_angle"]:
            dataset[field] = xr.DataArray(
                data=[1],
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        # TODO: This is temporary.
        #  The data type will be set in the data class when that's created
        elif field == "exposure_time":
            dataset[field] = xr.DataArray(
                data=np.ones((1, 7), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

        elif "rate" in field:
            dataset[field] = xr.DataArray(
                data=np.ones((1, 3600, 7), dtype=np.float16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )
        else:
            dataset[field] = xr.DataArray(
                data=np.ones((1, 3600, 7), dtype=np.int16),
                dims=dims,
                attrs=attr_mgr.get_variable_attributes(field),
            )

    return dataset
