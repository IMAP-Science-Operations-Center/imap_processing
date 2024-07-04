"""IMAP-Lo L1A Data Processing."""

from collections import namedtuple
from dataclasses import Field, fields
from pathlib import Path
from typing import Any

import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.lo.l0.data_classes.science_direct_events import ScienceDirectEvents


def lo_l1a(dependency: Path, data_version: str) -> list[Path]:
    """
    Will process IMAP-Lo L0 data into L1A CDF data products.

    Parameters
    ----------
    dependency : dict
        Dependency file needed for data product creation.
        Should always be only one for L1A.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    # TODO: decom the CCSDS file

    # TODO: sort the decommed packets by SHCOARSE

    # TODO: group the packets by APID

    # TODO: Loop through packets
    # TODO: Inside Loop: Check for each APID and Initialize data
    #  class object with packet contents, add to list

    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    # TODO: replace with real processing when sample data is available
    # Temporary code until I get sample data. The fields will still be pulled
    # from the data classes, but the data class will be determined by the
    # APID in the decommed file and the data class will first be populated
    # with the packet data.
    packet_file_name = dependency.name
    if "_de_" in packet_file_name:
        data_fields = fields(ScienceDirectEvents)
        logical_source = "imap_lo_l1a_de"
    elif "_spin_" in packet_file_name:
        # Spin data class doesn't exist yet. fake the fields() output
        data_field_tup = namedtuple("data_field_tup", ["name"])
        data_fields = [  # type: ignore[assignment]
            # TODO, The errors are because the fake data is not in the correct format.
            data_field_tup("NUM_COMPLETED"),
            data_field_tup("ACQ_END"),
            data_field_tup("SPIN_SECONDS"),
            data_field_tup("SPIN_SUBSECONDS"),
            data_field_tup("ESA_P_DAC"),
            data_field_tup("ESA_N_DAC"),
            data_field_tup("VAL_PERIOD"),
            data_field_tup("VAL_SPIN"),
            data_field_tup("SOURCE"),
        ]
        logical_source = "imap_lo_l1a_spin"

    dataset: list[Path] = create_datasets(attr_mgr, logical_source, data_fields)

    return dataset


# TODO: This is going to work differently when I sample data.
#  The data_fields input is temporary.
def create_datasets(
    attr_mgr: ImapCdfAttributes,
    logical_source: str,
    data_fields: tuple[Field[Any], ...],
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
        Dataset with all data product fields in xarray.DataArray.
    """
    # Convert each packet's spacecraft time to an absolute epoch time
    # TODO: replace temp hardcoded values with packet values
    epoch_converted_time = met_to_j2000ns([0, 1, 2])

    # Create a data array for the poch time
    # TODO: might need to update the attrs to use new YAML file
    epoch_time = xr.DataArray(
        data=epoch_converted_time,
        name="epoch",
        dims=["epoch"],
        attrs=attr_mgr.get_variable_attributes("epoch"),
    )

    # Set any extra coordinates and initialize the dataset
    # based on the L1A data product that will be created
    # by the current packets
    if logical_source == "imap_lo_l1a_spin":
        # The spin packet contains a variable number of spins in each
        # packet. To avoid needing to use a 2-dimensional array to
        # store the spin fields, epoch will not be used for many
        # of the fields. They will instead use the coordinate `spin`
        # which is an index of each spin in the pointing (0 to num spins in pointing)
        spin = xr.DataArray(
            data=[0, 1, 2],
            name="spin",
            dims=["spin"],
            attrs=attr_mgr.get_variable_attributes("spin"),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "spin": spin},
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    elif logical_source == "imap_lo_l1a_de":
        # The de packet contains a variable number of events in each
        # packet. To avoid needing to use a 2-dimensional array to
        # store the de fields, epoch will not be used for many
        # of the fields and they will instead use the coordinate `direct_events`
        # which is an index of each event in the pointing (0 to num de in pointing)
        direct_events = xr.DataArray(
            data=[0, 1, 2],
            name="direct_events",
            dims=["direct_events"],
            attrs=attr_mgr.get_variable_attributes("direct_events"),
        )

        dataset = xr.Dataset(
            coords={"epoch": epoch_time, "direct_events": direct_events},
            attrs=attr_mgr.get_global_attributes(logical_source),
        )

    # Loop through the data fields that were pulled from the
    # data class. These should match the field names given
    # to each field in the YAML attribute file
    for data_field in data_fields:
        field = data_field.name.lower()
        # TODO: should add ground_sw_version and packet_file_name to
        # CDF and should use IntVar for DATA. Should only need to skip
        # SHCOARSE and ccsds_header (unless Lo wants this in the CDF as well).
        if field not in [
            "shcoarse",
            "ground_sw_version",
            "packet_file_name",
            "ccsds_header",
            "data",
        ]:
            # Create a list of all the dimensions using the DEPEND_I keys in the
            # YAML attributes
            dims = [
                value
                for key, value in attr_mgr.get_variable_attributes(field).items()
                if "DEPEND" in key
            ]

            # Create a data array for the current field and add it to the dataset
            # TODO: need to update to use packet data once that's available
            dataset[field] = xr.DataArray(
                [1, 1, 1], dims=dims, attrs=attr_mgr.get_variable_attributes(field)
            )

    return dataset
