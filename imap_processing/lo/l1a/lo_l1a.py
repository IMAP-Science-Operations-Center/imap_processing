"""IMAP-Lo L1A Data Processing."""

from dataclasses import fields
from pathlib import Path

import numpy as np
import xarray as xr

from imap_processing import decom, imap_module_directory, utils
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import met_to_j2000ns
from imap_processing.lo.l0.data_classes.science_counts import ScienceCounts
from imap_processing.lo.l0.lo_apid import LoAPID
from imap_processing.lo.l1a.lo_data_container import LoContainer
from imap_processing.utils import packet_file_to_datasets
from imap_processing.lo.l0.lo_science import parse_histogram


def lo_l1a(dependency: Path, data_version: str) -> list[xr.Dataset]:
    """
    Will process IMAP-Lo L0 data into L1A CDF data products.

    Parameters
    ----------
    dependency : Path
        Dependency file needed for data product creation.
        Should always be only one for L1A.
    data_version : str
        Version of the data product being created.

    Returns
    -------
    created_file_paths : list[Path]
        Location of created CDF files.
    """
    xtce_file = imap_module_directory / "lo/packet_definitions/lo_xtce.xml"
    datasets_by_apid = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=dependency.resolve(),
        xtce_packet_definition=xtce_file.resolve(),
    )

    # create the attribute manager for this data level
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="lo")
    attr_mgr.add_instrument_variable_attrs(instrument="lo", level="l1a")
    attr_mgr.add_global_attribute("Data_version", data_version)

    if LoAPID.ILO_SCI_CNT in datasets_by_apid:
        logical_source = "imap_lo_l1a_histogram"

        azimuth_60 = xr.DataArray(
            data=np.arange(0, 6),
            name="azimuth_60",
            dims=["azimuth_60"],
            attrs=attr_mgr.get_variable_attributes("azimuth_60"),
        )
        azimuth_60_label = xr.DataArray(
            data=azimuth_60.values.astype(str),
            name="azimuth_60_label",
            dims=["azimuth_60_label"],
            attrs=attr_mgr.get_variable_attributes("azimuth_60_label"),
        )
        azimuth_6 = xr.DataArray(
            data=np.arange(0, 60),
            name="azimuth_6",
            dims=["azimuth_6"],
            attrs=attr_mgr.get_variable_attributes("azimuth_6"),
        )
        azimuth_6_label = xr.DataArray(
            data=azimuth_6.values.astype(str),
            name="azimuth_6_label",
            dims=["azimuth_6_label"],
            attrs=attr_mgr.get_variable_attributes("azimuth_6_label"),
        )

        esa_step = xr.DataArray(
            data=np.arange(1, 8),
            name="esa_step",
            dims=["esa_step"],
            attrs=attr_mgr.get_variable_attributes("esa_step"),
        )
        esa_step_label = xr.DataArray(
            esa_step.values.astype(str),
            name="esa_step_label",
            dims=["esa_step_label"],
            attrs=attr_mgr.get_variable_attributes("esa_step_label"),
        )

        datasets_by_apid[LoAPID.ILO_SCI_CNT].shcoarse.attrs.update(attr_mgr.get_variable_attributes("shcoarse"))

        datasets_by_apid[LoAPID.ILO_SCI_CNT] = (
            parse_histogram(datasets_by_apid[LoAPID.ILO_SCI_CNT], attr_mgr))

        datasets_by_apid[LoAPID.ILO_SCI_CNT] =(
            datasets_by_apid[LoAPID.ILO_SCI_CNT].assign_coords(
            azimuth_60=azimuth_60,
            azimuth_60_label=azimuth_60_label,
            azimuth_6=azimuth_6,
            azimuth_6_label=azimuth_6_label,
            esa_step=esa_step,
            esa_step_label=esa_step_label,
        ))
        datasets_by_apid[LoAPID.ILO_SCI_CNT].attrs.update(attr_mgr.get_global_attributes(logical_source))
        # remove the binary field from the dataset
        datasets_by_apid[LoAPID.ILO_SCI_CNT] = datasets_by_apid[LoAPID.ILO_SCI_CNT].drop_vars(["sci_cnt", "chksum"])

    print(datasets_by_apid[LoAPID.ILO_SCI_CNT])
    return [datasets_by_apid[LoAPID.ILO_SCI_CNT]]




#     data = LoContainer()
#     for packet in sorted_packets:
#         if packet.header["PKT_APID"].derived_value == LoAPID.ILO_SCI_CNT:
#             data.add(ScienceCounts(packet, data_version, dependency.name))
#
#     lo_datasets = []
#     if data.filter_apid(LoAPID.ILO_SCI_CNT):
#         lo_datasets.append(
#             create_datasets(attr_mgr, data.filter_apid(LoAPID.ILO_SCI_CNT))
#         )
#
#     # # TODO: replace with real processing when sample data is available
#     # # Temporary code until I get sample data. The fields will still be pulled
#     # # from the data classes, but the data class will be determined by the
#     # # APID in the decommed file and the data class will first be populated
#     # # with the packet data.
#     # packet_file_name = dependency.name
#     # if "_de_" in packet_file_name:
#     #     data_fields = fields(ScienceDirectEvents)
#     #
#     # elif "_spin_" in packet_file_name:
#     #     # Spin data class doesn't exist yet. fake the fields() output
#     #     data_field_tup = namedtuple("data_field_tup", ["name"])
#     #     data_fields = [  # type: ignore[assignment]
#     #         # TODO, The errors are because the fake data is not in the correct format.
#     #         data_field_tup("NUM_COMPLETED"),
#     #         data_field_tup("ACQ_END"),
#     #         data_field_tup("SPIN_SECONDS"),
#     #         data_field_tup("SPIN_SUBSECONDS"),
#     #         data_field_tup("ESA_P_DAC"),
#     #         data_field_tup("ESA_N_DAC"),
#     #         data_field_tup("VAL_PERIOD"),
#     #         data_field_tup("VAL_SPIN"),
#     #         data_field_tup("SOURCE"),
#     #     ]
#     #     logical_source = "imap_lo_l1a_spin"
#     #
#     # dataset: list[Path] = create_datasets(attr_mgr, logical_source, data_fields)
#     #
#     print(f"type: {type(lo_datasets[0])}")
#     return lo_datasets
#
#
# # TODO: This is going to work differently when I sample data.
# #  The data_fields input is temporary.
# def create_datasets(attr_mgr: ImapCdfAttributes, data: list) -> xr.Dataset:
#     """
#     Create a dataset using the populated data classes.
#
#     Parameters
#     ----------
#     attr_mgr : ImapCdfAttributes
#         Attribute manager used to get the data product field's attributes.
#     data : list
#         List of Lo L1 data class objects.
#
#     Returns
#     -------
#     dataset : xarray.Dataset
#         Dataset with all data product fields in xarray.DataArray.
#     """
#     # Convert each packet's spacecraft time to an absolute epoch time
#     # TODO: replace temp hardcoded values with packet values
#     epoch_converted_time = met_to_j2000ns([packet.SHCOARSE for packet in data])
#
#     # Create a data array for the epoch time
#     # TODO: might need to update the attrs to use new YAML file
#     epoch_time = xr.DataArray(
#         data=epoch_converted_time,
#         name="epoch",
#         dims=["epoch"],
#         attrs=attr_mgr.get_variable_attributes("epoch"),
#     )
#
#     if data[0].ccsds_header.PKT_APID == LoAPID.ILO_SCI_CNT:
#         logical_source = "imap_lo_l1a_histogram"
#
#         azimuth_60 = xr.DataArray(
#             data=np.arange(0, 6),
#             name="azimuth_60",
#             dims=["azimuth_60"],
#             attrs=attr_mgr.get_variable_attributes("azimuth_60"),
#         )
#         azimuth_60_label = xr.DataArray(
#             data=azimuth_60.values.astype(str),
#             name="azimuth_60_label",
#             dims=["azimuth_60_label"],
#             attrs=attr_mgr.get_variable_attributes("azimuth_60_label"),
#         )
#         azimuth_6 = xr.DataArray(
#             data=np.arange(0, 60),
#             name="azimuth_6",
#             dims=["azimuth_6"],
#             attrs=attr_mgr.get_variable_attributes("azimuth_6"),
#         )
#         azimuth_6_label = xr.DataArray(
#             data=azimuth_6.values.astype(str),
#             name="azimuth_6_label",
#             dims=["azimuth_6_label"],
#             attrs=attr_mgr.get_variable_attributes("azimuth_6_label"),
#         )
#
#         esa_step = xr.DataArray(
#             data=np.arange(1, 8),
#             name="esa_step",
#             dims=["esa_step"],
#             attrs=attr_mgr.get_variable_attributes("esa_step"),
#         )
#         esa_step_label = xr.DataArray(
#             esa_step.values.astype(str),
#             name="esa_step_label",
#             dims=["esa_step_label"],
#             attrs=attr_mgr.get_variable_attributes("esa_step_label"),
#         )
#
#         dataset = xr.Dataset(
#             coords={
#                 "epoch": epoch_time,
#                 "azimuth_60": azimuth_60,
#                 "azimuth_60_label": azimuth_60_label,
#                 "azimuth_6": azimuth_6,
#                 "azimuth_6_label": azimuth_6_label,
#                 "esa_step": esa_step,
#                 "esa_step_label": esa_step_label,
#             },
#             attrs=attr_mgr.get_global_attributes(logical_source),
#         )
#
#     data_fields = fields(data[0])
#     print(f"DATA FIELDS: {data_fields}")
#     for data_field in data_fields:
#         field = data_field.name.lower()
#         # TODO: should add ground_sw_version and packet_file_name to
#         # CDF and should use IntVar for DATA. Should only need to skip
#         # SHCOARSE and ccsds_header (unless Lo wants this in the CDF as well).
#         if field not in [
#             "shcoarse",
#             "ground_sw_version",
#             "packet_file_name",
#             "ccsds_header",
#             "sci_cnt",
#         ]:
#             # Create a list of all the dimensions using the DEPEND_I keys in the
#             # YAML attributes
#             dims = [
#                 value
#                 for key, value in attr_mgr.get_variable_attributes(field).items()
#                 if "DEPEND" in key
#             ]
#
#             print(f"FIELD: {field} DIMS: {dims}")
#             # Create a data array for the current field and add it to the dataset
#             # TODO: need to update to use packet data once that's available
#             dataset[field] = xr.DataArray(
#                 np.stack(
#                     [getattr(packet, field.upper()) for packet in data], dtype=int
#                 ),
#                 dims=dims,
#                 attrs=attr_mgr.get_variable_attributes(field),
#             )
#
#     # # Set any extra coordinates and initialize the dataset
#     # # based on the L1A data product that will be created
#     # # by the current packets
#     # if logical_source == "imap_lo_l1a_spin":
#     #     # The spin packet contains a variable number of spins in each
#     #     # packet. To avoid needing to use a 2-dimensional array to
#     #     # store the spin fields, epoch will not be used for many
#     #     # of the fields. They will instead use the coordinate `spin`
#     #     # which is an index of each spin in the pointing (0 to num spins in pointing)
#     #     spin = xr.DataArray(
#     #         data=[0, 1, 2],
#     #         name="spin",
#     #         dims=["spin"],
#     #         attrs=attr_mgr.get_variable_attributes("spin"),
#     #     )
#     #
#     #     dataset = xr.Dataset(
#     #         coords={"epoch": epoch_time, "spin": spin},
#     #         attrs=attr_mgr.get_global_attributes(logical_source),
#     #     )
#     #
#     # elif logical_source == "imap_lo_l1a_de":
#     #     # The de packet contains a variable number of events in each
#     #     # packet. To avoid needing to use a 2-dimensional array to
#     #     # store the de fields, epoch will not be used for many
#     #     # of the fields and they will instead use the coordinate `direct_events`
#     #     # which is an index of each event in the pointing (0 to num de in pointing)
#     #     direct_events = xr.DataArray(
#     #         data=[0, 1, 2],
#     #         name="direct_events",
#     #         dims=["direct_events"],
#     #         attrs=attr_mgr.get_variable_attributes("direct_events"),
#     #     )
#     #
#     #     dataset = xr.Dataset(
#     #         coords={"epoch": epoch_time, "direct_events": direct_events},
#     #         attrs=attr_mgr.get_global_attributes(logical_source),
#     #     )
#     #
#     # # Loop through the data fields that were pulled from the
#     # # data class. These should match the field names given
#     # # to each field in the YAML attribute file
#     # for data_field in data_fields:
#     #     field = data_field.name.lower()
#     #     # TODO: should add ground_sw_version and packet_file_name to
#     #     # CDF and should use IntVar for DATA. Should only need to skip
#     #     # SHCOARSE and ccsds_header (unless Lo wants this in the CDF as well).
#     #     if field not in [
#     #         "shcoarse",
#     #         "ground_sw_version",
#     #         "packet_file_name",
#     #         "ccsds_header",
#     #         "data",
#     #     ]:
#     #         # Create a list of all the dimensions using the DEPEND_I keys in the
#     #         # YAML attributes
#     #         dims = [
#     #             value
#     #             for key, value in attr_mgr.get_variable_attributes(field).items()
#     #             if "DEPEND" in key
#     #         ]
#     #
#     #         # Create a data array for the current field and add it to the dataset
#     #         # TODO: need to update to use packet data once that's available
#     #         dataset[field] = xr.DataArray(
#     #             [1, 1, 1], dims=dims, attrs=attr_mgr.get_variable_attributes(field)
#     #         )
#
#     return dataset
