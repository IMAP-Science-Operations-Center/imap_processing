"""Common functions that every instrument can use."""

import collections
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from space_packet_parser import definitions, encodings, packets, parameters

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_j2000ns

logger = logging.getLogger(__name__)


def sort_by_time(packets: list, time_key: str) -> list:
    """
    Sort packets by specified key.

    Parameters
    ----------
    packets : list
        Decom data packets.
    time_key : str
        Key to sort by. Must be a key in the packets data dictionary.
        e.g. "SHCOARSE" or "MET_TIME" or "ACQ_START_COARSE".

    Returns
    -------
    sorted_packets : list
        Sorted packets.
    """
    sorted_packets = sorted(packets, key=lambda x: x.data[time_key])
    return sorted_packets


def group_by_apid(packets: list) -> dict:
    """
    Group data by apid.

    Parameters
    ----------
    packets : list
        Packet list.

    Returns
    -------
    grouped_packets : dict
        Grouped data by apid.
    """
    grouped_packets: dict[list] = collections.defaultdict(list)
    for packet in packets:
        apid = packet.header["PKT_APID"].raw_value
        grouped_packets.setdefault(apid, []).append(packet)
    return grouped_packets


def convert_raw_to_eu(
    dataset: xr.Dataset,
    conversion_table_path: str,
    packet_name: str,
    **read_csv_kwargs: dict,
) -> xr.Dataset:
    """
    Convert raw data to engineering unit.

    Parameters
    ----------
    dataset : xr.Dataset
        Raw data.
    conversion_table_path : str,
        Path object or file-like object
        Path to engineering unit conversion table.
        Eg:
        f"{imap_module_directory}/swe/l1b/engineering_unit_convert_table.csv"
        Engineering unit conversion table must be a csv file with required
        informational columns: ('packetName', 'mnemonic', 'convertAs') and
        conversion columns named 'c0', 'c1', 'c2', etc. Conversion columns
        specify the array of polynomial coefficients used for the conversion.
        Comment lines are allowed in the csv file specified by starting with
        the '#' character.
    packet_name : str
        Packet name.
    **read_csv_kwargs : dict
        In order to allow for some flexibility in the format of the csv
        conversion table, any additional keywords passed to this function are
        passed in the call to `pandas.read_csv()`. See pandas documentation
        for a list of keywords and their functionality:
        https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html.

    Returns
    -------
    dataset : xr.Dataset
        Raw data converted to engineering unit as needed.
    """
    # Make sure there is column called "index" with unique
    # value such as 0, 1, 2, 3, ...
    eu_conversion_df = pd.read_csv(
        conversion_table_path,
        **read_csv_kwargs,
    )

    # Look up all metadata fields for the packet name
    packet_df = eu_conversion_df.loc[eu_conversion_df["packetName"] == packet_name]

    # for each metadata field, convert raw value to engineering unit
    for _, row in packet_df.iterrows():
        if row["convertAs"] == "UNSEGMENTED_POLY":
            # On this line, we are getting the coefficients from the
            # table and then reverse them because the np.polyval is
            # expecting coefficient in descending order
            # coeff columns must have names 'c0', 'c1', 'c2', ...
            coeff_values = row.filter(regex=r"c\d").values[::-1]
            row_key = row["mnemonic"]
            # TODO: remove this check once everyone has lowercase
            # all of CDF data variable names to match SPDF requirement.
            # Right now, check if dataset mnemonics is lowercase of row mnemonics.
            # If so, make them match
            mnemonics = row_key.lower() if row_key.lower() in dataset else row_key
            try:
                # Convert the raw value to engineering unit
                dataset[mnemonics].data = np.polyval(
                    coeff_values, dataset[mnemonics].data
                )
                # Modify units attribute
                if "unit" in row:
                    dataset[mnemonics].attrs.update({"units": row["unit"]})
            except KeyError:
                # TODO: Don't catch this error once packet definitions stabilize
                logger.warning(f"Input dataset does not contain key: {row_key}")
        else:
            raise ValueError(
                f"Unexpected conversion type: {row['convertAs']} encountered in"
                f" engineering unit conversion table: {conversion_table_path}"
            )

    return dataset


def create_dataset(
    packets: list[packets.CCSDSPacket],
    spacecraft_time_key: str = "shcoarse",
    include_header: bool = True,
    skip_keys: Optional[list[str]] = None,
) -> xr.Dataset:
    """
    Create dataset for each metadata field.

    Parameters
    ----------
    packets : list[Packet]
        Packet list.
    spacecraft_time_key : str
        Default is "shcoarse" because many instrument uses it, optional.
        This key is used to get spacecraft time for epoch dimension.
    include_header : bool
        Whether to include CCSDS header data in the dataset, optional.
    skip_keys : list
        Keys to skip in the metadata, optional.

    Returns
    -------
    dataset : xr.dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    metadata_arrays = collections.defaultdict(list)
    description_dict = {}

    sorted_packets = sort_by_time(packets, spacecraft_time_key.upper())

    for data_packet in sorted_packets:
        data_to_include = (
            (data_packet.header | data_packet.user_data)
            if include_header
            else data_packet.user_data
        )

        # Drop keys using skip_keys
        if skip_keys is not None:
            for key in skip_keys:
                data_to_include.pop(key, None)

        # Add metadata to array
        for key, value in data_to_include.items():
            # convert key to lower case to match SPDF requirement
            data_key = key.lower()
            metadata_arrays[data_key].append(value.raw_value)
            # description should be same for all packets
            description_dict[data_key] = (
                value.long_description or value.short_description
            )

    # NOTE: At this point, we keep epoch time as raw value from packet
    # which is in seconds and spacecraft time. Some instrument uses this
    # raw value in processing.
    # Load the CDF attributes
    cdf_manager = ImapCdfAttributes()
    epoch_time = xr.DataArray(
        metadata_arrays[spacecraft_time_key],
        name="epoch",
        dims=["epoch"],
        attrs=cdf_manager.get_variable_attributes("epoch"),
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
    )

    # create xarray dataset for each metadata field
    for key, value in metadata_arrays.items():
        # replace description and fieldname
        data_attrs = cdf_manager.get_variable_attributes("metadata_attrs")
        data_attrs["CATDESC"] = description_dict[key]
        data_attrs["FIELDNAM"] = key
        data_attrs["LABLAXIS"] = key

        dataset[key] = xr.DataArray(
            value,
            dims=["epoch"],
            attrs=data_attrs,
        )

    return dataset


def _get_minimum_numpy_datatype(  # noqa: PLR0912 - Too many branches
    name: str,
    definition: definitions.XtcePacketDefinition,
    use_derived_value: bool = True,
) -> Optional[str]:
    """
    Get the minimum datatype for a given variable.

    Parameters
    ----------
    name : str
        The variable name.
    definition : space_packet_parser.definitions.XtcePacketDefinition
        The XTCE packet definition.
    use_derived_value : bool, default True
        Whether or not the derived value from the XTCE definition was used.

    Returns
    -------
    datatype : str
        The minimum datatype.
    """
    data_encoding = definition.named_parameters[name].parameter_type.encoding

    if use_derived_value and isinstance(
        definition.named_parameters[name].parameter_type,
        parameters.EnumeratedParameterType,
    ):
        # We don't have a way of knowing what is enumerated,
        # let numpy infer the datatype
        return None
    elif isinstance(data_encoding, encodings.NumericDataEncoding):
        if use_derived_value and (
            data_encoding.context_calibrators is not None
            or data_encoding.default_calibrator is not None
        ):
            # If there are calibrators, we need to default to None and
            # let numpy infer the datatype
            return None
        nbits = data_encoding.size_in_bits
        if isinstance(data_encoding, encodings.IntegerDataEncoding):
            datatype = "int"
            if data_encoding.encoding == "unsigned":
                datatype = "uint"
            if nbits <= 8:
                datatype += "8"
            elif nbits <= 16:
                datatype += "16"
            elif nbits <= 32:
                datatype += "32"
            else:
                datatype += "64"
        elif isinstance(data_encoding, encodings.FloatDataEncoding):
            datatype = "float"
            if nbits == 32:
                datatype += "32"
            else:
                datatype += "64"
    elif isinstance(data_encoding, encodings.BinaryDataEncoding):
        # TODO: Binary string representation right now, do we want bytes or
        # something else like the new StringDType instead?
        datatype = "object"
    elif isinstance(data_encoding, encodings.StringDataEncoding):
        # TODO: Use the new StringDType instead?
        datatype = "str"
    else:
        raise ValueError(f"Unsupported data encoding: {data_encoding}")

    return datatype


def packet_file_to_datasets(
    packet_file: Union[str, Path],
    xtce_packet_definition: Union[str, Path],
    use_derived_value: bool = False,
) -> dict[int, xr.Dataset]:
    """
    Convert a packet file to xarray datasets.

    The packet file can contain multiple apids and these will be separated
    into distinct datasets, one per apid. The datasets will contain the
    ``derived_value``s of the data fields, and the ``raw_value``s if no
    ``derived_value`` is available. If there are conversions in the XTCE
    packet definition, the ``derived_value`` will be the converted value.
    The dimension of the dataset will be the time field in J2000 nanoseconds.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename.
    xtce_packet_definition : str
        Path to XTCE file with filename.
    use_derived_value : bool, default False
        Whether or not to use the derived value from the XTCE definition.

    Returns
    -------
    datasets : dict
        Mapping from apid to xarray dataset, one dataset per apid.

    Notes
    -----
    This function only handles packet definitions with the same variable structure
    across all packets with the same ApId. For example, this cannot be used for IDEX
    due to the conditional XML structure defined for their science packet.
    """
    # Set up containers to store our data
    # We are getting a packet file that may contain multiple apids
    # Each apid has consistent data fields, so we want to create a
    # dataset per apid.
    # {apid1: dataset1, apid2: dataset2, ...}
    data_dict: dict[int, dict] = dict()
    # Also keep track of the datatype mapping for each field
    datatype_mapping: dict[int, dict] = dict()
    # Keep track of which variables (keys) are in the dataset
    variable_mapping: dict[int, set] = dict()

    # Set up the parser from the input packet definition
    packet_definition = definitions.XtcePacketDefinition(xtce_packet_definition)

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_definition.packet_generator(binary_data)
        for packet in packet_generator:
            apid = packet.header["PKT_APID"].raw_value
            if apid not in data_dict:
                # This is the first packet for this APID
                data_dict[apid] = collections.defaultdict(list)
                datatype_mapping[apid] = dict()
                variable_mapping[apid] = packet.user_data.keys()
            if variable_mapping[apid] != packet.user_data.keys():
                raise ValueError(
                    f"Packet fields do not match for APID {apid}. This could be "
                    f"due to a conditional packet definition in the XTCE, while this "
                    f"function currently only supports flat packet definitions."
                    f"\nExpected: {variable_mapping[apid]},\n"
                    f"got: {packet.user_data.keys()}"
                )

            # TODO: Do we want to give an option to remove the header content?
            packet_content = packet.user_data | packet.header

            for key, value in packet_content.items():
                val = value.raw_value
                if use_derived_value:
                    # Use the derived value if it exists, otherwise use the raw value
                    val = value
                data_dict[apid][key].append(val)
                if key not in datatype_mapping[apid]:
                    # Add this datatype to the mapping
                    datatype_mapping[apid][key] = _get_minimum_numpy_datatype(
                        key, packet_definition, use_derived_value=use_derived_value
                    )

    dataset_by_apid = {}

    for apid, data in data_dict.items():
        # The time key is always the first key in the data dictionary on IMAP
        time_key = next(iter(data.keys()))
        # Convert to J2000 time and use that as our primary dimension
        time_data = met_to_j2000ns(data[time_key])
        # data_dict = {}
        # for key, list_of_values in data.items():
        #     # Get the datatype for this field
        #     datatype = datatype_mapping[apid][key]
        #     if datatype == "object":
        #         # convert <class bytes> to <class str>
        #         # TODO: we all need to update our code to use <class bytes> instead
        #         binary_str_val = [None] * len(list_of_values)
        #         for index, data in enumerate(list_of_values):
        #             binary_str_val[index] = ''.join(f'{byte:08b}' for byte in data)
        #         # Update to new datatype and values
        #         datatype = "str"
        #         list_of_values = binary_str_val
        #     data_dict[key.lower()] = (
        #             "epoch",
        #             np.asarray(list_of_values, dtype=datatype),
        #         )
        ds = xr.Dataset(
            {
                key.lower(): (
                    "epoch",
                    np.asarray(list_of_values, dtype=datatype_mapping[apid][key]),
                )
                for key, list_of_values in data.items()
            },
            coords={"epoch": time_data},
        )
        ds = ds.sortby("epoch")

        # Strip any leading characters before "." from the field names which was due
        # to the packet_name being a part of the variable name in the XTCE definition
        ds = ds.rename(
            {
                # partition splits the string into 3 parts: before ".", ".", after "."
                # if there was no ".", the second part is an empty string, so we use
                # the original key in that case
                key: key.partition(".")[2] or key
                for key in ds.variables
            }
        )

        dataset_by_apid[apid] = ds

    return dataset_by_apid
