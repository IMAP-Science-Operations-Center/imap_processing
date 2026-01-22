"""Common functions that every instrument can use."""

import collections
import logging
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pandas as pd
import space_packet_parser as spp
import xarray as xr
from space_packet_parser.exceptions import UnrecognizedPacketTypeError
from space_packet_parser.generators.ccsds import SequenceFlags
from space_packet_parser.xtce import definitions, encodings, parameter_types

from imap_processing.spice.time import met_to_ttj2000ns

logger = logging.getLogger(__name__)

# The time key is the secondary header, right after the primary header
# in the data dictionary on IMAP (8th key overall)
TIME_KEY_INDEX = 7


def convert_raw_to_eu(
    dataset: xr.Dataset,
    conversion_table_path: str,
    packet_name: str,
    **read_csv_kwargs: dict,
) -> xr.Dataset:  # numpydoc ignore=PR01,PR09
    """
    Convert raw data to engineering unit.

    Parameters
    ----------
    dataset : xr.Dataset
        Raw data.
    conversion_table_path : str
        Path object or file-like object
        Path to engineering unit conversion table.
        E.g.
        f"{imap_module_directory}/swe/l1b/engineering_unit_convert_table.csv"
        Engineering unit conversion table must be a csv file with required
        informational columns: ('packetName', 'mnemonic', 'convertAs') and
        conversion columns named 'c0', 'c1', 'c2', etc. Conversion columns
        specify the array of polynomial coefficients used for the conversion.
        If the column 'convertAs' is 'SEGMENTED_POLY' then there must be columns
        'dn_range_start' and 'dn_range_stop' that specifies the raw DN range and the
        coefficients that should be used for the conversion.

        E.g.:

        mnemonic       convertAs …       dn_range_start   dn_range_stop  c0    c1…
        --------------------------------------------------------------------------
        temperature  |  SEGMENTED_POLY | 0              | 2063         | 0.1  | 0.2
        temperature  |  SEGMENTED_POLY | 2064           | 3853         | 0    | 0.1
        temperature  |  SEGMENTED_POLY | 3854           | 4094         | 0.6  | 0.3
        sensor_v     |  UNSEGMENTED_POLY |              |              | 0.04 | .110

        Comment lines are allowed in the csv file specified by starting with
        the '#' character.
    packet_name : str
        Packet name.
    **read_csv_kwargs : dict
        In order to allow for some flexibility in the format of the csv
        conversion table, any additional keywords passed to this function are
        passed in the call to `pandas.read_csv()`.
        See pandas documentation
        for a list of keywords and their functionality:
        https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html.

    Returns
    -------
    dataset : xr.Dataset
        Raw data converted to engineering unit as needed.
    """
    eu_conversion_df = pd.read_csv(
        conversion_table_path,
        **read_csv_kwargs,
    )

    # Iterate through every variable in the dataset and check if there is an entry for
    # That variable in the conversion table.
    for var in dataset.variables:
        packet_df = eu_conversion_df.loc[
            (eu_conversion_df["packetName"] == packet_name)
            &
            # Filter for mnemonic case-insensitive
            (eu_conversion_df["mnemonic"].str.lower() == var.lower())
        ].reset_index(drop=True)

        if packet_df.empty:
            continue

        if np.all(packet_df["convertAs"] == "UNSEGMENTED_POLY"):
            if len(packet_df.index) > 1:
                raise ValueError(
                    "For unsegmented polynomial conversions, there should "
                    "only be one row per mnemonic and packet name."
                )
            row = packet_df.iloc[0]
            # On this line, we are getting the coefficients from the
            # table and then reverse them because the np.polyval is
            # expecting coefficient in descending order
            # coeff columns must have names 'c0', 'c1', 'c2', ...
            coeff_values = row.filter(regex=r"c\d").values[::-1]
            # Convert the raw value to engineering unit
            dataset[var].data = np.polyval(coeff_values, dataset[var].data)

        elif np.all(packet_df["convertAs"] == "SEGMENTED_POLY"):
            data = dataset[var].data
            # Check if any of the raw DN values fall outside the ranges
            bad_mask = np.logical_or(
                data < packet_df["dn_range_start"].min(),
                data > packet_df["dn_range_stop"].max(),
            )
            if np.any(data[bad_mask]):
                raise ValueError(
                    "Raw DN values found outside of the expected range"
                    f"for mnemonic: {var}"
                )
            # Create conditions and corresponding functions for np.piecewise
            conditions = [
                (data >= row["dn_range_start"]) & (data <= row["dn_range_stop"])
                for _, row in packet_df.iterrows()
            ]
            functions = [
                lambda x, r=row: np.polyval(r.filter(regex=r"c\d").values[::-1], x)
                for _, row in packet_df.iterrows()
            ]
            # Convert the raw value to engineering unit
            dataset[var].data = np.piecewise(data, conditions, functions)

        else:
            raise ValueError(
                "Column 'convertAs' must all be UNSEGMENTED_POLY or "
                "SEGMENTED_POLY for a packet name and mnemonic"
            )

        # Modify units attribute
        if "unit" in packet_df:
            dataset[var].attrs.update({"UNITS": packet_df.iloc[0]["unit"]})

    return dataset


def _get_minimum_numpy_datatype(  # noqa: PLR0912 - Too many branches
    name: str,
    definition: definitions.XtcePacketDefinition,
    use_derived_value: bool = True,
) -> str | None:
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
    data_encoding = definition.parameters[name].parameter_type.encoding

    if use_derived_value and isinstance(
        definition.parameters[name].parameter_type,
        parameter_types.EnumeratedParameterType,
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
    packet_file: str | Path,
    xtce_packet_definition: str | Path,
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
    packet_definition = spp.load_xtce(xtce_packet_definition)

    for packet in packet_generator(packet_file, xtce_packet_definition):
        apid = packet["PKT_APID"]
        if apid not in data_dict:
            # This is the first packet for this APID
            data_dict[apid] = collections.defaultdict(list)
            datatype_mapping[apid] = dict()
            variable_mapping[apid] = packet.keys()
        if variable_mapping[apid] != packet.keys():
            raise ValueError(
                f"Packet fields do not match for APID {apid}. This could be "
                f"due to a conditional packet definition in the XTCE, while this "
                f"function currently only supports flat packet definitions."
                f"\nExpected: {variable_mapping[apid]},\n"
                f"got: {packet.keys()}"
            )

        for key, value in packet.items():
            val = value if use_derived_value else value.raw_value
            data_dict[apid][key].append(val)
            if key not in datatype_mapping[apid]:
                # Add this datatype to the mapping
                datatype_mapping[apid][key] = _get_minimum_numpy_datatype(
                    key, packet_definition, use_derived_value=use_derived_value
                )

    dataset_by_apid = {}

    for apid, data in data_dict.items():
        try:
            time_key = list(data.keys())[TIME_KEY_INDEX]
        except IndexError:
            logger.debug(
                f"Could not determine time key for APID {apid}, skipping dataset."
            )
            continue
        # Convert to J2000 time and use that as our primary dimension
        time_data = met_to_ttj2000ns(data[time_key])
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
        # We may get duplicate packets within the packet file if packets were
        # ingested multiple times by the POC. We want to drop packets where
        # apid, epoch, and src_seq_ctr are the same.

        # xarray only supports dropping duplicates by index, so we instead go
        # to pandas multi-index dataframe to identify the unique positions
        unique_indices = (
            ds[["src_seq_ctr"]]
            .to_dataframe()
            .reset_index()
            .drop_duplicates()
            .index.values
        )
        nduplicates = len(ds["epoch"]) - len(unique_indices)
        if nduplicates != 0:
            logger.warning(
                f"Found [{nduplicates}] duplicate packets for APID {apid}. "
                "Dropping duplicate packets and continuing processing."
            )
            ds = ds.isel(epoch=unique_indices)

        # Log a warning if there are gaps in the source sequence counter
        _check_source_sequence_counter(ds, apid)

        # Strip any leading characters before "." from the field names which was due
        # to the packet_name being a part of the variable name in the XTCE definition
        ds = ds.rename(
            {
                # partition splits the string into 3 parts: before ".", after "."
                # if there was no ".", the second part is an empty string, so we use
                # the original key in that case
                key: key.partition(".")[2] or key
                for key in ds.variables
            }
        )

        dataset_by_apid[apid] = ds

    return dataset_by_apid


def combine_segmented_packets(
    packets: xr.Dataset, binary_field_name: str = "packetdata"
) -> xr.Dataset:
    """
    Combine segmented packets into unsegmented packets.

    To combine the segmented packets, we only concatenate along the `binary_field_name`
    and place all values into the first packet of the group. The binary_field_name
    is the name of the XTCE Parameter that contains the binary data for the packet.
    The other fields are left as-is from the first packet of the group.

    Parameters
    ----------
    packets : xarray.Dataset
        Dataset containing the packets to combine.
    binary_field_name : str, default "packetdata"
        Name of the binary field in the dataset representing the packet data.
        Defined in the XTCE definition for each instrument.

    Returns
    -------
    combined_packets : xarray.Dataset
        Dataset containing the combined packets.
    """
    # Identification of group starts
    # NOTE: seq_flgs is the same variable name for all instruments on IMAP
    #       but could be different for other missions depending on the XTCE definition.
    is_group_start = (packets["seq_flgs"].data == SequenceFlags.UNSEGMENTED) | (
        packets["seq_flgs"].data == SequenceFlags.FIRST
    )

    # Assign group IDs using cumulative sum - each group start increments the ID
    group_ids = np.cumsum(is_group_start)

    # Get indices of packets we'll keep (first packet of each group)
    group_start_indices = np.where(is_group_start)[0]
    # Keep track of the groups that don't have the expected sequences
    bad_groups = []

    # Concatenate binary data in-place for each group
    for group_id in np.unique(group_ids):
        # Find all packets belonging to this group
        group_mask = group_ids == group_id
        group_indices = np.where(group_mask)[0]

        # If multiple packets, concatenate into the first packet
        # [b"abc", b"def", b"ghi"] -> b"abcdefghi"
        if (
            len(group_indices) > 1
            or packets["seq_flgs"].data[group_indices[0]] != SequenceFlags.UNSEGMENTED
        ):
            start_index = group_indices[0]
            # Lets do some quick validation on these packets since we've had
            # some missing packet groups in the past
            seq_flags = packets["seq_flgs"].data[group_indices]
            if (
                seq_flags[0] != SequenceFlags.FIRST
                or seq_flags[-1] != SequenceFlags.LAST
                or (
                    len(seq_flags) > 2
                    and not np.all(seq_flags[1:-1] == SequenceFlags.CONTINUATION)
                )
            ):
                bad_groups.append(start_index)
                logger.warning(
                    f"Incorrect/incomplete sequence flags in group {group_id}. "
                    f"Flags: {seq_flags}, "
                    f"SHCOARSEs: {packets['shcoarse'].data[group_indices]}"
                )

            packets[binary_field_name].data[start_index] = np.sum(
                packets[binary_field_name].data[group_indices]
            )

    # Remove any bad groups from the start indices we are keeping
    group_start_indices = np.setdiff1d(group_start_indices, bad_groups)
    # Select only the first packet of each group (drop the middle/last packets)
    combined_packets = packets.isel(epoch=group_start_indices)

    return combined_packets


def _check_source_sequence_counter(ds: xr.Dataset, apid: int) -> None:
    """
    Check for gaps in the source sequence counter.

    Log a warning if gaps are found, but don't do anything else.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the packets to check.
    apid : int
        APID of the packets.
    """
    # Check for sequential source sequence counters
    # CCSDS source sequence counter is a 14-bit field (0-16383)
    counter_max = 16384
    src_seq_ctr = ds["src_seq_ctr"].data

    if len(src_seq_ctr) <= 1:
        return

    # Check if each counter equals (previous + 1) % counter_max
    # This handles both normal increments and rollover (16383 -> 0)
    expected = (src_seq_ctr[:-1] + 1) % counter_max
    actual = src_seq_ctr[1:]
    non_sequential = expected != actual

    if np.any(non_sequential):
        gap_indices = np.where(non_sequential)[0]
        # Calculate total missing packets across all gaps
        total_missing = sum(
            (src_seq_ctr[idx + 1] - src_seq_ctr[idx] - 1) % counter_max
            for idx in gap_indices
        )
        # Show the counter values before and after each gap
        gap_starts = src_seq_ctr[gap_indices].tolist()
        gap_ends = src_seq_ctr[gap_indices + 1].tolist()
        gap_pairs = list(zip(gap_starts, gap_ends, strict=True))
        logger.warning(
            f"Found [{len(gap_indices)}] gap(s) in source sequence counter "
            f"for APID {apid} at {gap_pairs} "
            f"({total_missing} total missing packets)"
        )


def packet_generator(
    packet_file: str | Path,
    xtce_packet_definition: str | Path,
) -> Generator[spp.SpacePacket, None, None]:
    """
    Parse packets from a packet file.

    Parameters
    ----------
    packet_file : str | Path
        Path to data packet path with filename.
    xtce_packet_definition : str | Path
        Path to XTCE file with filename.

    Yields
    ------
    packet : space_packet_parser.SpacePacket
        Parsed packet dictionary.
    """
    # Set up the parser from the input packet definition
    packet_definition = spp.load_xtce(xtce_packet_definition)

    with open(packet_file, "rb") as binary_data:
        for binary_packet in spp.ccsds_generator(binary_data):
            try:
                packet = packet_definition.parse_bytes(binary_packet)
            except UnrecognizedPacketTypeError as e:
                # NOTE: Not all of our definitions have all of the APIDs
                #       we may encounter, so we only want to process ones
                #       we can actually parse.
                logger.debug(e)
                continue
            yield packet


def separate_ccsds_header_userdata(packet: dict) -> tuple[dict, dict]:
    """
    Separate header and userdata from a parsed packet.

    DO NOT USE:
    This function is not used by instruments other than GLOWS and MAG and should
    not be relied upon for general use since XTCE definitions may have different
    structures defining the header items.

    This assumes that the first 7 items in the packet dictionary are the CCSDS
    header and the following are the userdata section. It assumes insertion order
    is kept and puts the first 7 items into one dictionary, with all of the following
    variables assumed to be userdata in a second dictionary. All values are
    raw values and it doesn't not return the derived values.

    Parameters
    ----------
    packet : dict
        Packet dictionary.

    Returns
    -------
    header : dict
        Packet header dictionary.
    user_data : dict
        Packet userdata dictionary (raw values).
    """
    it = iter(packet.items())
    # take first 7 items for header (indices 0..6)
    header = {}
    for _, (k, v) in zip(range(7), it, strict=False):
        header[k] = v
    # remaining items are userdata; prefer raw_value if present
    userdata = {k: v.raw_value for k, v in it}
    return header, userdata


def convert_to_binary_string(data: bytes) -> str:
    """
    Convert bytes to a string representation.

    Parameters
    ----------
    data : bytes
        Bytes to convert to a binary string.

    Returns
    -------
    binary_data : str
        The binary data as a string.
    """
    binary_str_data = f"{int.from_bytes(data, byteorder='big'):0{len(data) * 8}b}"
    return binary_str_data
