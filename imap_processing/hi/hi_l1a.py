"""IMAP-HI L1A processing module."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from numpy import _typing as npt
from numpy._typing import NDArray

from imap_processing import imap_module_directory
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.hi.utils import HIAPID
from imap_processing.spice.time import met_to_ttj2000ns
from imap_processing.utils import packet_file_to_datasets

# TODO: read DE_CLOCK_TICK_US from
# instrument status summary later. This value
# is rarely change but want to be able to change
# it if needed. It stores information about how
# fast the time was ticking. It is in microseconds.
DE_CLOCK_TICK_US = 1999
DE_CLOCK_TICK_S = DE_CLOCK_TICK_US / 1e6
HALF_CLOCK_TICK_S = DE_CLOCK_TICK_S / 2

MILLISECOND_TO_S = 1e-3

# define the names of the 24 counter arrays
# contained in the histogram packet
QUALIFIED_COUNTERS = (
    "ab_qualified",
    "c1c2_qualified",
    "ac1_qualified",
    "bc1_qualified",
    "abc1_qualified",
    "ac1c2_qualified",
    "bc1c2_qualified",
    "abc1c2_qualified",
)
LONG_COUNTERS = (
    "a_first_only",
    "b_first_only",
    "c_first_only",
    "ab_long",
    "c1c2_long",
    "ac1_long",
    "bc1_long",
    "abc1_long",
    "ac1c2_long",
    "bc1c2_long",
    "abc1c2_long",
)
TOTAL_COUNTERS = ("a_total", "b_total", "c_total", "fee_de_recd", "fee_de_sent")

logger = logging.getLogger(__name__)


def hi_l1a(packet_file_path: Union[str, Path]) -> list[xr.Dataset]:
    """
    Will process IMAP raw data to l1a.

    Parameters
    ----------
    packet_file_path : str
        Data packet file path.

    Returns
    -------
    processed_data : list[xarray.Dataset]
        List of processed xarray dataset.
    """
    datasets_by_apid = hi_packet_file_to_datasets(packet_file_path)

    # Process science to l1a.
    processed_data = []
    for apid in datasets_by_apid:
        try:
            apid_enum = HIAPID(apid)
        except ValueError as err:
            raise RuntimeError(f"Encountered unexpected APID [{apid}]") from err

        logger.info(f"Processing IMAP-Hi data for {apid_enum.name} packets")

        if apid_enum in [HIAPID.H45_SCI_CNT, HIAPID.H90_SCI_CNT]:
            data = finish_hist_dataset(datasets_by_apid[apid])
            gattr_key = "imap_hi_l1a_hist_attrs"
        elif apid_enum in [HIAPID.H45_SCI_DE, HIAPID.H90_SCI_DE]:
            data = finish_de_dataset(datasets_by_apid[apid])
            gattr_key = "imap_hi_l1a_de_attrs"
        elif apid_enum in [HIAPID.H45_APP_NHK, HIAPID.H90_APP_NHK]:
            data = datasets_by_apid[apid]
            gattr_key = "imap_hi_l1a_hk_attrs"
        elif apid_enum in [HIAPID.H45_DIAG_FEE, HIAPID.H90_DIAG_FEE]:
            data = datasets_by_apid[apid]
            gattr_key = "imap_hi_l1a_diagfee_attrs"

        # Update dataset global attributes
        attr_mgr = ImapCdfAttributes()
        attr_mgr.add_instrument_global_attrs("hi")
        data.attrs.update(attr_mgr.get_global_attributes(gattr_key))

        # set the sensor string in Logical_source
        sensor_str = apid_enum.sensor
        data.attrs["Logical_source"] = data.attrs["Logical_source"].format(
            sensor=sensor_str
        )
        processed_data.append(data)
    return processed_data


def hi_packet_file_to_datasets(
    packet_file_path: Union[str, Path], use_derived_value: bool = False
) -> dict[int, xr.Dataset]:
    """
    Extract hi datasets from packet file.

    Parameters
    ----------
    packet_file_path : str
        L0 packet file path.
    use_derived_value : bool
        Whether to use the derived value from the XTCE definition. Default is False.

    Returns
    -------
    datasets : dict[int, xarray.Dataset]
        Dictionary of xarray datasets keyed by APID.
    """
    packet_def_file = (
        imap_module_directory / "hi/packet_definitions/TLM_HI_COMBINED_SCI.xml"
    )
    datasets_by_apid = packet_file_to_datasets(
        packet_file=packet_file_path,
        xtce_packet_definition=packet_def_file,
        use_derived_value=use_derived_value,
    )
    return datasets_by_apid


def finish_de_dataset(packets_data: xr.Dataset) -> xr.Dataset:
    """
    Unpack IMAP-Hi direct event data.

    Processing step:

    |    1. Break binary stream data into unit of 48-bits
    |    2. Parse direct event data
    |    5. Save the data into xarray dataset.

    Parameters
    ----------
    packets_data : xarray.Dataset
        Packets extracted into a dataset.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset.
    """
    de_data_dict: dict[str, list] = defaultdict(list)

    # Add packet data to the dictionary, renaming some fields
    # This is done first so that these variables are first in the CDF
    for from_key, to_key in {
        "shcoarse": "ccsds_met",
        "src_seq_ctr": "src_seq_ctr",
        "pkt_len": "pkt_len",
        "last_spin_num": "last_spin_num",
        "spin_invalids": "spin_invalids",
        "esa_step_num": "esa_step",
        "esa_step_seconds": "esa_step_seconds",
        "esa_step_milliseconds": "esa_step_milliseconds",
    }.items():
        de_data_dict[to_key] = packets_data[from_key].data

    # For each packet, parse the DE data and add it to the Pointing
    # list of DE data usint `extend()`
    for i, data in enumerate(packets_data["de_tof"].data):
        parsed_de_data = parse_direct_events(data)
        for key, new_data in parsed_de_data.items():
            de_data_dict[key].extend(new_data)
        # Record the ccsds packet index for each DE
        de_data_dict["ccsds_index"].extend([i] * len(parsed_de_data["de_tag"]))

    # create dataset
    return create_de_dataset(de_data_dict)


def create_de_dataset(de_data_dict: dict[str, npt.ArrayLike]) -> xr.Dataset:
    """
    Create Hi L1A direct event xarray dataset.

    Parameters
    ----------
    de_data_dict : Dict[list]
        Dictionary of packet telemetry and direct event data lists.

    Returns
    -------
    dataset : xarray.Dataset
        Xarray dataset.
    """
    # Load the CDF attributes
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs("hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # check_schema=False keeps DEPEND_0 = '' from being auto added
    epoch_attrs = attr_mgr.get_variable_attributes("epoch", check_schema=False)
    epoch_attrs["CATDESC"] = (
        "CCSDS creation time, number of nanoseconds since J2000 with leap "
        "seconds included"
    )
    epoch = xr.DataArray(
        met_to_ttj2000ns(de_data_dict["ccsds_met"]),
        name="epoch",
        dims=["epoch"],
        attrs=epoch_attrs,
    )

    event_met_attrs = attr_mgr.get_variable_attributes(
        "hi_de_event_met", check_schema=False
    )
    # For L1A DE, event_met is its own dimension, so we remove the DEPEND_0 attribute
    _ = event_met_attrs.pop("DEPEND_0")

    # Compute the meta-event MET in seconds
    meta_event_met = (
        np.array(de_data_dict["esa_step_seconds"]).astype(np.float64)
        + np.array(de_data_dict["esa_step_milliseconds"]) * MILLISECOND_TO_S
    )
    # Compute the MET of each event in seconds
    # event MET = meta_event_met + de_clock
    # See Hi Algorithm Document section 2.2.5
    event_met_array = np.array(
        meta_event_met[de_data_dict["ccsds_index"]]
        + np.array(de_data_dict["de_tag"]) * DE_CLOCK_TICK_S,
        dtype=event_met_attrs.pop("dtype"),
    )
    event_met = xr.DataArray(
        event_met_array,
        name="event_met",
        dims=["event_met"],
        attrs=event_met_attrs,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch, "event_met": event_met},
    )

    for var_name, data in de_data_dict.items():
        attrs = attr_mgr.get_variable_attributes(
            f"hi_de_{var_name}", check_schema=False
        ).copy()
        dtype = attrs.pop("dtype")
        dataset[var_name] = xr.DataArray(
            np.array(data, dtype=np.dtype(dtype)),
            dims=attrs["DEPEND_0"],
            attrs=attrs,
        )

    return dataset


def parse_direct_events(de_data: bytes) -> dict[str, npt.ArrayLike]:
    """
    Parse event data from a binary blob.

    IMAP-Hi direct event data information is stored in
    48-bits as follows:

    |        Read 48-bits into 16, 2, 10, 10, 10, bits. Each of these breaks
    |        down as:
    |
    |            de_tag - 16 bits
    |            start_bitmask_data - 2 bits (tA=1, tB=2, tC1=3)
    |            tof_1 - 10 bit counter
    |            tof_2 - 10 bit counter
    |            tof_3 - 10 bit counter

    There are at most total of 664 of 48-bits in each data packet.
    This data packet is of variable length. If there is one event, then
    DE_TOF will contain 48-bits. If there are 664 events, then
    DE_TOF will contain 664 x 48-bits. If there is no event, then
    DE_TOF will contain 0-bits.

    There should be two data packets per ESA. Each packet contains meta-event
    data that is identical between the two packets for a common ESA.
    If there is no event record for certain ESA step, then both packets will
    contain 0-bits in DE_TOF.

    Parameters
    ----------
    de_data : bytes
        Binary blob from de_tag field of SCI_DE packet. Must be an integer
        multiple of 48-bits of data.

    Returns
    -------
    Dict[str, list]
        Parsed event data.
    """
    # The de_data is a binary blob with Nx6 bytes of data where N = number of
    # direct events encoded into the binary blob. Interpreting the data as
    # big-endian uint16 data and reshaping into a (3, -1) ndarray results
    # in an array with shape (3, N). Indexing the first axis of that array
    # (e.g. data_uint16[i]) gives the ith 2-bytes of data for each of the N
    # direct events.
    # Considering the 6-bytes of data for each DE as 3 2-byte words,
    # each word contains the following:
    # word_0: full 16-bits is the de_tag
    # word_1: 2-bits of Trigger ID, 10-bits tof_1, upper 4-bits of tof_2
    # word_2: lower 6-bits of tof_2, 10-bits of tof_3
    data_uint16 = np.reshape(
        np.frombuffer(de_data, dtype=">u2"), (3, -1), order="F"
    ).astype(np.uint16)

    de_dict = dict()
    de_dict["de_tag"] = data_uint16[0]
    de_dict["trigger_id"] = (data_uint16[1] >> 14).astype(np.uint8)
    de_dict["tof_1"] = (data_uint16[1] & int(b"00111111_11110000", 2)) >> 4
    de_dict["tof_2"] = ((data_uint16[1] & int(b"00000000_00001111", 2)) << 6) + (
        data_uint16[2] >> 10
    )
    de_dict["tof_3"] = data_uint16[2] & int(b"00000011_11111111", 2)

    return de_dict


def finish_hist_dataset(input_ds: xr.Dataset) -> xr.Dataset:
    """
    Create dataset for a number of Hi Histogram packets.

    Parameters
    ----------
    input_ds : xarray.Dataset
        Dataset of packets generated using the
        `imap_processing.utils.packet_file_to_datasets` function.

    Returns
    -------
    dataset : xarray.Dataset
        Dataset with all metadata field data in xr.DataArray.
    """
    attr_mgr = ImapCdfAttributes()
    attr_mgr.add_instrument_global_attrs(instrument="hi")
    attr_mgr.add_instrument_variable_attrs(instrument="hi", level=None)

    # Rename shcoarse variable (do this first since it copies the input_ds)
    dataset = input_ds.rename_vars({"shcoarse": "ccsds_met"})

    dataset.epoch.attrs.update(
        attr_mgr.get_variable_attributes("epoch"),
    )
    # Add the hist_angle coordinate
    # Histogram data is binned in 90, 4-degree bins
    attrs = attr_mgr.get_variable_attributes("hi_hist_angle")
    dataset.coords.update(
        {
            "angle": xr.DataArray(
                np.arange(2, 360, 4),
                name="angle",
                dims=["angle"],
                attrs=attrs,
            )
        }
    )
    # Update existing variable attributes
    for var_name in [
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
        "ccsds_met",
        "esa_step",
        "num_of_spins",
        "cksum",
    ]:
        attrs = attr_mgr.get_variable_attributes(f"hi_hist_{var_name}")
        dataset.data_vars[var_name].attrs.update(attrs)

    new_vars = dict()
    # Populate 90-element histogram counters
    default_counter_attrs = attr_mgr.get_variable_attributes("hi_hist_counters")
    for counter_name in (*QUALIFIED_COUNTERS, *LONG_COUNTERS, *TOTAL_COUNTERS):
        # Inject counter name into generic counter attributes
        counter_attrs = default_counter_attrs.copy()
        for key, val in counter_attrs.items():
            if isinstance(val, str) and "{counter_name}" in val:
                counter_attrs[key] = val.format(counter_name=counter_name)
        # Instantiate the counter DataArray
        new_vars[counter_name] = xr.DataArray(
            data=unpack_hist_counter(input_ds[counter_name].data.sum()),
            dims=["epoch", "angle"],
            attrs=counter_attrs,
        )

    # Generate label variable for angle coordinate
    new_vars["angle_label"] = xr.DataArray(
        dataset.coords["angle"].values.astype(str),
        name="angle_label",
        dims=["angle"],
        attrs=attr_mgr.get_variable_attributes(
            "hi_hist_angle_label", check_schema=False
        ),
    )

    dataset.update(new_vars)

    return dataset


def unpack_hist_counter(counter_bytes: bytes) -> NDArray[np.uint16]:
    """
    Unpack Hi SCI_CNT counter data for a single counter.

    Parameters
    ----------
    counter_bytes : bytes
        Sum individual bytes for all epochs of a Hi SCI_CNT counter.

    Returns
    -------
    output_array : numpy.ndarray[numpy.uint16]
        The unpacked 12-bit unsigned integers for the input bytes. The
        output array has a shape of (n, 90) where n is the number of SCI_CNT
        packets in the input dataset.
    """
    # Interpret bytes for all epochs of current counter as uint8 array
    counter_uint8 = np.frombuffer(counter_bytes, dtype=np.uint8)
    # Split into triplets of upper-byte, split-byte and lower-byte arrays
    upper_uint8, split_unit8, lower_uint8 = np.reshape(
        counter_uint8, (3, -1), order="F"
    ).astype(np.uint16)
    # Compute even indexed uint12 values from upper-byte and first 4-bits of
    # split-byte
    even_uint12 = (upper_uint8 << 4) + (split_unit8 >> 4)
    # Compute odd indexed uint12 values from lower 4-bits of split-byte and
    # lower-byte
    odd_uint12 = ((split_unit8 & (2**4 - 1)) << 8) + lower_uint8
    output_array = np.column_stack((even_uint12, odd_uint12)).reshape(-1, 90)
    return output_array
