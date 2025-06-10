"""IMAP-Hi direct event processing."""

import logging
from collections import defaultdict

import numpy as np
import numpy._typing as npt
import xarray as xr

from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.spice.time import met_to_ttj2000ns

# TODO: read DE_CLOCK_TICK_US from
# instrument status summary later. This value
# is rarely change but want to be able to change
# it if needed. It stores information about how
# fast the time was ticking. It is in microseconds.
DE_CLOCK_TICK_US = 1999
DE_CLOCK_TICK_S = DE_CLOCK_TICK_US / 1e6
HALF_CLOCK_TICK_S = DE_CLOCK_TICK_S / 2

MILLISECOND_TO_S = 1e-3

logger = logging.getLogger(__name__)


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


def create_dataset(de_data_dict: dict[str, npt.ArrayLike]) -> xr.Dataset:
    """
    Create xarray dataset.

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
        np.array(de_data_dict["meta_seconds"]).astype(np.float64)
        + np.array(de_data_dict["meta_subseconds"]) * MILLISECOND_TO_S
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


def science_direct_event(packets_data: xr.Dataset) -> xr.Dataset:
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
        "meta_seconds": "meta_seconds",
        "meta_subseconds": "meta_subseconds",
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
    return create_dataset(de_data_dict)
