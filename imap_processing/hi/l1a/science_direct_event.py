"""IMAP-Hi direct event processing."""

import numpy as np
import xarray as xr

from imap_processing import launch_time
from imap_processing.cdf.global_attrs import ConstantCoordinates
from imap_processing.hi import hi_cdf_attrs

# TODO: read LOOKED_UP_DURATION_OF_TICK from
# instrument status summary later. This value
# is rarely change but want to be able to change
# it if needed. It stores information about how
# fast the time was ticking. It is in microseconds.
LOOKED_UP_DURATION_OF_TICK = 3999


def get_direct_event_time(met_seconds: int, met_subseconds: int, de_tag: int):
    """Create MET(Mission Elapsed Time) time using input times.

    Parameters
    ----------
    met_seconds : int
        MET time in seconds
    met_subseconds : int
        MET subseconds in milliseconds
    de_tag : int
        Direct event time tag in ticks

    Returns
    -------
    met_datetime : numpy.datetime64
        Human-readable MET time
    """
    # Combine these direct event times to DE MET time:
    #   seconds (from metaevent)
    #   subseconds (milliseconds) (from metaevent)
    #   de_tag (milliseconds) (from direct event)
    #   LOOKED_UP_DURATION_OF_TICK (milliseconds)

    time_in_ns = (
        met_seconds * 1e9
        + met_subseconds * 1e6
        + de_tag * LOOKED_UP_DURATION_OF_TICK * 1e3
    )

    met_datetime = launch_time + np.timedelta64(int(time_in_ns), "ns")

    return met_datetime


def parse_direct_event(event_data: str):
    """Parse event data.

    IMAP-Hi direct event data information is stored in
    48-bits as follow:
        Read first two bits (start_bitmask_data) to find
        out which type of event it is. start_bitmask_data value mapping:
            1 - A
            2 - B
            3 - C
            0 - META
        If it's a metaevent:
            Read 48-bits into 2, 4, 10, 32 bits. Each of these breaks
            down as:
                start_bitmask_data - 2 bits (tA=1, tB=2, tC1=3, META=0)
                ESA step - 4 bits
                integer millisecond of MET(subseconds) - 10 bits
                integer MET(seconds) - 32 bits
        If it's not a metaevent:
            Read 48-bits into 2, 10, 10, 10, 16 bits. Each of these breaks
            down as:
                start_bitmask_data - 2 bits (tA=1, tB=2, tC1=3, META=0)
                tof_1 - 10 bit counter
                tof_2 - 10 bit counter
                tof_3 - 10 bit counter
                de_tag - 16 bits

    There are at most total of 665 of 48-bits in each data packet.
    This data packet is of variable length. If there is one event, then
    DE_TOF will contain 48-bits. If there are 665 events, then
    DE_TOF will contain 665 x 48-bits. If there is no event, then
    DE_TOF will contain 0-bits.

    Per ESA, there should be two data packets. First one will begin with
    metaevent followed by direct events data. Second one will begin with
    direct event data only. If there is no event record for certain ESA step,
    then as mentioned above, first packet will contain metaevent in DE_TOF
    information and second packet will contain 0-bits in DE_TOF. In general,
    every two packets will look like this.
        first packet = [
            (start_bitmask_data, ESA step, int millisecond of MET, int MET),
            (start_bitmask_data, tof_1, tof_2, tof_3, de_tag),
            .....
        ]
        second packet = [
            (start_bitmask_data, tof_1, tof_2, tof_3, de_tag),
            .....
        ]

    In direct event data, if no hit is registered, the tof_x field in
    the DE to a value of negative one. However, since the field is described as a
    "10-bit unsigned counter," it cannot actually store negative numbers.
    Instead, the value negative is represented by the maximum value that can
    be stored in a 10-bit unsigned integer, which is 0x3FF (in hexadecimal)
    or 1023 in decimal. This value is used as a special marker to
    indicate that no hit was registered. Ideally, the system should
    not be able to register a hit with a value of 1023 for all
    tof_1, tof_2, tof_3, because this is in case of an error. But,
    IMAP-Hi like to process it still to investigate the data.
    Example of what it will look like if no hit was registered.
            (start_bitmask_data, 1023, 1023, 1023, de_tag)
            start_bitmask_data will be 1 or 2 or 3.

    Parameters
    ----------
    event_data : str
        48-bits Event data

    Returns
    -------
    dict
        Parsed event data
    """
    if int(event_data[:2]) == 0:
        # parse metaevent
        metaevent = {
            "start_bitmask_data": int(event_data[:2], 2),
            "esa_step": int(event_data[2:6], 2),
            "subseconds": int(event_data[6:16], 2),
            "seconds": int(event_data[16:], 2),
        }
        return metaevent

    # parse direct event
    direct_event = {
        "start_bitmask_data": int(event_data[:2], 2),
        "tof_1": int(event_data[2:12], 2),
        "tof_2": int(event_data[12:22], 2),
        "tof_3": int(event_data[22:32], 2),
        "de_tag": int(event_data[32:], 2),
    }
    return direct_event


def break_into_bits_size(binary_data: str):
    """Break binary stream data into 48-bits.

    Parameters
    ----------
    binary_data : str
        Binary data

    Returns
    -------
    list
        List of 48-bits
    """
    # TODO: ask Paul what to do if the length of
    # binary_data is not a multiple of 48
    return [binary_data[i : i + 48] for i in range(0, len(binary_data), 48)]


def create_dataset(de_data_list: list, packet_met_time: list):
    """Create xarray dataset.

    Parameters
    ----------
    de_data_list : list
        Parsed direct event data list
    packet_met_time : list
        List of packet MET time

    Returns
    -------
    dataset: xarray.Dataset
        xarray dataset
    """
    # These are the variables that we will store in the dataset
    de_met_time = []
    esa_step = []
    trigger_id = []
    tof_1 = []
    tof_2 = []
    tof_3 = []
    de_tag = []
    ccsds_met = []

    # How to handle if first event is not metaevent? This
    # means that current data file started with direct event.
    # Per Paul, log a warning and discard all direct events
    # until we see next metaevent because it could mean
    # that the instrument was turned off during repoint.

    # Find the index of the first occurrence of the metaevent
    first_metaevent_index = next(
        (i for i, d in enumerate(de_data_list) if d.get("start_bitmask_data") == 0),
        None,
    )

    if first_metaevent_index is None:
        return None
    elif first_metaevent_index != 0:
        # Discard all direct events until we see next metaevent
        # TODO: log a warning
        de_data_list = de_data_list[first_metaevent_index:]
        packet_met_time = packet_met_time[first_metaevent_index:]

    for index, event in enumerate(de_data_list):
        if event["start_bitmask_data"] == 0:
            # metaevent is a way to store information
            # about bigger portion of time information. Eg.
            # metaevent stores information about, let's say
            # "20240319T09:30:01.000". Then direct event time
            # tag stores information of time ticks since
            # that time. Then we use those two to combine and
            # get exact time information of each event.

            # set time and esa step values to
            # be used for direct event followed by
            # this metaevent
            int_subseconds = event["subseconds"]
            int_seconds = event["seconds"]
            current_esa_step = event["esa_step"]
            # Add half a tick once per algorithm document
            # and Paul Janzen.
            haf_tick = LOOKED_UP_DURATION_OF_TICK // 2
            # convert microseconds to millieseconds to
            # match subseconds time format
            half_tick_ms = haf_tick // 1e3
            int_subseconds += half_tick_ms
            continue

        # calculate direct event time using time information from metaevent
        # and de_tag. epoch in this dataset uses this time of the event
        de_time = get_direct_event_time(int_seconds, int_subseconds, event["de_tag"])
        de_met_time.append(de_time)
        esa_step.append(current_esa_step)
        # start_bitmask_data is 1, 2, 3 for detector A, B, C
        # respectively. This is used to identify which detector
        # was hit first for this current direct event.
        trigger_id.append(event["start_bitmask_data"])
        tof_1.append(event["tof_1"])
        tof_2.append(event["tof_2"])
        tof_3.append(event["tof_3"])
        # IMAP-Hi like to keep de_tag value for diagnostic purposes
        de_tag.append(event["de_tag"])
        # add packet time to ccsds_met list
        ccsds_met.append(packet_met_time[index])

    epoch_time = xr.DataArray(
        de_met_time,
        name="epoch",
        dims=["epoch"],
        attrs=ConstantCoordinates.EPOCH,
    )

    dataset = xr.Dataset(
        coords={"epoch": epoch_time},
        attrs=hi_cdf_attrs.hi_de_l1a_attrs.output(),
    )

    dataset["esa_step"] = xr.DataArray(
        esa_step, dims="epoch", attrs=hi_cdf_attrs.esa_step_attrs.output()
    )
    dataset["trigger_id"] = xr.DataArray(
        trigger_id, dims="epoch", attrs=hi_cdf_attrs.trigger_id_attrs.output()
    )
    dataset["tof_1"] = xr.DataArray(
        tof_1, dims="epoch", attrs=hi_cdf_attrs.tof_attrs.output()
    )
    dataset["tof_2"] = xr.DataArray(
        tof_2, dims="epoch", attrs=hi_cdf_attrs.tof_attrs.output()
    )
    dataset["tof_3"] = xr.DataArray(
        tof_3, dims="epoch", attrs=hi_cdf_attrs.tof_attrs.output()
    )
    dataset["de_tag"] = xr.DataArray(
        de_tag, dims="epoch", attrs=hi_cdf_attrs.de_tag_attrs.output()
    )
    dataset["ccsds_met"] = xr.DataArray(
        ccsds_met, dims="epoch", attrs=hi_cdf_attrs.ccsds_met_attrs.output()
    )
    # TODO: figure out how to store information about
    # input data(one or more) it used to produce this dataset
    return dataset


def science_direct_event(packets_data: list):
    """Unpack IMAP-Hi direct event data.

    Processing step:
        1. Break binary stream data into unit of 48-bits
        2. Parse direct event data
        5. Save the data into xarray dataset.

    Parameters
    ----------
    packets_data : list
        List of packets data

    Returns
    -------
    dataset: xarray.Dataset
        xarray dataset
    """
    de_data_list = []
    packet_met_time = []
    # TODO: ask Paul if he wants MET time in the dataset
    # If so, get MET time of every packets

    # Because DE_TOF is a variable length data,
    # I am using extend to add another list to the
    # end of the list. This way, I don't need to flatten
    # the list later.
    for data in packets_data:
        # break binary stream data into unit of 48-bits
        event_48bits_list = break_into_bits_size(data.data["DE_TOF"].raw_value)
        # parse 48-bits into meaningful data such as metaevent or direct event
        de_data_list.extend([parse_direct_event(event) for event in event_48bits_list])
        # add packet time to packet_met_time
        packet_met_time.extend(
            [data.data["CCSDS_MET"].raw_value] * len(event_48bits_list)
        )

    # create dataset
    return create_dataset(de_data_list, packet_met_time)
