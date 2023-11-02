import logging
from collections import defaultdict
from enum import IntEnum

import xarray as xr

from imap_processing import decom

logging.basicConfig(level=logging.INFO)


class HitAPID(IntEnum):
    """
    HIT APID Mappings.

    Attributes
    ----------
    HIT_AUT : int
        Autonomy
    HIT_HSKP: int
        Housekeeping
    HIT_SCIENCE : int
        Science
    HIT_IALRT : int
        I-ALiRT
    HIT_MEMDUMP : int
        Memory dump
    """

    HIT_AUT = 1250  # Autonomy
    HIT_HSKP = 1251  # Housekeeping
    HIT_SCIENCE = 1252  # Science
    HIT_IALRT = 1253  # I-ALiRT
    HIT_MEMDUMP = 1255  # Memory dump


def decom_hit_packets(packet_file: str, xtce: str):
    """
    Unpack and decode HIT packets using CCSDS format and XTCE packet definitions.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.

    Returns
    -------
    dict
        A dictionary containing xr.Dataset for each APID. each dataset in the
        dictionary will be converted to a CDF.
    """
    # TODO: XTCE Files need to be combined
    logging.info(f"Unpacking {packet_file} using xtce definitions in {xtce}")
    packets = decom.decom_packets(packet_file, xtce)
    logging.info(f"{packet_file} unpacked")
    # print(packets[0])
    # sort all the packets in the list by their spacecraft time
    sorted_packets = sorted(packets, key=lambda x: x.data["SHCOARSE"].derived_value)

    # Store data for each apid
    # unpacked_data =
    #   {apid0: {var0: [item0, item1, ...], var1: [item0, item1, ...]}, ...}
    unpacked_data = {}
    for apid_name, apid in [(id.name, id.value) for id in HitAPID]:
        # TODO: if science packet, do decompression
        logging.info(f"Grouping packet values for {apid_name}:{apid}")
        # get all the packets for this apid and groups them together in a
        # dictionary
        unpacked_data[apid_name] = group_apid_data(sorted_packets, apid)
        logging.info(f"Finished grouping {apid_name}:{apid} packet values")

    # create datasets
    logging.info("Creating a dataset for HIT L1A data")
    dataset_dict = create_datasets(unpacked_data)
    logging.info("HIT L1A dataset created")
    return dataset_dict


def create_datasets(data):
    """
    Create a dataset for each APID in the data.

    Parameters
    ----------
    data : dict
        A single dictionary containing data for all instances of an APID.

    Returns
    -------
    dict
        A dictionary containing xr.Dataset for each APID. each dataset in the
        dictionary will be converted to a CDF.
    """
    dataset_dict = defaultdict(list)
    # create one dataset for each APID in the data
    for apid, data_dict in data.items():
        # if data for the APID exists, create the dataset
        if data_dict != {}:
            epoch = xr.DataArray(
                name="Epoch", data=data_dict.pop("SHCOARSE"), dims=("Epoch")
            )
            dataset = xr.Dataset(data_vars={}, coords={"Epoch": epoch})
            dataset_dict[apid] = dataset.assign(**data_dict)

    return dataset_dict


def group_apid_data(packets, apid):
    """
    Create a dictionary of lists containing all the data for the APID.

    If packets contain N of the same APIDs, the data
    for those N matching APIDs will be grouped together into
    a dictionary of lists.

    Parameters
    ----------
    packets : list
        List of all the unpacked data from decom.decom_packets()
    apid : int
        APID number for the data you want to group together

    Returns
    -------
    dict
        A dictionary where each field in the specified APID
        is a key, and the value for that key is a list of
        that fields values in all packets within the CCSDS file
    """
    data_dict = defaultdict(list)
    for packet in packets:
        if packet.header["PKT_APID"].derived_value == apid:
            for field in packet.data:
                # put the value of the field in a dictionary
                data_dict[field].append(packet.data[field].derived_value)
    return data_dict
