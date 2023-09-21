import logging

import xarray as xr

from imap_processing import decom
from imap_processing.hit.l0.hit_apid import HitAPID

logging.basicConfig(level=logging.INFO)


def decom_hit_packets(packet_file: str, xtce: str):
    """
    Unpack and decode IMAP-Lo packets using CCSDS format and XTCE packet definitions.
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

    unpacked_data = {}
    for apid in [id.value for id in HitAPID]:
        # TODO: if science packet, do decompression
        apid_name = HitAPID(apid).name
        logging.info(f"Getting packet values for {apid_name}:{apid}")
        # get all the packets for this apid and groups them together in a list
        unpacked_data[apid_name] = group_apid_data(packets, apid)
        # sort all the packets in the list by their spacecraft time
        unpacked_data[apid_name] = sorted(
            unpacked_data[apid_name], key=lambda x: x["SHCOARSE"]
        )
        # merge all the packets for this apid together
        unpacked_data[apid_name] = merge_data(unpacked_data[apid_name])

    # create datasets
    dataset_dict = create_datasets(unpacked_data)
    return dataset_dict


def create_datasets(data):
    """
    Creates a dataset for each APID in the data
    ----------
    data : dict
        A single dictionary containing data for all instances of an APID.
    Returns
    -------
    dict
        A dictionary containing xr.Dataset for each APID. each dataset in the
        dictionary will be converted to a CDF.
    """
    dataset_dict = {}
    # create one dataset for each APID in the data
    for apid, data_dict in data.items():
        # if data for the APID exists, create the dataset
        if data_dict != {}:
            epoch = xr.DataArray(
                name="Epoch", data=data_dict["SHCOARSE"], dims=("Epoch")
            )
            dataset = xr.Dataset(data_vars={}, coords={"Epoch": epoch})
            data_dict.pop("SHCOARSE")
            dataset_dict[apid] = dataset.assign(**data_dict)

    return dataset_dict


def merge_data(data_list):
    """
    Merge data dictionaries for each mactching APID into
    a single dictionary. For example:
    [{'SHCOARSE':400, ...}, {'SHCOARSE':500, ...}, ...]
    becomes
    {'SHCOARSE':[400, 500, ...], ...}
    ----------
    data_list : list
        list containing a dictionary with data for each instance a matching APID.
        For example, if a CCSDS packet has N instances of APID X, this list
        should contain N dictionaries all containing data for APID X.
    Returns
    -------
    dict
        A single dictionary containing data for all instances of an APID.
    """
    merged_data = {}
    for data in data_list:
        for field, value in data.items():
            if field not in merged_data:
                merged_data[field] = [value]
            else:
                merged_data[field].append(value)
    return merged_data


def group_apid_data(packets, apid):
    """
    Creates a list of dictionaries containing data for each instance
    of a specified APID
    ----------
    packets: list
        List of all the unpacked data from decom.decom_packets()
    apid: int
        APID number for the data you want to group together
    Returns
    -------
    list
        list containing a dictionary with data for each instance the specified APID.
        For example, if a CCSDS packet has N instances of APID X, this list
        will contain N dictionaries all containing data for APID X.
    """
    all_data = []
    for packet in packets:
        if packet.header["PKT_APID"].derived_value == apid:
            data_dict = {}
            # get a list of all the APID fields
            packet_fields = list(packet.data.keys())
            for field in packet_fields:
                # put the value of the field in a dictionary
                data_dict[field] = packet.data[field].derived_value
            # put the data dictionary in a list
            all_data.append(data_dict)
    return all_data
