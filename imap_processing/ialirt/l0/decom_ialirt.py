import logging
from enum import Enum

import bitstring
import xarray as xr
from space_packet_parser import parser, xtcedef

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


class PacketLength(Enum):
    EXPECTED_LENGTH = 1464


def decom_packets(packet_file, xtce_packet_definition):
    """
    Unpack data packet. In this function, we unpack and return data
    as it is. Data modification will not be done at this step.

    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename
    xtce_packet_definition : str
        Path to XTCE file with filename

    Returns
    -------
    List
        List of all the unpacked data
    """

    packet_definition = xtcedef.XtcePacketDefinition(xtce_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    packets = []

    with open(packet_file) as file:
        for line_number, line in enumerate(file, 1):
            if not line.startswith("#"):
                # Split the line by semicolons
                # Discard the first value since it is only a counter
                hex_values = line.strip().split(";")[1::]

                binary_values = ""
                for h in hex_values:
                    # Convert hex to integer
                    # 16 is the base of hexadecimal
                    int_value = int(h, 16)

                    # Convert integer to binary and remove the '0b' prefix
                    bin_value = bin(int_value)[2:]

                    # Make sure each binary string is 8 bits long
                    bin_value_padded = bin_value.zfill(8)

                    # Append the padded binary string to the final string
                    binary_values += bin_value_padded

                # Check the length of binary_values
                if len(binary_values) != PacketLength.EXPECTED_LENGTH.value:
                    error_message = f"Error on line {line_number}: " \
                                    f"Length of binary_values (" \
                                    f"{len(binary_values)}) does not equal " \
                                    f"{PacketLength.EXPECTED_LENGTH.value}."
                    logger.error(error_message)
                    raise ValueError(error_message)

                packet_generator = packet_parser.generator(
                    bitstring.ConstBitStream(bin=binary_values))

                for packet in packet_generator:
                    packets.append(packet)

    return packets


def generate_xarray(packet_file: str, xtce: str):
    """
    Generate xarray from unpacked data.

    Parameters
    ----------
    packet_file : str
        Path to the CCSDS data packet file.
    xtce : str
        Path to the XTCE packet definition file.

    Returns
    -------
    xr.Dataset
        A dataset containing the decoded data fields with 'time' as the coordinating
        dimension.
    """

    try:
        packets = decom_packets(packet_file, xtce)
    except Exception as e:
        logger.error(f"Error during packet decomposition: {str(e)}")
        return

    if not packets:
        logger.warning(f"No packets found in {packet_file}.")
        return

    logger.info(f"Decomposed {len(packets)} packets from {packet_file}.")

    # List of instruments and their corresponding MET keys
    instruments = ['SC', 'HIT', 'MAG', 'COD_LO', 'COD_HI', 'SWE', 'SWAPI']
    instrument_coords = ['SC_SCLK_SEC', 'HIT_SC_TICK', 'MAG_ACQ', 'COD_LO_ACQ',
                         'COD_HI_ACQ', 'SWE_ACQ_SEC', 'SWAPI_ACQ']

    # Create a dictionary mapping each instrument to its time-dimension key
    time_keys = dict(zip(instruments, instrument_coords))

    # Initialize storage dictionary
    data_storage = {inst: {} for inst in instruments}

    for packet in packets:
        for key, value in packet.data.items():
            for inst in instruments:
                if key.startswith(inst):
                    if key not in data_storage[inst]:
                        data_storage[inst][key] = []
                    data_storage[inst][key].append(value.derived_value)
                    break
            else:
                logger.warning(f"Unexpected key '{key}' found in packet data.")

    logger.info("Generating datasets for each instrument.")

    # Generate xarray dataset for each instrument and spacecraft
    datasets = {}
    for inst in instruments:
        dataset_dict = {key: (time_keys[inst], value)
                        for key, value in data_storage[inst].items() if
                        key != time_keys[inst]}
        datasets[inst] = xr.Dataset(dataset_dict, coords={
            time_keys[inst]: data_storage[inst][time_keys[inst]]})

    logger.info(f"Generated datasets for {len(datasets)} instruments.")

    return datasets
