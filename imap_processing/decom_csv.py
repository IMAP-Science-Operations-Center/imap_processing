from space_packet_parser import parser, xtcedef
from space_packet_parser.csvdef import CsvPacketDefinition

def decom_csv_packets(packet_file: str, csv_packet_definition: str):
    """Unpack CCSDS data packet. In this function, we unpack and return data
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
    packet_definition = CsvPacketDefinition(csv_packet_definition)
    packet_parser = parser.PacketParser(packet_definition)

    with open(packet_file, "rb") as binary_data:
        packet_generator = packet_parser.generator(binary_data)

        return list(packet_generator)

import csv
import os

def rename_header_in_csv(file_path, old_header, new_header):
    """
    Rename a header in a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        old_header (str): The header name you want to replace.
        new_header (str): The new header name you want to set.

    Returns:
        bool: True if the header was found and renamed, False otherwise.
    """

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found!")
        return False

    # Read the CSV content
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Check if the old header exists in the first row (headers)
    if old_header in rows[0]:
        header_index = rows[0].index(old_header)
        rows[0][header_index] = new_header
    else:
        print(f"Header '{old_header}' not found in '{file_path}'!")
        return False

    # Write the modified content back to the CSV
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

    print(f"Header '{old_header}' has been renamed to '{new_header}' in '{file_path}'!")
    return True
