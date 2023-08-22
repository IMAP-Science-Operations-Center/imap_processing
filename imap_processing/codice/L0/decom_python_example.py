"""
Reads the contents of a binary file and returns the data as a bytes object.

Parameters:
    file_path (str): The path to the binary file to be read.
    encoding (str, optional): The character encoding to use for decoding the binary data.

Returns:
    bytes or str: The binary data read from the file as a bytes object, or decoded to a string if an encoding is provided.

Raises:
    FileNotFoundError: If the specified file does not exist.
    IOError: If an error occurs while reading the file.


"""


def read_binary_file(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
    return data


"""
Extracts a value from binary data by interpreting a specified range of bits.

This function is used to extract a value from a sequence of binary data by specifying the starting bit position and the number of bits to consider. The bits are interpreted as an unsigned integer value.

Parameters:
    data (bytes): The binary data from which the value will be extracted.
    start_bit (int): The index of the starting bit for extraction.
    length (int): The number of bits to extract.

Returns:
    int: The extracted value represented as an integer.


"""


def extract_bits(data, start_bit, length):
    byte_offset = start_bit // 8
    bit_shift = start_bit % 8
    mask = (1 << length) - 1

    value = 0
    for i in range(length):
        byte = data[byte_offset + i]
        value |= (byte >> bit_shift) << (i * 8)

    return value & mask


"""
Finds the index of the first occurrence of a packet with a specific APID in binary data.

This function searches through a sequence of binary data to find the index of the first packet that matches the specified Application Process Identifier (APID). The APID is a unique identifier used in packet-based data communication protocols.

Parameters:
    bin_data (bytes): The binary data to search within.
    target_apid (int): The target APID to search for.

Returns:
    int: The index of the first occurrence of the packet with the specified APID, or -1 if not found.

Example:
    binary_data = bytes([0x12, 0x34, 0x56, 0x12, 0x78, 0x90])  # Example binary data
    target_apid = 0x1234  # Example target APID
    packet_index = find_packet_with_apid(binary_data, target_apid)
    # In this example, the target APID 0x1234 is found at index 0 in the binary_data.
    # Therefore, packet_index will be 0.
"""


def find_packet_with_apid(bin_data, target_apid):
    # Search for the target APID in the binary data
    target_apid_bytes = target_apid.to_bytes(2, byteorder='big')
    idx = bin_data.find(target_apid_bytes)

    return idx


"""
Decommutes packet data using a provided decommutation table.

This function takes a packet's binary data and a decommutation table as input, and returns a dictionary of parameter values extracted from the packet according to the table.

Parameters:
    packet_data (bytes): Binary data of the packet to decommute.
    decomm_table (list): List of dictionaries, each containing decommutation information for a parameter.
        Each dictionary should contain:
            - "mnemonic": A unique identifier for the parameter.
            - "sequence": An optional parameter sequence number.
            - "startByte": Starting byte index in the packet.
            - "startBitInByte": Starting bit index within the starting byte.
            - "startBit": Overall starting bit index in the packet.
            - "lengthInBits": Number of bits to extract for this parameter.
            - "dataType": Data type of the parameter, e.g., "unsigned_int", "float", etc.

Returns:
    dict: A dictionary containing extracted parameter values with their respective mnemonics as keys.

Example:
    packet_binary_data = bytes([0b11011010, 0b10101100, 0b01010101])  # Example packet binary data
    decommutation_info = [
        {"mnemonic": "PARAM1", "sequence": 1, "startByte": 0, "startBitInByte": 2, "lengthInBits": 6, "dataType": "unsigned_int"},
        {"mnemonic": "PARAM2", "sequence": 2, "startByte": 1, "startBitInByte": 0, "lengthInBits": 8, "dataType": "unsigned_int"}
    ]
    decommuted_parameters = decommute_packet(packet_binary_data, decommutation_info)
    # In this example, two parameters are extracted using the provided decommutation info.
    # PARAM1 will be extracted from bits 2 to 7, and PARAM2 from bits 8 to 15 of the packet.
"""


def decommute_packet(packet_data, decomm_table):
    parameters = {}

    for entry in decomm_table:
        mnemonic = entry["mnemonic"]
        sequence = entry["sequence"]
        start_byte = entry["startByte"]
        start_bit_in_byte = entry["startBitInByte"]
        start_bit = entry["startBit"]
        length_in_bits = entry["lengthInBits"]
        data_type = entry["dataType"]

        value = extract_bits(packet_data, start_bit, length_in_bits)
        parameters[mnemonic] = value

    return parameters


"""
Main script to extract and decommute parameters from a binary file containing packets.

This script reads a binary file containing packet data, searches for a packet with a specific Application Process Identifier (APID),
and then decommutes the packet's parameters using a provided decommutation table. The extracted parameter values are printed.

Usage:
    1. Set the 'bin_file_path' variable to the path of the binary file containing packet data.
    2. Replace 'target_apid' with the desired APID to search for.
    3. Define the 'decomm_table' with the decommutation information for different parameters.
    4. Run the script to extract and print decommuted parameter values.

Note:
    - The 'read_binary_file', 'find_packet_with_apid', and 'decommute_packet' functions are assumed to be defined elsewhere.

Example:
    Assuming 'read_binary_file' and other functions are defined:
    - Given a binary file at 'bin_file_path' and a desired 'target_apid':
    - If a packet with the 'target_apid' is found, its parameters are extracted and printed.
    - If no matching packet is found, a message indicating such is printed.
"""

if __name__ == "__main__":
    bin_file_path = "/Users/gamo6782/Desktop/RAW.bin"
    target_apid = 0x460  # Replace with the APID of the desired packet
    decomm_table = [
        {
            "mnemonic": "SHCOARSE",
            "sequence": 7,
            "startByte": 6,
            "startBitInByte": 0,
            "startBit": 48,
            "lengthInBits": 32,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Spare",
            "sequence": 8,
            "startByte": 10,
            "startBitInByte": 0,
            "startBit": 80,
            "lengthInBits": 6,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Power_Cycle_Rq",
            "sequence": 9,
            "startByte": 10,
            "startBitInByte": 6,
            "startBit": 86,
            "lengthInBits": 1,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Power_Off_Rq",
            "sequence": 10,
            "startByte": 10,
            "startBitInByte": 7,
            "startBit": 87,
            "lengthInBits": 1,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Heater_Control_Enabled",
            "sequence": 11,
            "startByte": 11,
            "startBitInByte": 0,
            "startBit": 88,
            "lengthInBits": 1,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Heater_1_State",
            "sequence": 12,
            "startByte": 11,
            "startBitInByte": 1,
            "startBit": 89,
            "lengthInBits": 1,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Heater_2_State",
            "sequence": 13,
            "startByte": 11,
            "startBitInByte": 2,
            "startBit": 90,
            "lengthInBits": 1,
            "dataType": "UINT"
        },
        {
            "mnemonic": "Spare2",
            "sequence": 14,
            "startByte": 11,
            "startBitInByte": 3,
            "startBit": 91,
            "lengthInBits": 5,
            "dataType": "UINT"
        },
    ]

    data = read_binary_file(bin_file_path)
    idx = find_packet_with_apid(data, target_apid)

    if idx >= 0:
        packet_data = data[idx:]
        parameters = decommute_packet(packet_data, decomm_table)
        print("Decommuted Parameters of Apid:")
        for mnemonic, value in parameters.items():
            print(f"{mnemonic}: {value}")
    else:
        print("Packet with APID not found in the binary data.")
