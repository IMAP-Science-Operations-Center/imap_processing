''' This function reads in a binary file and returns the binary data as a byte array'''


def read_binary_file(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
    return data


''' This function extracts the header from the packet data. The CCSDS header is 6 bytes long and 
located at the beginning of the packet data'''


def extract_header(packet_data):
    # Extract the header from the packet_data
    header = packet_data[:6]
    return header


''' This function extracts the data payload from the packet data. 
The data payload is the part of the packet after the header'''


def extract_data_payload(packet_data):
    # Extract the data payload from the packet_data (excluding the header)
    data_payload = packet_data[6:]
    return data_payload


''' This function searches for a packet with the specified APID (Application ID) in the binary data. 
If found, it returns the byte index where the packet starts.'''


def find_packet_with_apid(bin_data, target_apid):
    # Search for the target APID in the binary data
    target_apid_bytes = target_apid.to_bytes(2, byteorder='big')

    idx = bin_data.find(target_apid_bytes)

    return idx


''' This function is a more detailed version of the previous extract_header() function. 
It extracts specific fields from the CCSDS header, such as version number, packet type, secondary header flag, 
application ID, group flag, sequence count, and data length. It then returns these header fields as a dictionary.'''


def extract_ccsds_header(packet_data):
    header = packet_data[:6]

    phverno = int.from_bytes(header[0:1], byteorder='big') >> 5
    phtype = (int.from_bytes(header[0:1], byteorder='big') >> 4) & 0b1
    phshf = (int.from_bytes(header[0:1], byteorder='big') >> 3) & 0b1
    phapid = int.from_bytes(header[0:2], byteorder='big') & 0x7FF
    phgroupf = (int.from_bytes(header[2:3], byteorder='big') >> 6) & 0b11
    phseqcnt = int.from_bytes(header[2:4], byteorder='big') & 0x3FFF
    phdlen = int.from_bytes(header[4:6], byteorder='big')

    ccsds_header = {
        "PHVERNO": phverno,
        "PHTYPE": phtype,
        "PHSHF": phshf,
        "PHAPID": phapid,
        "PHGROUPF": phgroupf,
        "PHSEQCNT": phseqcnt,
        "PHDLEN": phdlen
    }

    return ccsds_header


'''    1. The binary file is read using the read_binary_file() function.
        2. The index of a packet with the target APID (in this case, APID 96) is found using the 
                find_packet_with_apid() function.
            3. If a packet with the target APID is found, the code extracts the packet header and creates 
                a dictionary of the CCSDS header fields using the extract_ccsds_header() function.
                    4. The extracted CCSDS header fields are printed.'''

if __name__ == "__main__":
    bin_file_path = "/Users/gamo6782/Desktop/RAW.bin"  # Replace this with your .bin file path

    data = read_binary_file(bin_file_path)
    idx = find_packet_with_apid(data, target_apid=96)

    if idx >= 0:
        print(f"Packet with APID found at byte index: {idx}")

        # Extract the packet header and data payload
        packet_data = data[idx:]
        ccsds_header = extract_ccsds_header(packet_data)

        print("CCSDS Header:")
        for field, value in ccsds_header.items():
            print(f"{field}: {value}")
    else:
        print("Packet with APID not found in the binary data.")
