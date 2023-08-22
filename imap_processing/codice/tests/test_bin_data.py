''' This code is used to test the size of a binary file'''

def test_bin_file_size():
    # Replace this path with the full path to your .bin file
    bin_file_path = "/Users/gamo6782/Desktop/RAW.bin"

    with open(bin_file_path, "rb") as file:
        data = file.read()
        file_size = len(data)
        print(f"The size of the .bin file is: {file_size} bytes")


#if __name__ == "__main__":
    #test_bin_file_size()

################################

def read_binary_file(file_path):
    with open(file_path, "rb") as file:
        data = file.read()
    return data

def extract_header(packet_data):
    # Extract the header from the packet_data
    header = packet_data[:6]
    return header

def extract_data_payload(packet_data):
    # Extract the data payload from the packet_data
    data_payload = packet_data[6:]
    return data_payload

def binary_to_int(binary_data):
    # Convert binary data to an integer
    return int.from_bytes(binary_data, byteorder='big')

if __name__ == "__main__":
    bin_file_path = "/Users/gamo6782/Desktop/RAW.bin"  # Replace this with your .bin file path

    data = read_binary_file(bin_file_path)
    packet_length_bytes = binary_to_int(data[4:6]) + 1  # Convert the 16-bit packet length to integer
    header = extract_header(data)
    data_payload = extract_data_payload(data)

    print(f"Header (in hex): {header.hex()}")
    print(f"Data Payload (in hex): {data_payload.hex()}")
    print(f"Total Packet Size (in bytes): {packet_length_bytes}")
