from pathlib import Path

from imap_processing import imap_module_directory
from imap_processing.decom import decom_packets

if __name__ == "__main__":
    # Define paths
    packet_file = Path("housekeeping_data.bin")
    xtce_document = Path(
        f"{imap_module_directory}/codice/packet_definitions/P_COD_NHK.xml"
    )

    # Decommutated packets
    decomposed_packets = decom_packets(packet_file, xtce_document)

    # Print decommutated packets
    for packet in decomposed_packets:
        if packet.header["PKT_APID"].raw_value == 1136:
            print(packet.data)
