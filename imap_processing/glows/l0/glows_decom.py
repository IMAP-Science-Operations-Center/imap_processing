from pathlib import Path

from bitstring import ReadError
from space_packet_parser import parser, xtcedef
from space_packet_parser.parser import ParsedDataItem

from imap_processing import imap_module_directory


def decom_packets(packet_file: str) -> list[ParsedDataItem]:
    """Decom GLOWS data packets using GLOWS packet definition
    Parameters
    ----------
    packet_file : str
        Path to data packet path with filename

    Returns
    -------
    List
        List of all the unpacked data
    """

    hist_apid = 1480
    de_apid = 1481

    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/P_GLX_TMSCHIST.xml"
    )

    hist_packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    histparser = parser.PacketParser(hist_packet_definition, hist_apid)

    with open(packet_file, "rb") as binary_data:
        try:
            hist_packets = histparser.generator(
                binary_data,
                parse_bad_pkts=False,
                buffer_read_size_bytes=5790778,
                show_progress=True,
            )

            for packet in hist_packets:
                # Do something with the packet data
                if packet.header["PKT_APID"].derived_value == hist_apid:
                    print(f"Decommed histogram packet: {packet.header}")
                    # print(f"full packet data: {packet.data}")
                if packet.header["PKT_APID"].derived_value == de_apid:
                    print(f"Decommed DE packet: {packet.header}")
                # print(packet.data)
        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        return list(hist_packets)


def process_packets(packet_list: list[ParsedDataItem]):
    for packet in packet_list:
        print(packet.derived_value)


if __name__ == "__main__":
    decom_packets("../tests/imap_l0_sci_glows_20230920_v00.pcts")
