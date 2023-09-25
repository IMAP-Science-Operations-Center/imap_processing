from pathlib import Path

from bitstring import ReadError
from glows_l0_data import GlowsDeL0, GlowsHistL0
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory


def decom_packets(packet_file: str) -> list[GlowsHistL0]:
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

    de_apid = 1481

    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/P_GLX_TMSCDE.xml"
    )

    hist_packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    histparser = parser.PacketParser(hist_packet_definition, de_apid)

    histdata = []
    dedata = []

    with open(packet_file, "rb") as binary_data:
        try:
            hist_packets = histparser.generator(
                binary_data,
                parse_bad_pkts=False,
                buffer_read_size_bytes=5790778,
            )

            for packet in hist_packets:
                # Do something with the packet data
                # if packet.header["PKT_APID"].derived_value == hist_apid:
                #     hist_l0 = GlowsHistL0(packet)
                #     histdata.append(hist_l0)

                if packet.header["PKT_APID"].derived_value == de_apid:
                    de_l0 = GlowsDeL0(packet)
                    dedata.append(de_l0)

        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        return histdata, dedata


if __name__ == "__main__":
    histograms, direct_events = decom_packets(
        "../tests/imap_l0_sci_glows_20230920_v00.pcts"
    )
    print(direct_events[0])
