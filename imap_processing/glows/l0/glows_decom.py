from pathlib import Path

from bitstring import ReadError
from space_packet_parser import parser, xtcedef
from space_packet_parser.parser import ParsedDataItem

from imap_processing import imap_module_directory


def convert_histogram_data(raw_hist_data: int):
    # Convert the histogram data from a large raw string into a list of 8 bit values
    bin_hist_data = bin(raw_hist_data)[2:]
    histograms = []
    for i in range(8, len(bin_hist_data) + 2, 8):
        # print(bin_hist_data[i-8:i])
        histograms.append(int(bin_hist_data[i - 8 : i], 2))

    if len(histograms) != 3599:
        raise ValueError(
            f"Histogram packet is lacking bins. Expected a count of 3599, "
            f"actually recieved {len(histograms)}"
        )

    return histograms


def decom_packets(packet_file: str) -> list[dict[str, int]]:
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

    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/glows/packet_definitions/P_GLX_TMSCHIST.xml"
    )

    hist_packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    histparser = parser.PacketParser(hist_packet_definition, hist_apid)

    # Expected data keys from historgram packets
    histdata_keys = [
        "SHCOARSE",
        "STARTID",
        "ENDID",
        "FLAGS",
        "SWVER",
        "SEC",
        "SUBSEC",
        "OFFSETSEC",
        "OFFSETSUBSEC",
        "GLXSEC",
        "GLXSUBSEC",
        "GLXOFFSEC",
        "GLXOFFSUBSEC",
        "SPINS",
        "NBINS",
        "TEMPAVG",
        "TEMPVAR",
        "HVAVG",
        "HVVAR",
        "SPAVG",
        "SPVAR",
        "ELAVG",
        "ELVAR",
        "EVENTS",
        "HISTOGRAM_DATA",
    ]
    histdata = []

    with open(packet_file, "rb") as binary_data:
        try:
            hist_packets = histparser.generator(
                binary_data,
                parse_bad_pkts=False,
                buffer_read_size_bytes=5790778,
            )

            for packet in hist_packets:
                # Do something with the packet data
                if packet.header["PKT_APID"].derived_value == hist_apid:
                    histdata_dict = {}
                    for key in histdata_keys:
                        histdata_dict[key] = packet.data[key].derived_value
                    histdata_dict["HISTOGRAM_DATA"] = convert_histogram_data(
                        histdata_dict["HISTOGRAM_DATA"]
                    )
                    histdata.append(histdata_dict)

                # if packet.header["PKT_APID"].derived_value == de_apid:
                #     print(f"Decommed DE packet: {packet.header}")
        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        print(histdata[0]["HISTOGRAM_DATA"][:10])
        return histdata


def process_packets(packet_list: list[ParsedDataItem]):
    for packet in packet_list:
        print(packet.derived_value)


if __name__ == "__main__":
    histograms = decom_packets("../tests/imap_l0_sci_glows_20230920_v00.pcts")
