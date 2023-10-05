from pathlib import Path

from bitstring import ReadError
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.mag.l0.mag_l0_data import MagApid, MagL0


# TODO: Keep burst and norm packets separate here?
def decom_packets(packet_file_path: str) -> list[MagL0]:
    """Decom MAG data packets using MAG packet definition.

    Parameters
    ----------
    packet_file_path : str
        Path to data packet path with filename.

    Returns
    -------
    data : list[MagL0]
        A list of MAG L0 data classes, including both burst and normal packets. (the
        packet type is defined in each instance of L0.)
    """
    # Define paths
    xtce_document = Path(
        f"{imap_module_directory}/mag/packet_definitions/MAG_SCI_COMBINED.xml"
    )

    packet_definition = xtcedef.XtcePacketDefinition(xtce_document)
    mag_parser = parser.PacketParser(packet_definition)

    data_list = []

    with open(packet_file_path, "rb") as binary_data:
        try:
            mag_packets = mag_parser.generator(
                binary_data,
                buffer_read_size_bytes=5790778,  # Todo: what size?
            )

            for packet in mag_packets:
                apid = packet.header["PKT_APID"].derived_value
                for enum_val in MagApid:
                    if apid == enum_val.value:
                        data_list.append(MagL0(packet, enum_val))

        except ReadError as e:
            print(e)
            print("This may mean reaching the end of an incomplete packet.")

        return data_list
