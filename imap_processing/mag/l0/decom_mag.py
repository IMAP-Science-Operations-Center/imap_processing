import logging
from pathlib import Path
import xarray as xr
import numpy as np

from bitstring import ReadError
from space_packet_parser import parser, xtcedef

from imap_processing import imap_module_directory
from imap_processing.ccsds.ccsds_data import CcsdsData
from imap_processing.mag.l0.mag_l0_data import MagL0, Mode
from imap_processing.cdf.global_attrs import ConstantCoordinates, ScienceAttrs
from imap_processing.mag import mag_cdf_attrs

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


# TODO: write the output of this into a file
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
                if apid in (Mode.BURST, Mode.NORM):
                    values = [
                        item.derived_value
                        if item.derived_value is not None
                        else item.raw_value
                        for item in packet.data.values()
                    ]
                    data_list.append(MagL0(CcsdsData(packet.header), *values))
        except ReadError as e:
            logger.warning(
                f"Found error: {e}\n This may mean reaching the end of an "
                f"incomplete packet."
            )

        return data_list


def export_to_xarray(l1a_data: list[MagL0]):
    """ Mag outputs "RAW" CDF files just after decomming. These are the same as the L1A
     CDF data files, but with raw binary data for the vectors instead of a list of
     vector values. """

    # TODO split by mago and magi using primary sensor
    # TODO split by norm and burst
    norm_data = {"SHCOARSE": [],
                 "raw_vectors": []}
    burst_data = norm_data.copy()

    for datapoint in l1a_data:
        if datapoint.ccsds_header.PKT_APID == Mode.NORM:
            norm_data["SHCOARSE"].append(datapoint.SHCOARSE)
            norm_data["raw_vectors"].append(datapoint.VECTORS)

        if datapoint.ccsds_header.PKT_APID == Mode.BURST:
            burst_data["SHCOARSE"].append(datapoint.SHCOARSE)
            burst_data["raw_vectors"].append(datapoint.VECTORS)

    epoch_time = xr.DataArray(
        norm_data["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH
    )

    # Used in L1A vectors
    direction = xr.DataArray(
        np.arange(194),
        name="Direction",
        dims=["Direction"],
        attrs=mag_cdf_attrs.direction_attrs.output()
    )

    # print(direction)
    print(mag_cdf_attrs.direction_attrs.output())
    norm_epoch_time = xr.DataArray(
        norm_data["SHCOARSE"],
        name="Epoch",
        dims=["Epoch"],
        attrs=ConstantCoordinates.EPOCH
    )

    # TODO: raw vectors units
    norm_raw_vectors = xr.DataArray(
        norm_data["raw_vectors"],
        name="Raw Vectors",
        dims=["Epoch", "Direction"],
        attrs=mag_cdf_attrs.mag_vector_attrs.output()
    )

    # TODO add norm to attrs somehow
    norm_dataset = xr.Dataset(
        coords={
            "Epoch": norm_epoch_time,
            "Direction": direction
        },
        attrs=mag_cdf_attrs.mag_l1a_attrs.output()
    )

    norm_dataset["RAW-VECTORS"] = norm_raw_vectors
    # vectors
    # flags

    return norm_dataset