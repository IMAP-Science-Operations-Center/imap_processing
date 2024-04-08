"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

import logging

from imap_processing.cdf.utils import write_cdf
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0 import decom_mag
from imap_processing.mag.l0.mag_l0_data import MagL0
from imap_processing.mag.l1a.mag_l1a_data import MagL1a, TimeTuple

logger = logging.getLogger(__name__)


def mag_l1a(packet_filepath):
    """
    Process MAG L0 data into L1A CDF files at cdf_filepath.

    Parameters
    ----------
    packet_filepath :
        Packet files for processing
    """
    packets = decom_mag.decom_packets(packet_filepath)

    norm_data = packets["norm"]
    burst_data = packets["burst"]

    if norm_data is not None:
        mag_norm_raw = decom_mag.generate_dataset(
            norm_data, mag_cdf_attrs.mag_l1a_norm_raw_attrs.output()
        )
        file = write_cdf(mag_norm_raw)
        logger.info(f"Created RAW CDF file at {file}")

    if burst_data is not None:
        mag_burst_raw = decom_mag.generate_dataset(
            burst_data, mag_cdf_attrs.mag_l1a_burst_raw_attrs.output()
        )
        file = write_cdf(mag_burst_raw)
        logger.info(f"Created RAW CDF file at {file}")


def process_packets(mag_l0_list: list[MagL0]) -> dict[str, list[MagL1a]]:
    """
    Given a list of MagL0 packets, process them into MagO and MagI L1A data classes.

    Parameters
    ----------
    mag_l0_list : list[MagL0]
        List of Mag L0 packets to process

    Returns
    -------
    packet_dict: dict[str, list[MagL1a]]
        Dictionary containing two keys: "mago" which points to a list of mago MagL1A
        objects, and "magi" which points to a list of magi MagL1A objects.

    """
    magi = []
    mago = []

    for mag_l0 in mag_l0_list:
        if mag_l0.COMPRESSION:
            raise NotImplementedError("Unable to process compressed data")

        primary_start_time = TimeTuple(mag_l0.PRI_COARSETM, mag_l0.PRI_FNTM)
        secondary_start_time = TimeTuple(mag_l0.SEC_COARSETM, mag_l0.SEC_FNTM)

        # seconds of data in this packet is the SUBTYPE plus 1
        seconds_per_packet = mag_l0.PUS_SSUBTYPE + 1

        # now we know the number of seconds of data in the packet, and the data rates of
        # each sensor, we can calculate how much data is in this packet and where the
        # byte boundaries are.

        # VECSEC is already decoded in mag_l0
        total_primary_vectors = seconds_per_packet * mag_l0.PRI_VECSEC
        total_secondary_vectors = seconds_per_packet * mag_l0.SEC_VECSEC

        primary_vectors, secondary_vectors = MagL1a.process_vector_data(
            mag_l0.VECTORS, total_primary_vectors, total_secondary_vectors
        )

        # Primary sensor is MAGO (most common expected case)
        if mag_l0.PRI_SENS == 0:
            mago_l1a = MagL1a(
                True,
                bool(mag_l0.MAGO_ACT),
                primary_start_time,
                mag_l0.PRI_VECSEC,
                total_primary_vectors,
                seconds_per_packet,
                mag_l0.SHCOARSE,
                primary_vectors,
            )

            magi_l1a = MagL1a(
                False,
                bool(mag_l0.MAGI_ACT),
                secondary_start_time,
                mag_l0.SEC_VECSEC,
                total_secondary_vectors,
                seconds_per_packet,
                mag_l0.SHCOARSE,
                secondary_vectors,
            )
        # Primary sensor is MAGI
        else:
            magi_l1a = MagL1a(
                False,
                bool(mag_l0.MAGI_ACT),
                primary_start_time,
                mag_l0.PRI_VECSEC,
                total_primary_vectors,
                seconds_per_packet,
                mag_l0.SHCOARSE,
                primary_vectors,
            )

            mago_l1a = MagL1a(
                True,
                bool(mag_l0.MAGO_ACT),
                secondary_start_time,
                mag_l0.SEC_VECSEC,
                total_secondary_vectors,
                seconds_per_packet,
                mag_l0.SHCOARSE,
                secondary_vectors,
            )

        mago.append(mago_l1a)
        magi.append(magi_l1a)

    return {"mago": mago, "magi": magi}
