"""Methods for decomming packets, processing to level 1A, and writing CDFs for MAG."""

import logging

import numpy as np

from imap_processing.cdf.utils import write_cdf
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
    mag_l0 = decom_mag.decom_packets(packet_filepath)

    mag_norm_raw, mag_burst_raw = decom_mag.export_to_xarray(mag_l0)

    if mag_norm_raw is not None:
        file = write_cdf(mag_norm_raw)
        logger.info(f"Created CDF file at {file}")

    if mag_burst_raw is not None:
        file = write_cdf(mag_burst_raw)
        logger.info(f"Created CDF file at {file}")


def process_packets(mag_l0_list: list[MagL0]):
    """
    Given a list of MagL0 packets, process them into MagO and MagI L1A data classes.

    Parameters
    ----------
    mag_l0_list : list[MagL0]
        List of Mag L0 packets to process

    Returns
    -------
    (list[MagL1a], list[MagL1a])
        Tuple containing MagI and MagO L1A data classes

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

        # now we know the number of secs of data in the packet, and the data rates of
        # each sensor, we can calculate how much data is in this packet and where the
        # byte boundaries are.

        # VECSEC is already decoded in mag_l0
        total_primary_vectors = seconds_per_packet * mag_l0.PRI_VECSEC
        total_secondary_vectors = seconds_per_packet * mag_l0.SEC_VECSEC

        primary_vectors, secondary_vectors = process_vector_data(
            mag_l0.VECTORS, total_primary_vectors, total_secondary_vectors
        )

        # Primary sensor is MAGO
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
        if mag_l0.PRI_SENS == 1:
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

        magi.append(magi_l1a)
        mago.append(mago_l1a)

    return magi, mago


def process_vector_data(
    vector_data: np.ndarray, primary_count: int, secondary_count: int
) -> (list[tuple], list[tuple]):
    """
    Given raw packet data, process into Vectors.

    Vectors are grouped into primary sensor and secondary sensor, and returned as a
    tuple (primary sensor vectors, secondary sensor vectors)

    Written by MAG instrument team

    Parameters
    ----------
    vector_data : np.ndarray
        Raw vector data, in bytes. Contains both primary and secondary vector data
        (first primary, then secondary)
    primary_count : int
        Count of the number of primary vectors
    secondary_count : int
        Count of the number of secondary vectors

    Returns
    -------
    (primary, secondary)
        Two arrays, each containing tuples of (x, y, z, sample_range) for each vector
        sample.
    """

    # TODO: error handling
    def to_signed16(n):
        n = n & 0xFFFF
        return n | (-(n & 0x8000))

    pos = 0
    primary_vectors = []
    secondary_vectors = []

    for i in range(primary_count + secondary_count):  # 0..63 say
        x, y, z, rng = 0, 0, 0, 0
        if i % 4 == 0:  # start at bit 0, take 8 bits + 8bits
            # pos = 0, 25, 50...
            x = (
                ((vector_data[pos + 0] & 0xFF) << 8)
                | ((vector_data[pos + 1] & 0xFF) << 0)
            ) & 0xFFFF
            y = (
                ((vector_data[pos + 2] & 0xFF) << 8)
                | ((vector_data[pos + 3] & 0xFF) << 0)
            ) & 0xFFFF
            z = (
                ((vector_data[pos + 4] & 0xFF) << 8)
                | ((vector_data[pos + 5] & 0xFF) << 0)
            ) & 0xFFFF
            rng = (vector_data[pos + 6] >> 6) & 0x3
            pos += 6
        elif i % 4 == 1:  # start at bit 2, take 6 bits, 8 bit, 2 bits per vector
            # pos = 6, 31...
            x = (
                ((vector_data[pos + 0] & 0x3F) << 10)
                | ((vector_data[pos + 1] & 0xFF) << 2)
                | ((vector_data[pos + 2] >> 6) & 0x03)
            ) & 0xFFFF
            y = (
                ((vector_data[pos + 2] & 0x3F) << 10)
                | ((vector_data[pos + 3] & 0xFF) << 2)
                | ((vector_data[pos + 4] >> 6) & 0x03)
            ) & 0xFFFF
            z = (
                ((vector_data[pos + 4] & 0x3F) << 10)
                | ((vector_data[pos + 5] & 0xFF) << 2)
                | ((vector_data[pos + 6] >> 6) & 0x03)
            ) & 0xFFFF
            rng = (vector_data[pos + 6] >> 4) & 0x3
            pos += 6
        elif i % 4 == 2:  # start at bit 4, take 4 bits, 8 bits, 4 bits per vector
            # pos = 12, 37...
            x = (
                ((vector_data[pos + 0] & 0x0F) << 12)
                | ((vector_data[pos + 1] & 0xFF) << 4)
                | ((vector_data[pos + 2] >> 4) & 0x0F)
            ) & 0xFFFF
            y = (
                ((vector_data[pos + 2] & 0x0F) << 12)
                | ((vector_data[pos + 3] & 0xFF) << 4)
                | ((vector_data[pos + 4] >> 4) & 0x0F)
            ) & 0xFFFF
            z = (
                ((vector_data[pos + 4] & 0x0F) << 12)
                | ((vector_data[pos + 5] & 0xFF) << 4)
                | ((vector_data[pos + 6] >> 4) & 0x0F)
            ) & 0xFFFF
            rng = (vector_data[pos + 6] >> 2) & 0x3
            pos += 6
        elif i % 4 == 3:  # start at bit 6, take 2 bits, 8 bits, 6 bits per vector
            # pos = 18, 43...
            x = (
                ((vector_data[pos + 0] & 0x03) << 14)
                | ((vector_data[pos + 1] & 0xFF) << 6)
                | ((vector_data[pos + 2] >> 2) & 0x3F)
            ) & 0xFFFF
            y = (
                ((vector_data[pos + 2] & 0x03) << 14)
                | ((vector_data[pos + 3] & 0xFF) << 6)
                | ((vector_data[pos + 4] >> 2) & 0x3F)
            ) & 0xFFFF
            z = (
                ((vector_data[pos + 4] & 0x03) << 14)
                | ((vector_data[pos + 5] & 0xFF) << 6)
                | ((vector_data[pos + 6] >> 2) & 0x3F)
            ) & 0xFFFF
            rng = (vector_data[pos + 6] >> 0) & 0x3
            pos += 7

        vector = (to_signed16(x), to_signed16(y), to_signed16(z), rng)
        if i < primary_count:
            primary_vectors.append(vector)
        else:
            secondary_vectors.append(vector)

    return (primary_vectors, secondary_vectors)
