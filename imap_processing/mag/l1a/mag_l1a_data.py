"""Data classes for storing and processing MAG Level 1A data."""

from __future__ import annotations

import logging
from dataclasses import InitVar, dataclass, field
from math import floor

import numpy as np

from imap_processing.cdf.utils import J2000_EPOCH, met_to_j2000ns

MAX_FINE_TIME = 65535  # maximum 16 bit unsigned int

logger = logging.getLogger(__name__)


@dataclass
class TimeTuple:
    """
    Class for storing fine time/coarse time for MAG data.

    Course time is mission SCLK in seconds. Fine time is 16bit unsigned sub-second
    counter.

    Attributes
    ----------
    coarse_time : int
        Coarse time in seconds.
    fine_time : int
        Subsecond.

    Methods
    -------
    to_seconds()
    """

    coarse_time: int
    fine_time: int

    def __add__(self, seconds: float) -> TimeTuple:
        """
        Add a number of seconds to the time tuple.

        Parameters
        ----------
        seconds : float
            Number of seconds to add.

        Returns
        -------
        time : TimeTuple
            New time tuple with the current time tuple + seconds.
        """
        # Add whole seconds to coarse time
        coarse = self.coarse_time + floor(seconds)
        # fine time is 1/65535th of a second
        fine = self.fine_time + round((seconds % 1) * MAX_FINE_TIME)

        # If fine is larger than the max, move the excess into coarse time.
        if fine > MAX_FINE_TIME:
            coarse = coarse + floor(fine / MAX_FINE_TIME)
            fine = fine % MAX_FINE_TIME

        return TimeTuple(coarse, fine)

    def to_seconds(self) -> float:
        """
        Convert time tuple into seconds (float).

        Returns
        -------
        seconds : float
            Time in seconds.
        """
        return self.coarse_time + self.fine_time / MAX_FINE_TIME


@dataclass
class MagL1aPacketProperties:
    """
    Data class with Mag L1A per-packet data.

    This contains per-packet variations in L1a data that is passed into CDF
    files. Since each L1a file contains multiple packets, the variables in this
    class vary by time in the end CDF file.

    seconds_per_packet, and total_vectors are calculated from pus_ssubtype and
    vecsec values, which are only passed in to the init method and cannot be
    accessed from an instance as they are InitVars.

    To use the class, pass in pus_ssubtype and either PRI_VECSEC or SEC_VECSEC,
    then you can access seconds_per_packet and total_vectors.

    Attributes
    ----------
    shcoarse : int
        Mission elapsed time for the packet
    start_time : TimeTuple
        Start time of the packet
    vectors_per_second : int
        Number of vectors per second
    pus_ssubtype : int
        PUS Service Subtype - used to calculate seconds_per_packet. This is an InitVar,
        meaning it is only used when creating the class and cannot be accessed from an
        instance of the class - instead seconds_per_packet should be used.
    src_seq_ctr : int
        Sequence counter from the ccsds header
    compression : int
        Science Data Compression Flag from level 0
    mago_is_primary : int
        1 if mago is designated the primary sensor, otherwise 0
    seconds_per_packet : int
        Number of seconds of data in this packet - calculated as pus_ssubtype + 1
    total_vectors : int
        Total number of vectors in this packet - calculated as
        seconds_per_packet * vecsec
    """

    shcoarse: int
    start_time: TimeTuple
    vectors_per_second: int
    pus_ssubtype: InitVar[int]
    src_seq_ctr: int  # From ccsds header
    compression: int
    mago_is_primary: int
    seconds_per_packet: int = field(init=False)
    total_vectors: int = field(init=False)

    def __post_init__(self, pus_ssubtype: int) -> None:
        """
        Calculate seconds_per_packet and total_vectors.

        Parameters
        ----------
        pus_ssubtype : int
            PUS Service Subtype, used to determine the seconds of data in the packet.
        """
        # seconds of data in this packet is the SUBTYPE plus 1
        self.seconds_per_packet = pus_ssubtype + 1

        # VECSEC is already decoded in mag_l0
        self.total_vectors = self.seconds_per_packet * self.vectors_per_second


@dataclass
class MagL1a:
    """
    Data class for MAG Level 1A data.

    One MAG L1A object corresponds to part of one MAG L0 packet, which corresponds to
    one packet of data from the MAG instrument. Each L0 packet consists of data from
    two sensors, MAGO (outboard) and MAGI (inboard). One of these sensors is designated
    as the primary sensor (first part of data stream), and one as the secondary.

    We expect the primary sensor to be MAGO, and the secondary to be MAGI, but this is
    not guaranteed. Each MagL1A object contains data from one sensor. The
    primary/secondary construct is only used to sort the vectors into MAGo and MAGi
    data, and therefore is not used at higher levels.

    Attributes
    ----------
    is_mago : bool
        True if the data is from MagO, False if data is from MagI
    is_active : int
        1 if the sensor is active, 0 if not
    shcoarse : int
        Mission elapsed time for the first packet, the start time for the whole day
    vectors : numpy.ndarray
        List of magnetic vector samples, starting at start_time. [x, y, z, range, time],
        where time is numpy.datetime64[ns]
    starting_packet : InitVar[MagL1aPacketProperties]
        The packet properties for the first packet in the day. As an InitVar, this
        cannot be accessed from an instance of the class. Instead, packet_definitions
        should be used.
    packet_definitions : dict[numpy.datetime64, MagL1aPacketProperties]
        Dictionary of packet properties for each packet in the day. The key is the start
        time of the packet, and the value is a dataclass of packet properties.
    most_recent_sequence : int
        Sequence number of the most recent packet added to the object
    missing_sequences : list[int]
        List of missing sequence numbers in the day
    start_time : numpy.datetime64
        Start time of the day, in ns since J2000 epoch

    Methods
    -------
    append_vectors()
    calculate_vector_time()
    process_vector_data()
    """

    is_mago: bool
    is_active: int
    shcoarse: int
    vectors: np.array
    starting_packet: InitVar[MagL1aPacketProperties]
    packet_definitions: dict[np.datetime64, MagL1aPacketProperties] = field(init=False)
    most_recent_sequence: int = field(init=False)
    missing_sequences: list[int] = field(default_factory=list)
    start_time: np.datetime64 = field(init=False)

    def __post_init__(self, starting_packet: MagL1aPacketProperties) -> None:
        """
        Initialize the packet_definition dictionary and most_recent_sequence.

        Parameters
        ----------
        starting_packet : MagL1aPacketProperties
            The packet properties for the first packet in the day, including start time.
        """
        # TODO should this be from starting_packet
        self.start_time = (J2000_EPOCH + met_to_j2000ns(self.shcoarse)).astype(
            "datetime64[D]"
        )
        self.packet_definitions = {self.start_time: starting_packet}
        # most_recent_sequence is the sequence number of the packet used to initialize
        # the object
        self.most_recent_sequence = starting_packet.src_seq_ctr

    def append_vectors(
            self, additional_vectors: np.array,
            packet_properties: MagL1aPacketProperties
    ) -> None:
        """
        Append additional vectors to the current vectors array.

        Parameters
        ----------
        additional_vectors : numpy.array
            New vectors to append.
        packet_properties : MagL1aPacketProperties
            Additional vector definition to add to the l0_packets dictionary.
        """
        vector_sequence = packet_properties.src_seq_ctr

        self.vectors = np.concatenate([self.vectors, additional_vectors])
        self.packet_definitions[self.start_time] = packet_properties

        print(self.vectors.shape)

        # Every additional packet should be the next one in the sequence, if not, add
        # the missing sequence(s) to the gap data
        if not self.most_recent_sequence + 1 == vector_sequence:
            self.missing_sequences += list(
                range(self.most_recent_sequence + 1, vector_sequence)
            )
        self.most_recent_sequence = vector_sequence

    @staticmethod
    def calculate_vector_time(
            vectors: np.ndarray, vectors_per_sec: int, start_time: TimeTuple
    ) -> np.array:
        """
        Add timestamps to the vector list, turning the shape from (n, 4) to (n, 5).

        The first vector starts at start_time, then each subsequent vector time is
        computed by adding 1/vectors_per_second to the previous vector's time.

        Parameters
        ----------
        vectors : numpy.array
            List of magnetic vector samples, starting at start_time. Shape of (n, 4).
        vectors_per_sec : int
            Number of vectors per second.
        start_time : TimeTuple
            Start time of the vectors, the timestamp of the first vector.

        Returns
        -------
        vector_objects : numpy.ndarray
            Vectors with timestamps added in seconds, calculated from
            cdf.utils.met_to_j2000ns.
        """
        timedelta = np.timedelta64(int(1 / vectors_per_sec * 1e9), "ns")
        # TODO: validate that start_time from SHCOARSE is precise enough
        start_time_ns = met_to_j2000ns(start_time.to_seconds())

        # Calculate time skips for each vector in ns
        times = np.reshape(
            np.arange(
                start_time_ns,
                start_time_ns + timedelta * vectors.shape[0],
                timedelta,
                dtype=np.int64,
                like=vectors,
            ),
            (vectors.shape[0], -1),
        )
        vector_objects = np.concatenate([vectors, times], axis=1, dtype=np.int64)
        return vector_objects

    @staticmethod
    def process_vector_data(
            vector_data: np.ndarray, primary_count: int, secondary_count: int,
            compression: int
    ) -> tuple[np.ndarray, np.ndarray]:

        if compression:
            return MagL1a.process_compressed_vectors(vector_data, primary_count,
                                                     secondary_count)

        return MagL1a.process_uncompressed_vectors(vector_data, primary_count,
                                                   secondary_count)

    @staticmethod
    def process_uncompressed_vectors(
            vector_data: np.ndarray, primary_count: int, secondary_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given raw uncompressed packet data, process into Vectors.

        Vectors are grouped into primary sensor and secondary sensor, and returned as a
        tuple (primary sensor vectors, secondary sensor vectors).

        Written by MAG instrument team.

        Parameters
        ----------
        vector_data : numpy.ndarray
            Raw vector data, in bytes. Contains both primary and secondary vector data
            (first primary, then secondary).
        primary_count : int
            Count of the number of primary vectors.
        secondary_count : int
            Count of the number of secondary vectors.

        Returns
        -------
        (primary, secondary): (numpy.ndarray, numpy.ndarray)
            Two arrays, each containing tuples of (x, y, z, sample_range) for each
            vector sample.
        """

        # TODO: error handling
        def to_signed16(n: int) -> int:
            """
            Convert an integer to a signed 16-bit integer.

            Parameters
            ----------
            n : int
                The integer to be converted.

            Returns
            -------
            int
                Converted integer.
            """
            n = n & 0xFFFF
            return n | (-(n & 0x8000))

        pos = 0
        primary_vectors = []
        secondary_vectors = []

        # Since the vectors are stored as 50 bit chunks but accessed via hex (4 bit
        # chunks) there is some shifting required for processing the bytes.
        # However, from a bit processing perspective, the first 48 bits of each 50 bit
        # chunk corresponds to 3 16 bit signed integers. The last 2 bits are the sensor
        # range.

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
        return primary_vectors, secondary_vectors

        return (
            np.array(primary_vectors, dtype=np.int32),
            np.array(secondary_vectors, dtype=np.int32),
        )

    @staticmethod
    def process_compressed_vectors(vector_data: np.ndarray, primary_count: int,
                                   secondary_count: int) -> tuple[
        np.ndarray, np.ndarray]:
        """
        Given raw compressed packet data, process into Vectors.

        Parameters
        ----------
        vector_data
        primary_count
        secondary_count

        Returns
        -------

        """
        # TODO
        # - decode first vector to get starting point
        # - for each vector after that - divide into 11 separated chunks
        # - decode each chunk with fib + zigzag
        # - Also check if > 60

        # first vector is uncompressed, takes up compression_width bits

        bit_array = np.unpackbits(vector_data)
        print(bit_array)

        # Pad the first 6 bits with 0s to get the compression width and convert to int
        compression_width = np.packbits(np.array([0, 0] + list(bit_array[:6])))[0]
        print(compression_width)
        has_range_data_section = bit_array[6]

        # The full vector includes 3 values of compression_width bits, plus 2 bits for
        # the range
        vector_length = compression_width*3 + 2

        # index 7 is a spare
        first_vector = bit_array[8:8+vector_length]
        print(len(first_vector))

        return np.array([]), np.array([])

    @staticmethod
    def unpack_one_vector(vector_data: np.ndarray, width: int, has_range: int):
        """
        Unpack a single vector from the vector data.

        Input should be a numpy array of bits, eg [0, 0, 0, 1], of the length width*3,
        or width*3 + 2 if has_range is True.

        Parameters
        ----------
        vector_data: numpy.ndarray
        width: int
        has_range: int

        Returns
        -------
        numpy.ndarray
        """
        if np.any(vector_data > 1):
            raise ValueError("unpack_one_vector method is expecting an array of bits as"
                             "input.")

        if width % 8 != 0:
            raise ValueError("Width of the vector data should be a multiple of 8.")

        # TODO: could width ever be a non-multiple of 8? if yes pad with 0s at the
        #  beginning
        # take slices of the input data and pack from an array of bits to an array of
        # uint8 bytes
        x = np.packbits(vector_data[:width])
        y = np.packbits(vector_data[width:2*width])
        z = np.packbits(vector_data[2*width:3*width])

        range_string = "".join([str(i) for i in vector_data[-2:]])

        rng = int(range_string, 2) if has_range\
            else 0

        out = [MagL1a.twos_complement(x, width),
                         MagL1a.twos_complement(y, width),
                         MagL1a.twos_complement(z, width),
                         rng]
        # Convert to signed integers using twos complement
        return np.array([MagL1a.twos_complement(x, width),
                         MagL1a.twos_complement(y, width),
                         MagL1a.twos_complement(z, width),
                         rng], dtype=np.int32)

    @staticmethod
    def twos_complement(value: np.ndarray, bits: int) -> np.int32:
        """Compute the two's complement of an integer

        This function will return the two's complement of a given bytearray value.
        The input value should be a bytearray or a numpy array of uint8 values.

        If the integer with respect to the number of bits does not have a sign bit
        set (first bit is 0), then the input value is returned without modification.

        Parameters
        ----------
        value : numpy.ndarray
            An array of bytes representing an integer. In numpy, this should be an
            array of uint8 values.
        bits : int
            Number of bits to use for the 2's complement.

        Returns
        -------
        numpy.int32
            two's complement of the input value, as a signed int.
        """
        integer_value = int.from_bytes(value, "big")

        if (integer_value & (1 << (bits - 1))) != 0:
            output_value = integer_value - (1 << bits)
        else:
            output_value = integer_value
        return np.int32(output_value)
