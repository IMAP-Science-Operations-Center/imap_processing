"""Data classes for storing and processing MAG Level 1A data."""

from __future__ import annotations

import math
from dataclasses import InitVar, dataclass, field
from math import floor

import numpy as np
import numpy.typing as npt

from imap_processing.mag.constants import (
    AXIS_COUNT,
    FIBONACCI_SEQUENCE,
    MAX_COMPRESSED_VECTOR_BITS,
    MAX_FINE_TIME,
    RANGE_BIT_WIDTH,
)
from imap_processing.spice.time import met_to_ttj2000ns


@dataclass
class TimeTuple:
    """
    Class for storing fine time/coarse time for MAG data.

    Coarse time is MET in seconds. Fine time is 16bit unsigned sub-second
    counter.

    Attributes
    ----------
    coarse_time : int
        Coarse time in seconds (MET).
    fine_time : int
        Subsecond counter, equal to fine_time/max_int16 seconds.

    Methods
    -------
    to_seconds()
    to_j2000ns()
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
        return float(self.coarse_time + self.fine_time / MAX_FINE_TIME)

    def to_j2000ns(self) -> np.int64:
        """
        Convert time tuple into J2000ns.

        Returns
        -------
        j2000ns : numpy.int64
            Time in nanoseconds since J2000 epoch.
        """
        coarse_j2000ns = np.int64(met_to_ttj2000ns(self.coarse_time))
        fine_ns = np.int64(self.fine_time / MAX_FINE_TIME * 1e9)
        return coarse_j2000ns + fine_ns


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
    first_byte : int
        First byte of the vector data. Needed to compute compression_width.
    seconds_per_packet : int
        Number of seconds of data in this packet - calculated as pus_ssubtype + 1
    total_vectors : int
        Total number of vectors in this packet - calculated as
        seconds_per_packet * vecsec
    compression_width : int
        Value between 0-20 indicating the width of the compressed data. If the packet
        is uncompressed, this defaults to 0.
    """

    shcoarse: int
    start_time: TimeTuple
    vectors_per_second: int
    pus_ssubtype: InitVar[int]
    src_seq_ctr: int  # From ccsds header
    compression: int
    mago_is_primary: int
    first_byte: InitVar[int]
    seconds_per_packet: int = field(init=False)
    total_vectors: int = field(init=False)
    compression_width: int = field(init=False)

    def __post_init__(self, pus_ssubtype: int, first_byte: int) -> None:
        """
        Calculate seconds_per_packet, total_vectors, and compression_width.

        Parameters
        ----------
        pus_ssubtype : int
            PUS Service Subtype, used to determine the seconds of data in the packet.
        first_byte : int
            First byte from the vectors. Needed to compute compression_width, and only
            really needed if compression == 1. The header is the first byte of the
            vector data. The compression width is the first 6 bits of that byte.
        """
        # seconds of data in this packet is the SUBTYPE plus 1
        self.seconds_per_packet = pus_ssubtype + 1

        # VECSEC is already decoded in mag_l0
        self.total_vectors = self.seconds_per_packet * self.vectors_per_second

        if self.compression == 1:
            header_bin = bin(first_byte)
            # first two values in the string are "0b"
            # going from the end because if the byte starts with zeros, they aren't
            # included by bin().
            self.compression_width = int(header_bin[2:-2], 2)
        else:
            self.compression_width = 0


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
    start_time : numpy.int64
        Start time of the day, in ns since J2000 epoch
    compression_flags : np.ndarray
        Array of flags to indication compression and width for all timestamps in the
        L1A file. Shaped like (n, 2) where n is the number of vectors. First value
        is a boolean for compressed/uncompressed, second vector is a number between 0-20
        if the data is compressed, which is the width in bits of the compressed data.

    Methods
    -------
    append_vectors()
    calculate_vector_time()
    convert_diffs_to_vectors()
    process_vector_data()
    process_uncompressed_vectors()
    process_compressed_vectors()
    process_range_data_section()
    _process_vector_section()
    unpack_one_vector()
    decode_fib_zig_zag()
    twos_complement()
    update_compression_array()
    vectors_per_second_attribute()
    """

    is_mago: bool
    is_active: int
    shcoarse: int
    vectors: np.ndarray
    starting_packet: InitVar[MagL1aPacketProperties]
    packet_definitions: dict[np.int64, MagL1aPacketProperties] = field(init=False)
    most_recent_sequence: int = field(init=False)
    missing_sequences: list[int] = field(default_factory=list)
    start_time: np.int64 = field(init=False)
    compression_flags: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self, starting_packet: MagL1aPacketProperties) -> None:
        """
        Initialize the packet_definition dictionary and most_recent_sequence.

        Parameters
        ----------
        starting_packet : MagL1aPacketProperties
            The packet properties for the first packet in the day, including start time.
        """
        self.start_time = np.int64(met_to_ttj2000ns(starting_packet.shcoarse))
        self.packet_definitions = {self.start_time: starting_packet}
        # most_recent_sequence is the sequence number of the packet used to initialize
        # the object
        self.most_recent_sequence = starting_packet.src_seq_ctr
        self.update_compression_array(starting_packet, self.vectors.shape[0])

    def append_vectors(
        self, additional_vectors: np.ndarray, packet_properties: MagL1aPacketProperties
    ) -> None:
        """
        Append additional vectors to the current vectors array.

        Parameters
        ----------
        additional_vectors : numpy.ndarray
            New vectors to append.
        packet_properties : MagL1aPacketProperties
            Additional vector definition to add to the l0_packets dictionary.
        """
        vector_sequence = packet_properties.src_seq_ctr

        self.vectors = np.concatenate([self.vectors, additional_vectors])
        start_time = np.int64(met_to_ttj2000ns(packet_properties.shcoarse))
        self.packet_definitions[start_time] = packet_properties

        # Every additional packet should be the next one in the sequence, if not, add
        # the missing sequence(s) to the gap data
        if not self.most_recent_sequence + 1 == vector_sequence:
            self.missing_sequences += list(
                range(self.most_recent_sequence + 1, vector_sequence)
            )
        self.most_recent_sequence = vector_sequence
        self.update_compression_array(packet_properties, additional_vectors.shape[0])

    def update_compression_array(
        self, packet_properties: MagL1aPacketProperties, length: int
    ) -> None:
        """
        Create or update compression_flags with 1 flag per timestamp.

        If the compression for the packet is true, the second value in the array is the
        compression width. Otherwise, it is zero.

        Parameters
        ----------
        packet_properties : MagL1aPacketProperties
            Packet properties that define compression flag and compression width.
        length : int
            Length of new array to add or append to the compression_flags attribute.
            This is expected to be the length of the vector array.
        """
        new_flags = np.full(
            (length, 2),
            [packet_properties.compression, packet_properties.compression_width],
            dtype=np.int8,
        )
        if self.compression_flags is None:
            self.compression_flags = new_flags
        else:
            self.compression_flags = np.concatenate([self.compression_flags, new_flags])

    @staticmethod
    def calculate_vector_time(
        vectors: np.ndarray, vectors_per_sec: int, start_time: TimeTuple
    ) -> npt.NDArray:
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
        start_time_ns = start_time.to_j2000ns()
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
        vector_data: np.ndarray | bytes,
        primary_count: int,
        secondary_count: int,
        compression: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform raw vector data into Vectors.

        Vectors are grouped into primary sensor and secondary sensor, and returned as a
        tuple (primary sensor vectors, secondary sensor vectors).

        Parameters
        ----------
        vector_data : numpy.ndarray
            Raw vector data, in bytes. Contains both primary and secondary vector data.
            Can be either compressed or uncompressed.
        primary_count : int
            Count of the number of primary vectors.
        secondary_count : int
            Count of the number of secondary vectors.
        compression : int
            Flag indicating if the data is compressed (1) or uncompressed (0).

        Returns
        -------
        (primary, secondary): (numpy.ndarray, numpy.ndarray)
            Two arrays, each containing tuples of (x, y, z, sample_range) for each
            vector sample.
        """
        if compression:
            # If the vectors are compressed, we need them to be uint8 to convert to
            # bits.
            return MagL1a.process_compressed_vectors(
                vector_data.astype(np.uint8), primary_count, secondary_count
            )

        # If the vectors are uncompressed, we need them to be int32, as there are
        # bitshifting operations. Either way, the return type should be int32.
        return MagL1a.process_uncompressed_vectors(
            vector_data.astype(np.int32), primary_count, secondary_count
        )

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

        return (
            np.array(primary_vectors, dtype=np.int32),
            np.array(secondary_vectors, dtype=np.int32),
        )

    @staticmethod
    def process_compressed_vectors(
        vector_data: np.ndarray, primary_count: int, secondary_count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Given raw compressed packet data, process into Vectors.

        To do this, we need to decode the compressed data. The compressed data starts
        with an 8 bit header that defines the width of the uncompressed vectors and
        if there is a range data section. Then, the vector data follows, then the range
        data section if it exists.

        To decode, we start by decoding the first compression_width bits. This is an
        uncompressed primary vector with range. Then, we proceed through the compressed
        data, where each value is fibonacci and zig-zag encoded. This means each value
        ends in 2 sequential ones (11). We split the data along these numbers until
        we reach primary_count vectors.

        The secondary vectors are decoded the same way, starting directly after the last
        primary vector with an uncompressed secondary starting vector and then
        secondary_count compressed vectors.

        The compressed values are differences from the previous vector, so after
        decoding we accumulate the values starting from the first known vector. The
        range data is copied from the starting vector if range_data_section is not
        included.

        Then, if a range data section is included, we decode it and assign it to each
        vector. There are 2 * (primary_count + secondary_count) bits assigned for the
        range data section.

        If any compressed vectors are > 60 bits long (MAX_COMPRESSED_VECTOR_BITS), then
        we switch to uncompressed vectors for the rest of the processing.

        Parameters
        ----------
        vector_data : numpy.ndarray
            Raw vector data, in bytes. Contains both primary and secondary vector data.
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
        bit_array = np.unpackbits(vector_data)
        # The first 8 bits are a header - 6 bits to indicate the compression width,
        # 1 bit to indicate if there is a range data section, and 1 bit spare.
        compression_width = int("".join([str(i) for i in bit_array[:6]]), 2)
        has_range_data_section = bit_array[6] == 1

        # The full vector includes 3 values of compression_width bits, and excludes
        # range.
        uncompressed_vector_size = compression_width * AXIS_COUNT
        # plus 8 to get past the compression width and range data section
        first_vector_width = uncompressed_vector_size + 8 + RANGE_BIT_WIDTH
        first_vector = MagL1a.unpack_one_vector(
            bit_array[8:first_vector_width], compression_width, True
        )

        # The range data length has 2 bits per vector, minus 2 for the uncompressed
        # first vectors in the primary and secondary sensors.
        # Then, the range data length is padded to the nearest 8 bits.
        expected_range_data_length = (primary_count + secondary_count - 2) * 2
        end_padding = expected_range_data_length // 8 * 8 - expected_range_data_length

        end_vector = len(bit_array) - (
            (expected_range_data_length - end_padding) * has_range_data_section
        )

        # Cut off the first vector width and the range data section.
        vector_bits = bit_array[first_vector_width - 1 : end_vector]

        # Shift the bit array over one to the left, then sum them up. This is used to
        # find all the places where two 1s occur next to each other, because the sum
        # will be 2 for those indices.
        # For example: [0 0 1 0 1 1] + [1 0 0 1 0 1] = [1 0 1 1 1 2], so the last index
        # has 2 ones in a row.
        # The first bit is invalid, so we remove it at the end.
        sequential_ones = np.where(
            np.add(
                vector_bits,
                np.roll(vector_bits, 1),
            )[1:]
            == 2
        )[0]
        # The first bit is only needed for the np.roll step, so now we remove it.
        # we are left with compressed primary vectors, and all the secondary vectors.
        vector_bits = vector_bits[1:]

        # that unneeded first bit might give us a false first sequential ones value,
        # so if the first index has 2 ones, skip it.
        primary_boundaries = [
            (
                sequential_ones[0] + 1
                if sequential_ones[0] != 0
                else sequential_ones[1] + 1
            )
        ]
        secondary_boundaries = []
        vector_count = 1
        end_primary_vector = 0
        for seq_val in sequential_ones:
            if vector_count > primary_count + secondary_count:
                break
            # Add the end indices of each primary vector to primary_boundaries
            # If we have 3 ones in a row, we should skip that index

            if vector_count < primary_count and (seq_val - primary_boundaries[-1] > 0):
                primary_boundaries.append(seq_val + 1)

                # 3 boundaries equal one vector
                if len(primary_boundaries) % AXIS_COUNT == 0:
                    vector_count += 1
                    # If the vector length is >60 bits, we switch to uncompressed.
                    # So we skip past all the remaining seq_ones.
                    if (
                        (len(primary_boundaries) > 4)
                        and (
                            primary_boundaries[-1] - primary_boundaries[-4]
                            > MAX_COMPRESSED_VECTOR_BITS
                        )
                    ) or (
                        vector_count == 2
                        and primary_boundaries[-1] > MAX_COMPRESSED_VECTOR_BITS
                    ):
                        # Since we know how long each uncompressed vector is,
                        # we can determine the end of the primary vectors.
                        end_primary_vector = (
                            primary_boundaries[-1]
                            + (primary_count - vector_count) * uncompressed_vector_size
                        )
                        vector_count = primary_count

            # If the vector count is equal to the primary count, we are in the first
            # uncompressed secondary vector.
            if vector_count == primary_count:
                # We won't have assigned end_primary_vector unless we hit uncompressed
                # vectors in the primary path. If there are no uncompressed values,
                # we can use the end of primary_boundaries.
                end_primary_vector = (
                    primary_boundaries[-1]
                    if end_primary_vector == 0
                    else end_primary_vector
                )
                if seq_val > end_primary_vector + uncompressed_vector_size + 2:
                    # Split just after the uncompressed secondary vector
                    secondary_boundaries = [
                        end_primary_vector + uncompressed_vector_size + 2
                    ]
                    # We have found the first secondary vector
                    secondary_boundaries += [seq_val + 1]
                    vector_count += 1

            # If we're greater than primary_count, we are in the secondary vectors.
            # Like before, we skip indices with 3 ones.
            if vector_count > primary_count and seq_val - secondary_boundaries[-1] > 0:
                secondary_boundaries.append(seq_val + 1)
                # We have the start of the secondary vectors in
                # secondary_boundaries, so we need to subtract one to determine
                # the vector count. (in primary_boundaries we know we start at 0.)
                if (len(secondary_boundaries) - 1) % AXIS_COUNT == 0:
                    vector_count += 1
                    if (
                        secondary_boundaries[-1] - secondary_boundaries[-4]
                        > MAX_COMPRESSED_VECTOR_BITS
                    ):
                        # The rest of the secondary values are uncompressed.
                        vector_count = primary_count + secondary_count + 1

        # Split along the boundaries of the primary vectors. This gives us a list of
        # bit arrays, each corresponding to a primary value (1/3 of a vector).
        primary_split_bits = np.split(
            vector_bits,
            primary_boundaries,
        )[:-1]
        primary_vectors = MagL1a._process_vector_section(
            vector_bits,
            primary_split_bits,
            primary_boundaries[-1],
            first_vector,
            primary_count,
            uncompressed_vector_size,
            compression_width,
        )[0]

        # Secondary vector processing
        first_secondary_vector = MagL1a.unpack_one_vector(
            vector_bits[
                end_primary_vector : end_primary_vector + uncompressed_vector_size + 2
            ],
            compression_width,
            True,
        )

        # Split up the bit array, skipping past the primary vector and uncompressed
        # starting vector
        secondary_split_bits = np.split(
            vector_bits[: secondary_boundaries[-1]], secondary_boundaries[:-1]
        )[1:]
        secondary_process_vectors = MagL1a._process_vector_section(
            vector_bits,
            secondary_split_bits,
            secondary_boundaries[-1],
            first_secondary_vector,
            secondary_count,
            uncompressed_vector_size,
            compression_width,
        )
        secondary_vectors = secondary_process_vectors[0]

        # The range data length has 2 bits per vector, minus 2 for the uncompressed
        # first vectors in the primary and secondary sensors.
        # Then, the range data length is padded to the nearest 8 bits.
        end_vector = ((secondary_process_vectors[1] + first_vector_width) + 7) // 8 * 8

        # If there is a range data section, it describes all the data, compressed or
        # uncompressed.
        if has_range_data_section:
            range_bits = bit_array[end_vector:]
            primary_vectors = MagL1a.process_range_data_section(
                range_bits[: (primary_count - 1) * 2],
                primary_vectors,
            )
            secondary_vectors = MagL1a.process_range_data_section(
                range_bits[
                    (primary_count - 1) * 2 : (primary_count + secondary_count - 2) * 2
                ],
                secondary_vectors,
            )

        return primary_vectors, secondary_vectors

    @staticmethod
    def _process_vector_section(
        vector_bits: np.ndarray,
        split_bits: list,
        last_index: int,
        first_vector: np.ndarray,
        vector_count: int,
        uncompressed_vector_size: int,
        compression_width: int,
    ) -> tuple[np.ndarray, int]:
        """
        Generate a section of vector data, primary or secondary.

        Should only be used by process_compressed_vectors.

        Parameters
        ----------
        vector_bits : numpy.ndarray
            Numpy array of bits, representing the vector data. Does not include the
            first primary vector.
        split_bits : list
            An array of where to split vector bits, by passing in a list of indices.
        last_index : int
            The index of the last vector in the section (primary or secondary).
        first_vector : numpy.ndarray
            The first vector in the section, (x, y, z, range).
        vector_count : numpy.ndarray
            The number of vectors in the section (primary or secondary).
        uncompressed_vector_size : int
            The size of an uncompressed vector in bits.
        compression_width : int
            The width of the uncompressed values - uncompressed_vector_size/3.

        Returns
        -------
        (numpy.ndarray, int)
            A tuple consisting of: an array of processed vectors, and the index of the
            end of the last vector. In a fully compressed case, this will be the same
            as last_index.
        """
        compressed_count = math.ceil(len(split_bits) / AXIS_COUNT) + 1
        uncompressed_count = vector_count - compressed_count

        # will have either 0 or 8 extra bits
        # If we have more splits than vectors, we have included part of the range
        # data section. Drop those values and update end_vector.
        if len(split_bits) > (vector_count - 1) * AXIS_COUNT:
            split_bits = split_bits[: (vector_count - 1) * AXIS_COUNT]
            last_index -= 8

        end = last_index + uncompressed_vector_size * uncompressed_count

        vector_diffs = list(map(MagL1a.decode_fib_zig_zag, split_bits))
        # if we got more than the expected length, we may have gotten some of the range
        # data section.

        vectors = MagL1a.convert_diffs_to_vectors(
            first_vector, vector_diffs, vector_count
        )
        # If we are missing any vectors from primary_split_bits, we know we have
        # uncompressed vectors to process.
        if uncompressed_count:
            uncompressed_vectors = vector_bits[last_index : end + 1]

            for i in range(uncompressed_count):
                decoded_vector = MagL1a.unpack_one_vector(
                    uncompressed_vectors[
                        i * uncompressed_vector_size : (i + 1)
                        * uncompressed_vector_size
                    ],
                    compression_width,
                    False,
                )
                vectors[i + compressed_count] = decoded_vector
                vectors[i + compressed_count][3] = vectors[0][3]
            # If there is an extra byte, this is from the range data section.
            if len(vector_bits[last_index:]) - end == 8:
                end -= 8

        return (vectors, end)

    @staticmethod
    def process_range_data_section(
        range_data: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """
        Given a range data section and vectors, return an updated vector array.

        Each range value has 2 bits. range_data will have a length of n*2, where n is
        the number of vectors in vectors.

        Parameters
        ----------
        range_data : numpy.ndarray
            Array of range values, where each value is one bit. The range values have
            2 bits per vector, so range data should be 2 * len(vectors) - 1 in length.
        vectors : numpy.ndarray
            Array of vectors, where each vector is a tuple of (x, y, z, range).
            The range value will be overwritten by range_data, and x, y, z will remain
            the same.

        Returns
        -------
        numpy.ndarray
            Updated array of vectors, identical to vectors with the range values
            updated from range_data.
        """
        if len(range_data) != (len(vectors) - 1) * 2:
            raise ValueError(
                "Incorrect length for range_data, there should be two bits per vector, "
                "excluding the first."
            )
        updated_vectors: np.ndarray = np.copy(vectors)
        range_str = "".join([str(i) for i in range_data])
        for i in range(len(vectors) - 1):
            range_int = int(range_str[i * 2 : i * 2 + 2], 2)
            updated_vectors[i + 1][3] = range_int
        return updated_vectors

    @staticmethod
    def convert_diffs_to_vectors(
        first_vector: np.ndarray,
        vector_differences: list[int],
        vector_count: int,
    ) -> np.ndarray:
        """
        Given a list of differences and the first vector, return calculated vectors.

        This is calculated as follows:
            vector[i][0] = vector[i-1][0] + vector_differences[i][0]
            vector[i][1] = vector[i-1][1] + vector_differences[i][1]
            vector[i][2] = vector[i-1][2] + vector_differences[i][2]
            vector[i][3] = first_vector[3]

        The third element of the array is the range value, which we assume is the same
        as the first vector.

        Parameters
        ----------
        first_vector : numpy.ndarray
            A numpy array of 3 signed integers and a range value, representing the
            start vector.
        vector_differences : numpy.ndarray
            A numpy array of shape (expected_vector_count, 4) of signed integers,
            representing the differences between vectors.
        vector_count : int
            The expected number of vectors in the output.

        Returns
        -------
        numpy.ndarray
            A numpy array of shape (expected_vector_count, 4) of signed integers,
            representing the calculated vectors.
        """
        vectors: np.ndarray = np.empty((vector_count, 4), dtype=np.int32)
        vectors[0] = first_vector
        if len(vector_differences) % AXIS_COUNT != 0:
            raise ValueError(
                "Error! Computed compressed vector differences are not "
                "divisible by 3 - meaning some data is missing. "
                "Expected length: %s, actual length: "
                "%s",
                vector_count * AXIS_COUNT,
                len(vector_differences),
            )
        index = 0
        vector_index = 1
        for diff in vector_differences:
            vectors[vector_index][index] = vectors[vector_index - 1][index] + diff
            index += 1
            if index == 3:
                # Update range section to match that of the first vector
                vectors[vector_index][3] = vectors[0][3]
                index = 0
                vector_index += 1
        return vectors

    @staticmethod
    def unpack_one_vector(
        vector_data: np.ndarray, width: int, has_range: int
    ) -> np.ndarray:
        """
        Unpack a single vector from the vector data.

        Input should be a numpy array of bits, eg [0, 0, 0, 1], of the length width*3,
        or width*3 + 2 if has_range is True.

        Parameters
        ----------
        vector_data : numpy.ndarray
            Vector data for the vector to unpack. This is uncompressed data as a numpy
            array of bits (the output of np.unpackbits).
        width : int
            The width of each vector component in bits. This needs to be a multiple of
            8 (including only whole bytes).
        has_range : int
            1 if the vector data includes range data, 0 if not. The first vector always
            has range data.

        Returns
        -------
        numpy.ndarray
            Unpacked vector data as a numpy array of 3 signed ints plus a range (0 if
            has_range is False).
        """
        if np.any(vector_data > 1):
            raise ValueError(
                "unpack_one_vector method is expecting an array of bits as input."
            )

        if len(vector_data) != width * AXIS_COUNT + RANGE_BIT_WIDTH * has_range:
            raise ValueError(
                f"Invalid length {len(vector_data)} for vector data. Expected "
                f"{width * AXIS_COUNT} or {width * AXIS_COUNT + RANGE_BIT_WIDTH} if "
                f"has_range."
            )
        padding = np.zeros(8 - (width % 8), dtype=np.uint8)

        # take slices of the input data and pack from an array of bits to an array of
        # uint8 bytes
        x = np.packbits(np.concatenate((padding, vector_data[:width])))
        y = np.packbits(np.concatenate((padding, vector_data[width : 2 * width])))
        z = np.packbits(np.concatenate((padding, vector_data[2 * width : 3 * width])))

        range_string = "".join([str(i) for i in vector_data[-2:]])

        rng = int(range_string, 2) if has_range else 0

        # Convert to signed integers using twos complement
        signed_vals: np.ndarray = np.array(
            [
                MagL1a.twos_complement(x, width),
                MagL1a.twos_complement(y, width),
                MagL1a.twos_complement(z, width),
                rng,
            ],
            dtype=np.int32,
        )
        return signed_vals

    @staticmethod
    def twos_complement(value: np.ndarray, bits: int) -> np.int32:
        """
        Compute the two's complement of an integer.

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
            Two's complement of the input value, as a signed int.
        """
        integer_value = int.from_bytes(value, "big")
        if (integer_value & (1 << (bits - 1))) != 0:
            output_value = integer_value - (1 << bits)
        else:
            output_value = integer_value
        return np.int32(output_value)

    @staticmethod
    def decode_fib_zig_zag(code: np.ndarray) -> int:
        """
        Decode a fibonacci and zig-zag encoded value.

        Parameters
        ----------
        code : numpy.ndarray
            The code to decode, in the form of an array of bits (eg [0, 1, 0, 1, 1]).
            This should always end in 2 ones (which indicates the end of a fibonacci
            encoding).

        Returns
        -------
        value: int
            Signed integer value, with fibonacci and zig-zag encoding removed.
        """
        if len(code) < 2 or code[-2] != 1 or code[-1] != 1:
            raise ValueError(
                f"Error when decoding {code} - fibonacci encoded values "
                f"should end in 2 sequential ones."
            )

        # Fibonacci decoding
        code = code[:-1]
        value: int = sum(FIBONACCI_SEQUENCE[: len(code)] * code) - 1
        # Zig-zag decode (to go from uint to signed int)
        value = int((value >> 1) ^ (-(value & 1)))

        return value

    def vectors_per_second_attribute(self) -> str:
        """
        Generate a string describing the vectors per second.

        Format is {start time}:{vectors per second},{start time}:{vectors per second}
        where it's only included if vectors per second changes.

        Returns
        -------
        output_str : str
            Output string describing the vectors per second in all the packets.
        """
        output_str = ""
        last_vectors_per_second = None
        for _, packet in self.packet_definitions.items():
            vecsec = packet.vectors_per_second
            time: np.int64 = packet.start_time.to_j2000ns().astype(np.int64)
            if vecsec != last_vectors_per_second:
                if output_str == "":
                    output_str = f"{time}:{vecsec}"
                else:
                    output_str += f",{time}:{vecsec}"
                last_vectors_per_second = vecsec

        return output_str
