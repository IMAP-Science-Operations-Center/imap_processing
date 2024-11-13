import random

import pytest


@pytest.fixture(scope="session")
def hi_test_data_path(imap_tests_path):
    return imap_tests_path / "hi" / "test_data"


@pytest.fixture(scope="session")
def hi_l0_test_data_path(hi_test_data_path):
    return hi_test_data_path / "l0"


@pytest.fixture(scope="session")
def hi_l1a_test_data_path(hi_test_data_path):
    return hi_test_data_path / "l1a"


def create_metaevent(esa_step, met_subseconds, met_seconds):
    start_bitmask_data = 0  # META
    return (
        f"{start_bitmask_data:02b}{esa_step:04b}{met_subseconds:010b}{met_seconds:032b}"
    )


def create_directevent(tof_1, tof_2, tof_3, de_tag):
    start_bitmask_data = random.choice([1, 2, 3])  # Detector A, B, C
    return f"{start_bitmask_data:02b}{tof_1:010b}{tof_2:010b}{tof_3:010b}{de_tag:016b}"


def ccsds_header_data(apid, pkt_len):
    """Create binary data for CCSDS header with apid provided."""
    # CCSDS primary header
    # 3 bits - Version number
    # 1 bit - Packet type
    # 1 bit - Secondary header flag
    # 16 bits - APID
    # 2 bits - Sequence flag
    # 14 bits - Packet sequence count
    # 16 bits - Packet length
    return f"{0:03b}{0:01b}{1:01b}{apid:011b}{1:02b}{0:014b}{pkt_len:016b}"


def check_sum(bits_size):
    """Create check test sum."""
    return f"{0:0{bits_size}b}"


@pytest.fixture()
def create_de_data(tmp_path):
    """Fixture to create fake direct event data. Note that there has been no
    effort to make simulate this data meaning that packets are not self
    consistent and values may not make sense.

    TODO: remove this once we have good sample data.
    TODO: remove "S311" from pyproject.toml when we remove this.
    """

    def create_data(apid):
        """
        The function that generates the fake direct event data and returns
        the Path to the binary file.

        Parameters
        ----------
        apid : int
            CCSDS packet apid

        Returns
        -------
        Path to the binary file with fake direct event data
        """
        num_packets = 4
        packets_data = []
        # This MET seconds evaluates to 2023-09-27T16:04:49.150
        met_seconds = 433522962
        esa_step = 0
        spare = f"{0:032b}"
        # To use a consistent random number
        random.seed(0)
        for index in range(num_packets):
            current_data = []
            if index % 2 == 0:
                # Two packets per ESA step
                # Every even index, event starts with metaevent
                if esa_step == 9:
                    esa_step = 0
                esa_step += 1
                met_subseconds = random.randint(0, 1023)
                met_seconds += 3600
                current_data.append(
                    create_metaevent(esa_step, met_subseconds, met_seconds)
                )
            # Random length for direct events. It could be empty
            num_directevents = random.randint(0, 10)
            for _ in range(num_directevents):
                # We don't always get triple coincidence. To make data
                # more realistic, we will use random.choices.
                # This line will select a random integer between 0 and 1023
                # with a probability of 0.xx and the value 1023 with a
                # probability of 0.xx. The `k=1` argument specifies that one
                # item should be chosen, and `[0]` is used to extract the
                # single item from the resulting list.
                tof_1, tof_2, tof_3 = random.choices(
                    [random.randint(0, 1022), 1023], weights=[0.45, 0.55], k=3
                )
                de_tag = random.randint(1, 65535)
                current_data.append(create_directevent(tof_1, tof_2, tof_3, de_tag))

            # create a CCSDS data using current data
            # pkt_len = 4 byte (MET seconds) + 4 byte (spare) +
            # 6 byte (direct event) * number of direct event + 2 byte (checksum) - 1å
            if index % 2 == 0:
                # Add 6 bytes for metaevent
                pkt_len = 4 + 4 + 6 + 6 * num_directevents + 2 - 1
            else:
                pkt_len = 4 + 4 + 6 * num_directevents + 2 - 1
            ccsds_data = (
                ccsds_header_data(apid, pkt_len)
                + f"{met_seconds:032b}"
                + spare
                + "".join(current_data)
                + check_sum(bits_size=16)
            )
            packets_data.append(ccsds_data)
        # Join all the events as one binary string
        packets_data = "".join(packets_data)
        # write data to pkts file
        with open(tmp_path / "imap_hi_l0_sdc-test-data_20240318_v000.pkts", "wb") as f:
            byte_data = int(packets_data, 2).to_bytes(
                (len(packets_data) + 7) // 8, byteorder="big"
            )
            f.write(byte_data)
        return tmp_path / "imap_hi_l0_sdc-test-data_20240318_v000.pkts"

    return create_data
