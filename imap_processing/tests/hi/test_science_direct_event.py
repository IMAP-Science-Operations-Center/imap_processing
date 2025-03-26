import numpy as np

from imap_processing.hi.l1a.science_direct_event import (
    create_dataset,
    parse_direct_events,
)


def test_parse_direct_events():
    """Test coverage for parse_direct_events function."""
    # Generate fake, binary blob using random numbers
    np.random.seed(2)
    n_events = 10_000
    exp_dict = dict()
    exp_dict["trigger_id"] = np.random.randint(1, 4, size=n_events, dtype=np.uint8)
    exp_dict["de_tag"] = np.random.randint(0, 2**16, size=n_events, dtype=np.uint16)
    exp_dict["tof_1"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)
    exp_dict["tof_2"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)
    exp_dict["tof_3"] = np.random.randint(0, 2**10, size=n_events, dtype=np.uint16)

    # Encode the random events data into a bit-string
    bin_str = ""
    for i in range(n_events):
        bin_str += f"{exp_dict['de_tag'][i]:016b}"  # 16-bits for de_tag
        bin_str += f"{exp_dict['trigger_id'][i]:02b}"  # 2-bits for trigger_id
        bin_str += f"{exp_dict['tof_1'][i]:010b}"  # 10-bits for tof_1
        bin_str += f"{exp_dict['tof_2'][i]:010b}"  # 10-bits for tof_2
        bin_str += f"{exp_dict['tof_3'][i]:010b}"  # 10-bits for tof_3
    # Convert the bit-string into a bytes object
    bytes_obj = bytes([int(bin_str[i : i + 8], 2) for i in range(0, len(bin_str), 8)])
    # Parse the fake events and check values
    de_dict = parse_direct_events(bytes_obj)
    for key in exp_dict.keys():
        np.testing.assert_array_equal(de_dict[key], exp_dict[key])


def test_create_dataset():
    """Test create_dataset"""
    # dummy data to test create_dataset
    n_packets = 3
    n_events = 4
    data_dict = {
        "ccsds_met": np.arange(n_packets) + 433522961,
        "src_seq_ctr": np.arange(n_packets) + 10,
        "pkt_len": np.full(n_packets, 146),
        "last_spin_num": np.arange(n_packets) + 3,
        "spin_invalids": np.zeros(n_packets),
        "esa_step": np.full(n_packets, 4),
        "meta_seconds": np.arange(n_packets) + 433522960,
        "meta_subseconds": np.full(n_packets, 500),
        "trigger_id": [1 + i % 3 for i in range(n_events)],
        "tof_1": [450] * n_events,
        "tof_2": [450] * n_events,
        "tof_3": [450] * n_events,
        "de_tag": [i for i in range(n_events)],
        "ccsds_index": [int(i / n_events * n_packets) for i in range(n_events)],
    }

    # Test for good data
    dataset = create_dataset(data_dict)
    assert dataset["epoch"].shape == (n_packets,)
    assert dataset["event_met"].shape == (n_events,)
