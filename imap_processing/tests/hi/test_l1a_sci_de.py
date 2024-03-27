import numpy as np

from imap_processing.hi.l1a.science_direct_event import create_dataset


def test_create_dataset():
    """Test create_dataset"""
    # dummy data to test create_dataset
    metaevent = {
        "start_bitmask_data": 0,
        "esa_step": 1,
        "subseconds": 512,
        "seconds": 433522962,
    }
    directevent_1 = {
        "start_bitmask_data": 1,
        "tof_1": 512,
        "tof_2": 512,
        "tof_3": 512,
        "de_tag": 1,
    }
    directevent_2 = {
        "start_bitmask_data": 2,
        "tof_1": 512,
        "tof_2": 512,
        "tof_3": 512,
        "de_tag": 2,
    }
    directevent_3 = {
        "start_bitmask_data": 3,
        "tof_1": 512,
        "tof_2": 512,
        "tof_3": 512,
        "de_tag": 3,
    }
    # Test for good data
    de_data_list = [metaevent, directevent_1, directevent_2, directevent_3]
    packet_time = [123] * len(de_data_list)
    dataset = create_dataset(de_data_list, packet_time)
    assert dataset["epoch"].shape == (3,)

    # Test for missing metaevent
    de_data_list = [directevent_1, directevent_2, directevent_3]
    packet_time = [123] * len(de_data_list)
    dataset = create_dataset(de_data_list, packet_time)
    assert dataset is None

    # test for first metaevent missing in the list
    de_data_list = [
        directevent_1,
        directevent_2,
        metaevent,
        directevent_1,
        directevent_2,
        directevent_3,
    ]
    packet_time = [123] * len(de_data_list)
    dataset = create_dataset(de_data_list, packet_time)
    assert dataset["epoch"].shape == (3,)
    np.testing.assert_array_equal(dataset["trigger_id"].data, [1, 2, 3])
