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
    list_with_meta_first = [metaevent, directevent_1, directevent_2, directevent_3]
    packet_time = [123] * len(list_with_meta_first)
    dataset = create_dataset(list_with_meta_first, packet_time)
    assert dataset["epoch"].shape == (3,)

    # Test for missing metaevent
    list_with_no_meta = [directevent_1, directevent_2, directevent_3]
    packet_time = [123] * len(list_with_no_meta)
    dataset = create_dataset(list_with_no_meta, packet_time)
    assert dataset is None

    # test for first metaevent missing in the list
    list_with_meta_middle = [
        directevent_1,
        directevent_2,
        metaevent,
        directevent_1,
        directevent_2,
        directevent_3,
    ]
    packet_time = [123] * len(list_with_meta_middle)
    dataset = create_dataset(list_with_meta_middle, packet_time)
    assert dataset["epoch"].shape == (3,)
    np.testing.assert_array_equal(dataset["trigger_id"].data, [1, 2, 3])

    list_with_meta_last = [
        directevent_1,
        directevent_2,
        directevent_3,
        directevent_1,
        directevent_2,
        metaevent,
    ]
    packet_time = [123] * len(list_with_meta_last)
    dataset = create_dataset(list_with_meta_last, packet_time)
    assert dataset["epoch"].shape == (0,)
