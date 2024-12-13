from imap_processing.hi.l1a.science_direct_event import create_dataset


def test_create_dataset():
    """Test create_dataset"""
    # dummy data to test create_dataset
    data_dict = {
        "trigger_id": [1, 2, 3],
        "tof_1": [512, 512, 512],
        "tof_2": [512, 512, 512],
        "tof_3": [512, 512, 512],
        "de_tag": [1, 2, 3],
        "meta_seconds": [433522962, 433522962, 433522962],
        "meta_subseconds": [512, 512, 512],
        "esa_step": [4, 4, 4],
        "ccsds_met": [433522961, 433522961, 433522961],
        "src_seq_ctr": [10, 10, 10],
        "pkt_len": [146, 146, 146],
        "last_spin_num": [4, 4, 4],
        "spin_invalids": [0, 0, 0],
    }

    # Test for good data
    dataset = create_dataset(data_dict)
    assert dataset["epoch"].shape == (3,)
