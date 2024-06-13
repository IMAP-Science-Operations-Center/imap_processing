from imap_processing.mag.l1c.mag_l1c import mag_l1c


def test_mag_attributes(mag_l1a_dataset):
    # Fixture from test_mag_l1b.py, since L1A and L1B are very similar
    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-mago"]

    output = mag_l1c(mag_l1a_dataset, mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-mago"

    mag_l1a_dataset.attrs["Logical_source"] = ["imap_mag_l1b_norm-magi"]

    output = mag_l1c(mag_l1a_dataset, mag_l1a_dataset, "v001")
    assert output.attrs["Logical_source"] == "imap_mag_l1c_norm-magi"

    assert output.attrs["Data_level"] == "L1C"
