from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1b.lo_l1b import lo_l1b


def test_lo_l1b():
    # Arrange
    de_file = (
        imap_module_directory
        / "tests/lo/test_l1a_cdfs/imap_lo_l1a_de_20100101_v001.cdf"
    )
    spin_file = (
        imap_module_directory
        / "tests/lo/test_l1a_cdfs/imap_lo_l1a_spin_20100101_v001.cdf"
    )
    data = {}
    for file in [de_file, spin_file]:
        dataset = load_cdf(file)
        data[dataset.attrs["Logical_source"]] = dataset

    expected_out = "imap_lo_l1b_de_20100101_v001.cdf"
    # Act
    output_file = lo_l1b(data)

    # Assert
    assert expected_out == output_file.name
