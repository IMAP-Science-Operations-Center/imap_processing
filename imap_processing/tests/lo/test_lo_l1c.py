from imap_processing import imap_module_directory
from imap_processing.cdf.utils import load_cdf
from imap_processing.lo.l1c.lo_l1c import lo_l1c


def test_lo_l1b():
    # Arrange
    de_file = (
        imap_module_directory
        / "tests/lo/test_cdfs/imap_lo_l1b_de_20100101_v001.cdf"
    )
    data = {}
    dataset = load_cdf(de_file)
    data[dataset.attrs["Logical_source"]] = dataset

    expected_out = "imap_lo_l1c_pset_20100101_v001.cdf"
    # Act
    output_file = lo_l1c(data)

    # Assert
    assert expected_out == output_file.name
