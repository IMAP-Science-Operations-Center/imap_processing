from pathlib import Path

import pytest

from imap_processing import imap_module_directory
from imap_processing.cdf.utils import write_cdf
from imap_processing.idex.l1.idex_l1 import PacketParser
from imap_processing.idex.l2.idex_l2 import read_and_return


@pytest.fixture()
def decom_test_data():
    test_file = Path(
        f"{imap_module_directory}/tests/idex/imap_idex_l0_raw_20230725_v001.pkts"
    )
    return PacketParser(test_file, "v001").data


@pytest.fixture()
def l1_cdf(decom_test_data):
    """
    from imap_processing.idex.idex_packet_parser import PacketParser
    l0_file = "imap_processing/tests/idex/imap_idex_l0_sci_20230725_v001.pkts"
    l1_data = PacketParser(l0_file, data_version)
    l1_data.write_l1_cdf()
    """

    return write_cdf(decom_test_data)


def test_read_and_return(l1_cdf):
    dataset = read_and_return(l1_cdf)

    assert dataset == decom_test_data
