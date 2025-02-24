"""Tests the decommutation process for CoDICE CCSDS Packets

The tests within ensure that the test L0 data can be decommed and result in
the expected APIDs, number of packets, and contain valid CCSDS header contents.
"""

import pytest
import xarray as xr

from imap_processing.codice import codice_l0
from imap_processing.codice.utils import CODICEAPID

from .conftest import TEST_L0_FILE

pytestmark = pytest.mark.external_test_data

EXPECTED_RESULTS = {
    CODICEAPID.COD_NHK: 31778,
    CODICEAPID.COD_LO_IAL: 18917,
    CODICEAPID.COD_LO_PHA: 616,
    CODICEAPID.COD_LO_SW_PRIORITY_COUNTS: 77,
    CODICEAPID.COD_LO_SW_SPECIES_COUNTS: 77,
    CODICEAPID.COD_LO_NSW_SPECIES_COUNTS: 77,
    CODICEAPID.COD_LO_SW_ANGULAR_COUNTS: 77,
    CODICEAPID.COD_LO_NSW_ANGULAR_COUNTS: 77,
    CODICEAPID.COD_LO_NSW_PRIORITY_COUNTS: 77,
    CODICEAPID.COD_LO_INST_COUNTS_AGGREGATED: 77,
    CODICEAPID.COD_LO_INST_COUNTS_SINGLES: 77,
    CODICEAPID.COD_HI_IAL: 18883,
    CODICEAPID.COD_HI_PHA: 633,
    CODICEAPID.COD_HI_INST_COUNTS_AGGREGATED: 77,
    CODICEAPID.COD_HI_INST_COUNTS_SINGLES: 77,
    CODICEAPID.COD_HI_OMNI_SPECIES_COUNTS: 77,
    CODICEAPID.COD_HI_SECT_SPECIES_COUNTS: 77,
    CODICEAPID.COD_HI_INST_COUNTS_PRIORITIES: 77,
}


@pytest.fixture(scope="session")
def decom_test_data(_download_test_data) -> xr.Dataset:
    """Read test data from file and return a decommutated housekeeping packet.

    Returns
    -------
    packet : xr.Dataset
        A decommutated housekeeping packet
    """

    packet = codice_l0.decom_packets(TEST_L0_FILE)

    return packet


@pytest.mark.parametrize("apid", EXPECTED_RESULTS.keys())
def test_ccsds_headers(decom_test_data: xr.Dataset, apid):
    """Tests that the CCSDS headers are present in the decommed data"""

    for ccsds_header_field in [
        "shcoarse",
        "version",
        "type",
        "sec_hdr_flg",
        "pkt_apid",
        "seq_flgs",
        "src_seq_ctr",
        "pkt_len",
    ]:
        assert ccsds_header_field in decom_test_data[apid]


@pytest.mark.parametrize("apid", EXPECTED_RESULTS.keys())
def test_expected_apids(decom_test_data: xr.Dataset, apid):
    """Tests that the expected APIDs are present in the decommed data"""

    assert apid in decom_test_data


@pytest.mark.parametrize("apid, expected_num_packets", EXPECTED_RESULTS.items())
def test_expected_total_packets(
    decom_test_data: xr.Dataset, apid, expected_num_packets
):
    """Test if total packets in the decommed data is correct"""

    assert len(decom_test_data[apid].epoch) == expected_num_packets
