from pathlib import Path

import pandas as pd
import pytest
from cdflib.xarray import cdf_to_xarray

from imap_processing.cdf import global_attrs
from imap_processing.cdf.utils import write_cdf
from imap_processing.mag.l0.decom_mag import decom_packets, export_to_xarray


def test_mag_decom():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_l0_test_data.pkts"
    l0 = decom_packets(burst_test_file)

    expected_output = pd.read_csv(current_directory / "mag_l0_test_output.csv")
    for index, test in enumerate(l0):
        assert test.ccsds_header.PKT_APID == expected_output["PHAPID"][index]
        assert test.ccsds_header.SRC_SEQ_CTR == expected_output["PHSEQCNT"][index]
        assert test.ccsds_header.PKT_LEN == expected_output["PHDLEN"][index]

        assert test.SHCOARSE == expected_output["SHCOARSE"][index]
        assert test.PUS_STYPE == expected_output["PUS_STYPE"][index]
        assert test.PUS_SSUBTYPE == expected_output["PUS_SSUBTYPE"][index]
        assert test.COMPRESSION == expected_output["COMPRESSION"][index]
        assert test.MAGO_ACT == expected_output["FOB_ACT"][index]
        assert test.MAGI_ACT == expected_output["FIB_ACT"][index]
        assert test.PRI_SENS == expected_output["PRI_SENS"][index]
        assert test.PRI_VECSEC == expected_output["PRI_VECSEC"][index]
        assert test.SEC_VECSEC == expected_output["SEC_VECSEC"][index]
        assert test.PRI_COARSETM == expected_output["PRI_COARSETM"][index]
        assert test.PRI_FNTM == expected_output["PRI_FNTM"][index]
        assert test.SEC_COARSETM == expected_output["SEC_COARSETM"][index]
        assert test.SEC_FNTM == expected_output["SEC_FNTM"][index]

    assert len(l0) == len(expected_output.index)


def test_mag_raw_xarray():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_l0_test_data.pkts"
    l0 = decom_packets(str(burst_test_file))

    norm_data, burst_data = export_to_xarray(l0)
    required_attrs = list(
        global_attrs.GlobalInstrumentAttrs("", "", "").output().keys()
    )

    assert all([item in list(norm_data.attrs.keys()) for item in required_attrs])
    assert all([item is not None for _, item in norm_data.attrs.items()])

    assert all([item in list(burst_data.attrs.keys()) for item in required_attrs])
    assert all([item is not None for _, item in burst_data.attrs.items()])

    expected_norm_len = 17
    assert norm_data.dims["Epoch"] == expected_norm_len

    expected_burst_len = 19
    assert burst_data.dims["Epoch"] == expected_burst_len


@pytest.mark.xfail(reason="Fix for updated cdflib")
def test_mag_raw_cdf_generation(tmp_path):
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l0_test_data.pkts"
    l0 = decom_packets(str(test_file))

    norm_data, burst_data = export_to_xarray(l0)

    test_data_path_norm = tmp_path / "mag_l1a_raw-normal_20210101_20210102_v01-01.cdf"

    assert not test_data_path_norm.exists()
    output = write_cdf(norm_data, test_data_path_norm)
    assert test_data_path_norm.exists()

    input_xarray = cdf_to_xarray(output)
    assert input_xarray.attrs.keys() == norm_data.attrs.keys()

    test_data_path_burst = tmp_path / "mag_l1a_raw-burst_20210101_20210102_v01-01.cdf"

    assert not test_data_path_burst.exists()
    output = write_cdf(burst_data, test_data_path_burst)
    assert test_data_path_burst.exists()

    input_xarray = cdf_to_xarray(output)
    assert input_xarray.attrs.keys() == burst_data.attrs.keys()
