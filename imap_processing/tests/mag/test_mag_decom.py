import inspect
from pathlib import Path

import pandas as pd

from imap_processing.cdf import global_attrs
from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes
from imap_processing.cdf.utils import load_cdf, write_cdf
from imap_processing.mag import mag_cdf_attrs
from imap_processing.mag.l0.decom_mag import decom_packets, generate_dataset


def test_mag_decom():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_l0_test_data.pkts"
    packets = decom_packets(str(burst_test_file))

    l0 = packets["burst"] + packets["norm"]

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

        # Remove bytes for header and previous attributes from CCSDS_HEX,
        # remaining bytes are vectors
        assert (
            test.VECTORS.tobytes().hex()
            == expected_output["CCSDS_HEX"][index][54:].lower()
        )

    assert len(l0) == len(expected_output.index)


def test_mag_raw_xarray():
    current_directory = Path(__file__).parent
    burst_test_file = current_directory / "mag_l0_test_data.pkts"
    packets = decom_packets(str(burst_test_file))
    l0_norm = packets["norm"]
    l0_burst = packets["burst"]

    template = ImapCdfAttributes()
    template.add_instrument_global_attrs("mag")
    template.add_instrument_variable_attrs("mag", "l1a")

    norm_data = generate_dataset(l0_norm, template, "imap_mag_l1a_norm-raw")
    burst_data = generate_dataset(
        l0_burst, template, "imap_mag_l1a_burst-raw")

    required_attrs = list(
        global_attrs.GlobalInstrumentAttrs("", "", "").output().keys()
    )

    for required_attr in required_attrs:
        assert(required_attr in list(norm_data.attrs.keys()))

    # TODO: Fails because we have a value in GlobalInstrumentAttrs that is not in the
    # default global schema

    assert all([item in list(norm_data.attrs.keys()) for item in required_attrs])
    assert all([item is not None for _, item in norm_data.attrs.items()])

    assert all([item in list(burst_data.attrs.keys()) for item in required_attrs])
    assert all([item is not None for _, item in burst_data.attrs.items()])

    expected_norm_len = 17
    assert norm_data.sizes["epoch"] == expected_norm_len

    expected_burst_len = 19
    assert burst_data.sizes["epoch"] == expected_burst_len


def test_mag_raw_cdf_generation():
    current_directory = Path(__file__).parent
    test_file = current_directory / "mag_l0_test_data.pkts"
    packets = decom_packets(str(test_file))
    l0_norm = packets["norm"]
    l0_burst = packets["burst"]

    template = ImapCdfAttributes()
    template.add_instrument_global_attrs("mag")
    template.add_instrument_variable_attrs("mag", "l1a")

    norm_data = generate_dataset(l0_norm, template, "imap_mag_l1a_norm-raw")
    burst_data = generate_dataset(
        l0_burst, template, "imap_mag_l1a_burst-raw"
    )

    output = write_cdf(norm_data)
    assert output.exists()
    assert output.name == "imap_mag_l1a_norm-raw_20231025_v001.cdf"

    input_xarray = load_cdf(output)
    assert input_xarray.attrs.keys() == norm_data.attrs.keys()

    output = write_cdf(burst_data)
    assert output.exists()
    assert output.name == "imap_mag_l1a_burst-raw_20231025_v001.cdf"

    input_xarray = load_cdf(output)
    assert input_xarray.attrs.keys() == burst_data.attrs.keys()
