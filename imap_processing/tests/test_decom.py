"""Tests aspects of the decommutation process that is generic to all instruments"""

import pytest

from imap_processing import imap_module_directory

# Scrape the repo for all packet definition XML files
xtce_document_list = imap_module_directory.glob("*/packet_definitions/*.xml")


@pytest.mark.parametrize("xtce_document", xtce_document_list)
def test_ccsds_header(xtce_document):
    """Test if the XTCE document contains the proper CCSDS header information"""

    # Define what is expected in the XTCE document
    header_keys = [
        ("VERSION", "UINT3"),
        ("TYPE", "UINT1"),
        ("SEC_HDR_FLG", "UINT1"),
        ("PKT_APID", "UINT11"),
        ("SEQ_FLGS", "UINT2"),
        ("SRC_SEQ_CTR", "UINT14"),
        ("PKT_LEN", "UINT16"),
    ]
    header_entry_list = (
        "<xtce:EntryList>"
        '        <xtce:ParameterRefEntry parameterRef="VERSION" />'
        '        <xtce:ParameterRefEntry parameterRef="TYPE" />'
        '        <xtce:ParameterRefEntry parameterRef="SEC_HDR_FLG" />'
        '        <xtce:ParameterRefEntry parameterRef="PKT_APID" />'
        '        <xtce:ParameterRefEntry parameterRef="SEQ_FLGS" />'
        '        <xtce:ParameterRefEntry parameterRef="SRC_SEQ_CTR" />'
        '        <xtce:ParameterRefEntry parameterRef="PKT_LEN" />'
        "</xtce:EntryList>"
    )

    # Read in the XTCE document
    with open(xtce_document) as f:
        document = f.read()

    # Check that each header key is defined as a Parameter
    for key in header_keys:
        assert (
            f'<xtce:Parameter name="{key[0]}" parameterTypeRef="{key[1]}"' in document
            or f'<xtce:Parameter name="{key[0]}" parameterTypeRef="{key[1].lower()}"'
            in document
        )

    # Check that the header is defined as a SequenceContainer in the XTCE file
    # First remove discrepancies in whitespace, tabs, and newlines
    header_entry_list = (
        header_entry_list.replace(" ", "").replace("\t", "").replace("\n", "")
    )
    document = document.replace(" ", "").replace("\t", "").replace("\n", "")
    assert header_entry_list in document
