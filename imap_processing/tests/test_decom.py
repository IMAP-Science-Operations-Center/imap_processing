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
        ("VERSION", "3"),
        ("TYPE", "1"),
        ("SEC_HDR_FLG", "1"),
        ("PKT_APID", "11"),
        ("SEQ_FLGS", "2"),
        ("SRC_SEQ_CTR", "14"),
        ("PKT_LEN", "16"),
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
        # Format for old XTCE generation format
        old_xtce_gen = (
            f'<xtce:Parameter name="{key[0]}" parameterTypeRef="UINT{key[1]}"'
            in document
            or f'<xtce:Parameter name="{key[0]}" parameterTypeRef="uint{key[1]}"'
            in document
        )
        # Format for new XTCE generation format
        new_xtce_gen = (
            f'<xtce:IntegerParameterType name="{key[0]}" signed="false">'
            f"\n\t\t\t\t"
            f'<xtce:IntegerDataEncoding sizeInBits="{key[1]}" '
            f'encoding="unsigned" />' in document
        )

        assert any((old_xtce_gen, new_xtce_gen))

    # Check that the header is defined as a SequenceContainer in the XTCE file
    # First remove discrepancies in whitespace, tabs, and newlines
    header_entry_list = (
        header_entry_list.replace(" ", "").replace("\t", "").replace("\n", "")
    )
    document = document.replace(" ", "").replace("\t", "").replace("\n", "")
    assert header_entry_list in document
