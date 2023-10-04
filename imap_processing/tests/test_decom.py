"""Tests aspects of the decommutation process that is generic to all instruments"""

import pytest

from imap_processing import imap_module_directory

# Scrape the repo for all packet definition XML files
xtce_documents = list(imap_module_directory.glob("*/packet_definitions/*.xml"))

# (Temporary) Mark IDEX as expected fail until packet definition is fixed
idex_document = (
    imap_module_directory / "idex" / "packet_definitions" / "idex_packet_definition.xml"
)
# Need to remove IDEX from the list in order to append it again with proper marking
xtce_documents = [item for item in xtce_documents if item != idex_document]
xtce_documents.append(
    pytest.param(
        idex_document,
        marks=pytest.mark.xfail(
            reason="Packet Definition does not include properly formatted CCSDS Header"
        ),
    )
)


@pytest.mark.parametrize("xtce_document", xtce_documents)
def test_ccsds_header(xtce_document):
    """Test if the XTCE document contains the proper CCSDS header information"""

    # Define what is expected in the XTCE document
    header_keys = [
        ("VERSION", "UINT3"),
        ("TYPE", "UINT1"),
        ("SEC_HDR_FLG", "UINT1"),
        ("PKT_APID", "UINT11"),
        ("SEG_FLGS", "UINT2"),
        ("SRC_SEQ_CTR", "UINT14"),
        ("PKT_LEN", "UINT16"),
    ]
    header_container_set = (
        "<xtce:ContainerSet>"
        '    <xtce:SequenceContainer name="CCSDSPacket">'
        "            <xtce:EntryList>"
        '                    <xtce:ParameterRefEntry parameterRef="VERSION" />'
        '                    <xtce:ParameterRefEntry parameterRef="TYPE" />'
        '                    <xtce:ParameterRefEntry parameterRef="SEC_HDR_FLG" />'
        '                    <xtce:ParameterRefEntry parameterRef="PKT_APID" />'
        '                    <xtce:ParameterRefEntry parameterRef="SEG_FLGS" />'
        '                    <xtce:ParameterRefEntry parameterRef="SRC_SEQ_CTR" />'
        '                    <xtce:ParameterRefEntry parameterRef="PKT_LEN" />'
        '                    <xtce:ParameterRefEntry parameterRef="SHCOARSE" />'
        "            </xtce:EntryList>"
        "    </xtce:SequenceContainer>"
    )

    # Read in the XTCE document
    with open(xtce_document) as f:
        document = f.read()

    # Check that each header key is defined as a Parameter
    for header_key in header_keys:
        assert (
            f'<xtce:Parameter name="{header_key[0]}" parameterTypeRef="{header_key[1]}"'
            in document
            or f'<xtce:Parameter name="{header_key[0]}" parameterTypeRef="{header_key[1].lower()}"'
            in document
        )

    # Check that the header is defined as a SequenceContainer in the XTCE file
    # First remove discrepancies in whitespace, tabs, and newlines
    header_container_set = (
        header_container_set.replace(" ", "").replace("\t", "").replace("\n", "")
    )
    document = document.replace(" ", "").replace("\t", "").replace("\n", "")
    assert header_container_set in document
