"""Tests aspects of the decommutation process that is generic to all instruments"""

import glob

import pytest

from imap_processing import imap_module_directory

xtce_documents = glob.glob(f"{imap_module_directory}/*/packet_definitions/*.xml")
xtce_documents = [
    item for item in xtce_documents if "idex_packet_definition.xml" not in item
]  # Temporary until IDEX packet definition is fixed


@pytest.mark.parametrize("xtce_document", xtce_documents)
def test_ccsds_header(xtce_document):
    """Test if the XTCE document contains the proper CCSDS header information"""

    print(f"\nTesting {xtce_document}")

    # Define what is expected in the XTCE document
    header_keys = [
        "VERSION",
        "TYPE",
        "SEC_HDR_FLG",
        "PKT_APID",
        "SEG_FLGS",
        "SRC_SEQ_CTR",
        "PKT_LEN",
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
        assert f'<xtce:Parameter name="{header_key}" parameterTypeRef=' in document

    # Check that the header is defined as a SequenceContainer in the XTCE file
    # First remove discrepancies in whitespace, tabs, and newlines
    header_container_set = (
        header_container_set.replace(" ", "").replace("\t", "").replace("\n", "")
    )
    document = document.replace(" ", "").replace("\t", "").replace("\n", "")
    assert header_container_set in document
