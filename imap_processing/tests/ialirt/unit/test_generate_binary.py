from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.generate_binary import generate_binary


def test_generate_binary():
    """
    Test test_generate_binary function.
    """
    test_file = "tests/ialirt/data/l0/apid_478.bin"
    packet_files = imap_module_directory / test_file
    packet_definition = imap_module_directory / "ialirt/packet_definitions/ialirt.xml"
    ingest_data = generate_binary(packet_files, packet_definition)
    assert len(ingest_data) == 44429
    assert ingest_data[0]["met"] == 482934903 + 0.9609375
