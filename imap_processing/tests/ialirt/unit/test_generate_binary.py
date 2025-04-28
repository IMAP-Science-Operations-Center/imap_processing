from imap_processing import imap_module_directory
from imap_processing.ialirt.l0.generate_binary import generate_binary


def test_generate_binary():
    """
    Test test_generate_binary function.
    """
    test_file = "tests/ialirt/data/l0/apid_478.bin"
    packet_files = imap_module_directory / test_file
    packet_definition = imap_module_directory / "ialirt/packet_definitions/ialirt.xml"
    binary_blob_data, time_data = generate_binary(packet_files, packet_definition)
    assert len(binary_blob_data) == 44429
    assert len(time_data) == 44429
    assert time_data[0] == 482934903 + 62976
