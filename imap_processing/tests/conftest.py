"""Global pytest configuration for the package."""
import imap_data_access
import pytest


@pytest.fixture(autouse=True)
def _set_global_config(monkeypatch, tmp_path):
    """Set the global data directory to a temporary directory."""
    monkeypatch.setitem(imap_data_access.config, "DATA_DIR", tmp_path)
    monkeypatch.setitem(
        imap_data_access.config, "DATA_ACCESS_URL", "https://api.test.com"
    )


def ccsds_header_data(apid, pkt_len):
    """Create binary data for CCSDS header with apid provided."""
    # CCSDS primary header
    # 3 bits - Version number
    # 1 bit - Packet type
    # 1 bit - Secondary header flag
    # 16 bits - APID
    # 2 bits - Sequence flag
    # 14 bits - Packet sequence count
    # 16 bits - Packet length
    return f"{0:03b}{0:01b}{1:01b}{apid:011b}{1:02b}{0:014b}{pkt_len:016b}"


def check_sum(bits_size):
    """Create check test sum."""
    return f"{0:0{bits_size}b}"
