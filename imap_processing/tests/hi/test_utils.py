"""Test coverage for imap_processing.hi.utils.py"""

from imap_processing.hi.utils import HIAPID


def test_hiapid():
    """Test coverage for HIAPID class"""
    hi_apid = HIAPID(754)
    assert isinstance(hi_apid, HIAPID)
    assert hi_apid.name == "H45_APP_NHK"
    assert hi_apid.sensor == "45sensor"

    hi_apid = HIAPID["H90_SCI_CNT"]
    assert hi_apid.value == 833
    assert hi_apid.sensor == "90sensor"
