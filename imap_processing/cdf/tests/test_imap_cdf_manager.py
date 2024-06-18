# from pathlib import Path
# from imap_processing.cdf.cdf_attribute_manager import CdfAttributeManager
# This is a test2
from imap_processing.cdf.imap_cdf_manager import ImapCdfAttributes

# test add_instrument_global_attrs(self, instrument: str):


def test_add_instrument_global_attrs():
    # Create an ImapCdfAttributes object
    imap_cdf_manager = ImapCdfAttributes()
    # Call function
    imap_cdf_manager.add_instrument_global_attrs("instrument")

    # Testing the actual function
    imap_instrument = imap_cdf_manager.get_global_attributes("imap_test_T1_test")
    assert imap_instrument["Data_type"] == "T1_test-one>Test-1 test one"
    assert imap_instrument["Project"] == "STP>Solar-Terrestrial Physics"

    # Testing reloading data
    imap_cdf_manager.add_instrument_global_attrs("swe")

    # Testing again
    swe_instrument = imap_cdf_manager.get_global_attributes("imap_swe_l1a_sci")
    assert swe_instrument["Data_type"] == "L1A_SCI>Level-1A Science data"
    assert swe_instrument["Project"] == "STP>Solar-Terrestrial Physics"


# test add_instrument_variable_attrs(self, instrument: str, level: str):
def test_add_instrument_variable_attrs():
    # Create an ImapCdfAttributes object
    imap_cdf_manager = ImapCdfAttributes()
    # Call function
    imap_cdf_manager.add_instrument_variable_attrs("instrument", "level")

    # Testing the actual function
    # imap_instrument = imap_cdf_manager.get_variable_attributes("imap_test_T1_test")
    # assert imap_instrument["DEPEND_0"] == "test_depend"
    # assert imap_instrument["CATDESC"] == "test time"
    # assert imap_instrument["VALIDMAX"] == 10

    # Testing reloading data
    imap_cdf_manager.add_instrument_variable_attrs("ultra", "l1b")

    # Testing again
    # ultra_instrument = imap_cdf_manager.get_variable_attributes("imap_swe_l1a_sci")
    # assert ultra_instrument["DEPEND_0"] == "epoch"
    # assert ultra_instrument["CATDESC"] == "Time, number of nanoseconds since J2000 with leap seconds included"
    # assert ultra_instrument["VALIDMAX"] == 9223372036854775807
