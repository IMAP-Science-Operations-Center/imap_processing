import unittest
from pathlib import Path

from imap_processing.swe.decommutation_swe import decom_packet


class TestSwePacketDecom(unittest.TestCase):
    def setUp(self):
        """Read test data from file
        """
        self.packet_file = Path('imap_processing/swe/tests/science_block_20221116_163611Z_idle.bin')
        self.xtce_document = Path('imap_processing/swe/swe_packet_definition.xml')
        self.data_packet_list = decom_packet(self.packet_file, self.xtce_document)

    def test_total_packets_in_data_file(self):
        total_packets = 23
        assert len(self.data_packet_list) == total_packets

    def test_ccsds_header(self):
        """Test if packet header contains default CCSDS header
        These are the field required in CCSDS header:
            'VERSION', 'TYPE', 'SEC_HDR_FLG', 'PKT_APID', 'SEG_FLGS', 'SRC_SEQ_CTR', 'PKT_LEN'
        """

        # Required CCSDS header fields
        ccsds_header_keys = ['VERSION', 'TYPE', 'SEC_HDR_FLG', 'PKT_APID', 'SEG_FLGS', 'SRC_SEQ_CTR', 'PKT_LEN']

        # self.data_packet_list[0].header is one way to get the header data. Another way to get it
        # is using list method. Eg. ccsds_header = self.data_packet_list[0][0]. Each packet's 0th index
        # has header data and index 1 has data.

        # First way to get header data
        ccsds_header = self.data_packet_list[0].header
        assert all(key in ccsds_header.keys() for key in ccsds_header_keys)

        # Second way to get header data
        ccsds_header = self.data_packet_list[0][0]
        assert all(key in ccsds_header.keys() for key in ccsds_header_keys)

    def test_ways_to_get_data(self):
        """Test if data can be retrieved using different ways
        """

        # First way to get metadata and data array
        data_value_using_key = self.data_packet_list[0].data
        # First way to get metadata and data array
        data_value_using_list = self.data_packet_list[0][1]
        assert data_value_using_key == data_value_using_list

    def test_enumerated_value(self):
        """Test if enumerated value is derived correctly
        """
        # CEM Nominal status bit:
        #     '1' -- nominal,
        #     '0' -- not nomimal
        parameter_name = 'CEM_NOMINAL_ONLY'
        first_packet_data = self.data_packet_list[0].data

        if first_packet_data[f'{parameter_name}'].raw_value == 1:
            assert first_packet_data[f'{parameter_name}'].derived_value == 'NOMINAL'
        if first_packet_data[f'{parameter_name}'].raw_value == 0:
            assert first_packet_data[f'{parameter_name}'].derived_value == 'NOT_NOMINAL'

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
