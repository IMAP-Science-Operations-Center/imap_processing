#README

This folder contains the following files:

1) H45_APP_NHK.bin, H45_SCI_DE.bin, H45_SCI_CNT.bin
2) 20231026_H45_APP_NHK.csv
3) 20231026_H45_SCI_DE.csv
4) 20231026_H45_SCI_CNT.csv


1) Three Binary files that the following Packets are contained inside of H45_CCSDS_PACKETS.zip:

	a) 20231026_H45_APP_NHK.bin -> 100 H45_APP_NHK CCSDS packets in a binary file

	b) 20231026_H45_SCI_DE.bin  -> 100 H45_SCI_DE CCSDS packets in a binary file

	c) 20231026_H45_SCI_CNT.bin -> 100 H45_SCI_DE_CCSDS packets in a binary file

Each packet contains the 6-byte CCSDS Primary Header. The primary header
contains the following fields:
    CCSDS_VER           -> 3 bits
    CCSDS_PKT_TYP       -> 1 bits
    CCSDS_SEC_HDR_FLAG  -> 1 bits
    CCSDS_APID          -> 11 bits
    CCSDS_GRP_FLAGS     -> 2 bits
    CCSDS_SSC           -> 14 bits
    CCSDS_PKT_DATA_LN   -> 16 bits*

    *The CCSDS_PKT_DATA_LN field is defined to be as:
     "Data size, number of bytes following the 'Packet Length'
     (SCLK Size + Data Size) minus 1"

2) 20231026_H45_APP_NHK.csv

	This CSV file contains all the fields and values of the NHK packet. There are 100 entries
which represent 100 NHK packets. All of the column headers are the name of the fields in the H45_APP_NHK packet which is defined in the 28650.02_TLMDEF.xlsx spreadsheet.

3) 20231026_H45_SCI_DE.csv
This CSV file contains the pertinent DE_TOF data (not all fields of the packets). The column headers are as follows:
	- DE Time Tag: A randomly sampled 16-bit integer.
	- Start Bit Mask: A randomly sampled 2-bit integer.
	- TOF1: A 10-bit value that is randomly sampled from a normal distribution.
	- TOF2: A 10-bit value that is randomly sampled from a normal distribution.
	- TOF3: A 10-bit value that is randomly sampled from a normal distribution.

4) 20231026_H45_SCI_CNT.csv
This CSV file contains the pertinent 12-bit counter data (not all the fields in the packets). The column headers are as follows:
	- 8 12-bit counters prefixed with "SH_": These are 12-bit values randomly sampled from a normal distribution.
	- 11 12-bit counters prefixed with "LSH_": These are 12-bit values randomly sampled from a normal distribution.
	- totalA: 12-bit value randomly sampled from normal distribution
	- totalB: 12-bit value randomly sampled from normal distribution
	- totalC: 12-bit value randomly sampled from normal distribution
	- FEE_DE_SENT: 12-bit value randomly sampled from normal distribution
	- FEE_DE_RECD: 12-bit value randomly sampled from normal distribution
