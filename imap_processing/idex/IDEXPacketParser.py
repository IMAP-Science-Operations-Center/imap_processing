import bitstring
import numpy as np
from space_packet_parser import parser, xtcedef
from imap_processing import packet_definition_directory


class IDEXPacketParser():
    def __init__(self, idex_packet_file: str):

        idex_xtce = f"{packet_definition_directory}/idex_packet_definition.xml"
        idex_definition = xtcedef.XtcePacketDefinition(xtce_document=idex_xtce)
        idex_parser = parser.PacketParser(idex_definition)

        idex_binary_data = bitstring.ConstBitStream(filename=idex_packet_file)
        idex_packet_generator = idex_parser.generator(idex_binary_data)

        self.epochs = []
        self.data = {}
        
        evtnum = 0
        for pkt in idex_packet_generator:
            if 'IDX__SCI0TYPE' in pkt.data:
                scitype = pkt.data['IDX__SCI0TYPE'].raw_value
                if scitype == 1:
                    evtnum += 1
                    self.epochs.append(pkt.data['SHCOARSE'].derived_value + 20*(10**(-6))*pkt.data['SHFINE'].derived_value)  # Use this as the CDF epoch
                    self._log_packet_info(evtnum, pkt) # These are for our logs, don't use

                if scitype in [2, 4, 8, 16, 32, 64]:
                    if scitype not in self.data:
                        self.data.update({scitype : {}})
                    if evtnum not in self.data[scitype]:
                        self.data[scitype][evtnum] = pkt.data['IDX__SCI0RAW'].raw_value
                    else:
                        self.data[scitype][evtnum] += pkt.data['IDX__SCI0RAW'].raw_value

        # Parse the waveforms according to the scitype present (high gain and low gain channels encode waveform data differently).
        self.scitype_to_names = {2: "TOF_High", 4: "TOF_Low", 8: "TOF_Mid", 16: "Target_Low",
                                32: "Target_High", 64: "Ion_Grid"}
        datastore = {}
        self.time_low_sr = []
        self.time_high_sr = []
        for scitype in self.data:
            datastore[self.scitype_to_names[scitype]] = []
            for evt in self.data[scitype]:
                datastore[self.scitype_to_names[scitype]].append(self._parse_waveform_data(self.data[scitype][evt], scitype))
                if self.scitype_to_names[scitype] == 'Target_Low':
                    self.time_low_sr.append(np.linspace(0, len(datastore['Target_Low'][0]), len(datastore['Target_Low'][0])))
                if self.scitype_to_names[scitype] == 'TOF_Low':
                    self.time_high_sr.append(np.linspace(0, len(datastore['TOF_Low'][0]), len(datastore['TOF_Low'][0])))

        self.data = datastore
        self.numevents = evtnum
    
    def _log_packet_info(self, evtnum, pkt):
        print(f"^*****Event header {evtnum}******^")
        # Extract the 17-22-bit integer (usually 8)
        self.lspretrigblocks = (pkt.data['IDX__TXHDRBLOCKS'].derived_value >> 16) &  0b1111
        # Extract the next 4-bit integer (usually 8)
        self.lsposttrigblocks = (pkt.data['IDX__TXHDRBLOCKS'].derived_value >> 12) & 0b1111
        # Extract the next 6 bits integer (usually 32)
        self.hspretrigblocks = (pkt.data['IDX__TXHDRBLOCKS'].derived_value >> 6) & 0b111111
        # Extract the first 6 bits (usually 32)
        self.hsposttrigblocks = (pkt.data['IDX__TXHDRBLOCKS'].derived_value) & 0b111111
        print("HS pre trig sampling blocks: ", self.hspretrigblocks)
        print("LS pre trig sampling blocks: ", self.lspretrigblocks)
        print("HS post trig sampling blocks: ", self.hsposttrigblocks)
        print("LS post trig sampling blocks: ", self.lsposttrigblocks)
        self.TOFdelay = pkt.data['IDX__TXHDRSAMPDELAY'].raw_value >> 2  # First two bits are padding
        mask = 0b1111111111
        self.lgdelay = (self.TOFdelay) & mask
        self.mgdelay = (self.TOFdelay >> 10) & mask
        self.hgdelay = (self.TOFdelay >> 20) & mask
        print(f"High gain delay = {self.hgdelay} samples.")
        print(f"Mid gain delay = {self.mgdelay} samples.")
        print(f"Low gain delay = {self.lgdelay} samples.")
        if(pkt.data['IDX__TXHDRLSTRIGMODE'].derived_value!=0):  # If this was a LS (Target Low Gain) trigger
            self.Triggerorigin = 'LS' 
            print("Low sampling trigger mode enabled.")
        print("Packet trigger mode = ", pkt.data['IDX__TXHDRLGTRIGMODE'].derived_value, pkt.data['IDX__TXHDRMGTRIGMODE'].derived_value, pkt.data['IDX__TXHDRHGTRIGMODE'].derived_value)
        if(pkt.data['IDX__TXHDRLGTRIGMODE'].derived_value!=0):
            print("Low gain TOF trigger mode enabled.")
            self.Triggerorigin = 'LG'
        if(pkt.data['IDX__TXHDRMGTRIGMODE'].derived_value!=0):
            print("Mid gain TOF trigger mode enabled.")
            self.Triggerorigin = 'MG'
        if(pkt.data['IDX__TXHDRHGTRIGMODE'].derived_value!=0):
            print("High gain trigger mode enabled.")
            self.Triggerorigin = 'HG'
        print(f"AID = {pkt.data['IDX__SCI0AID'].derived_value}")  # Instrument event number
        print(f"Event number = {pkt.data['IDX__SCI0EVTNUM'].raw_value}")  # Event number out of how many events constitute the file
        print(f"Rice compression enabled = {bool(pkt.data['IDX__SCI0COMP'].raw_value)}")   
        compressed = bool(pkt.data['IDX__SCI0COMP'].raw_value)  # If we need to decompress the data
        print(f"Timestamp = {self.epochs[evtnum-1]} seconds since epoch (Midnight January 1st, 2012)")

    def _parse_hs_waveform(self, waveform_raw: str):
        """Parse a binary string representing a high gain waveform"""
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read('pad:2')  # skip 2
            ints += w.readlist(['uint:10']*3)
        return ints

    def _parse_ls_waveform(self, waveform_raw: str):
        """Parse a binary string representing a low gain waveform"""
        w = bitstring.ConstBitStream(bin=waveform_raw)
        ints = []
        while w.pos < len(w):
            w.read('pad:8')  # skip 2
            ints += w.readlist(['uint:12']*2)
        return ints
  
    def _parse_waveform_data(self, waveform: str, scitype: int):
        """Parse the binary string that represents a waveform"""
        print(f'Parsing waveform for scitype={scitype}')
        if scitype in (2, 4, 8):
            return self._parse_hs_waveform(waveform)
        else:
            return self._parse_ls_waveform(waveform)