"""IMAP-HIT CDF Attributes."""

from dataclasses import replace

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    GlobalDataLevelAttrs,
    GlobalInstrumentAttrs,
    ScienceAttrs,
)
from imap_processing.hit import __version__

descriptor = "HIT>IMAP High-energy Ion Telescope"

hit_description_text = (
    "The High-energy Ion Telescope (HIT) measures the elemental composition, "
    "energy spectra, angle distributions, and arrival times of high-energy ions. "
    "HIT delivers full-sky coverage from a wide instrument field-of-view (FOV) to "
    "enable a high resolution of ion measurements, such as observing shock-accelerated "
    "ions, determining the origin of the solar energetic particles (SEPs) spectra, and "
    "resolving particle transport in the heliosphere."
)

# Define housekeeping instrument attributes
hit_base = GlobalInstrumentAttrs(
    version=__version__,
    descriptor=descriptor,
    text=hit_description_text,
    instrument_type="Particles (space)",
)

# Define housekeeping data level attributes
hit_hk_l1a_attrs = GlobalDataLevelAttrs(
    data_type="L1A_HK>Level-1A Housekeeping",
    logical_source="imap_hit_l1a_hk",
    logical_source_desc="IMAP Mission HIT Instrument Level-1A Data",
    instrument_base=hit_base,
)

hit_hk_l1b_attrs = GlobalDataLevelAttrs(
    data_type="L1B_HK>Level-1B Housekeeping",
    logical_source="imap_hit_l1b_hk",
    logical_source_desc="IMAP Mission HIT Instrument Level-1B Data",
    instrument_base=hit_base,
)

# Define housekeeping base attributes
hit_hk_base_attrs = ScienceAttrs(
    validmin=0,
    validmax=GlobalConstants.INT_MAXVAL,
    display_type="time_series",
    depend_0="epoch",
    var_type="data",  # Is this data or supporting data?
    variable_purpose="PRIMARY",
)

# TODO: add attributes for science packets
# TODO: the following may change as we look at different tools for handling
#  variation data attribute values


# Dictionary of housekeeping attributes that are common between L1A and L1B
l1a_l1b_hk_attrs = {
    "fsw_version_a": replace(
        hit_hk_base_attrs,
        validmax=3,
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number A",
        label_axis="FSW A",
        format="I1",
    ),
    "fsw_version_b": replace(
        hit_hk_base_attrs,
        validmax=15,
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number B",
        label_axis="FSW B",
        format="I2",
    ),
    "fsw_version_c": replace(
        hit_hk_base_attrs,
        validmax=63,
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number C",
        label_axis="FSW C",
        format="I2",
    ),
    "num_good_cmds": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Number of Good Commands",
        fieldname="Number of Good Commands",
        label_axis="Good cmds",
        format="I3",
    ),
    "last_good_cmd": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Last Good Command",
        fieldname="Last Good Command",
        label_axis="Last good command",
        format="I3",
    ),
    "last_good_seq_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Last Good Sequence Number",
        fieldname="Last Good Sequence Number",
        label_axis="Last good sequence number",
        format="I3",
    ),
    "num_bad_cmds": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Number of Bad Commands",
        fieldname="Number of Bad Commands",
        label_axis="Number of bad commands",
        format="I3",
    ),
    "last_bad_cmd": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Last Bad Command",
        fieldname="Last Bad Command",
        label_axis="Last bad command",
        format="I3",
    ),
    "last_bad_seq_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Last Bad Sequence Number",
        fieldname="Last Bad Sequence Number",
        label_axis="Last bad sequence number",
        format="I3",
    ),
    "fee_running": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - FEE Running (1) or Reset (0)",
        fieldname="FEE Running (1) or Reset (0)",
        label_axis="FEE running",
        format="I1",
    ),
    "mram_disabled": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - MRAM Disabled (1) or Enabled (0)",
        fieldname="MRAM Disabled (1) or Enabled (0)",
        label_axis="MRAM disabled",
        format="I1",
    ),
    "enable_50khz": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - 50kHz Enabled (1) or Disabled (0)",
        fieldname="50kHz Enabled (1) or Disabled (0)",
        label_axis="Enable 50kHz",
        format="I1",
    ),
    "enable_hvps": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - HVPS Enabled (1) or Disabled (0)",
        fieldname="HVPS Enabled (1) or Disabled (0)",
        label_axis="Enable HVPS",
        format="I1",
    ),
    "table_status": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - Table Status OK (1) or Error (0)",
        fieldname="Table Status OK (1) or Error (0)",
        label_axis="Table status",
        format="I1",
    ),
    "heater_control": replace(
        hit_hk_base_attrs,
        validmax=2,
        catdesc="State - Heater Control (0=None, 1=Pri, 2=Sec)",
        fieldname="Heater Control (0=None, 1=Pri, 2=Sec)",
        label_axis="Heater control",
        format="I1",
    ),
    "adc_mode": replace(
        hit_hk_base_attrs,
        validmax=3,
        catdesc="State - ADC Mode (0=quiet, 1=normal, 2=adcstim, 3=adcThreshold",
        fieldname="ADC Mode (0=quiet, 1=normal, 2=adcstim, 3=adcThreshold)",
        label_axis="ADC mode",
        format="I1",
    ),
    "mode": replace(
        hit_hk_base_attrs,
        validmax=3,
        catdesc="State - Mode (0=Boot, 1=Maintenance, 2=Standby, 3=Science)",
        fieldname="Mode (0=Boot, 1=Maintenance, 2=Standby, 3=Science)",
        label_axis="Mode",
        format="I1",
    ),
    "dynamic_threshold_level": replace(
        hit_hk_base_attrs,
        validmax=3,
        catdesc="Dynamic Threshold Level (0-3)",
        fieldname="Dynamic Threshold Level (0-3)",
        label_axis="Dynamic threshold level",
        format="I1",
    ),
    "num_evnt_last_hk": replace(
        hit_hk_base_attrs,
        validmax=262143,
        catdesc="Number of Events Since Last HK Update",
        fieldname="Number of Events Since Last HK Update",
        label_axis="Number of events since last HK update",
        format="I6",
    ),
    "num_errors": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="Number of Errors",
        fieldname="Number of Errors",
        label_axis="Number of errors",
        format="I3",
    ),
    "last_error_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        catdesc="State - Last Error Number",
        fieldname="Last Error Number",
        label_axis="Last error number",
        format="I3",
    ),
    "code_checksum": replace(
        hit_hk_base_attrs,
        validmax=65535,
        catdesc="Code Checksum",
        fieldname="Code Checksum",
        label_axis="Code checksum",
        format="I5",
    ),
    "spin_period_short": replace(
        hit_hk_base_attrs,
        validmax=65535,
        catdesc="Spin Period Short at T=0",
        fieldname="Spin Period Short at T=0",
        label_axis="Spin period short at T=0",
        format="I5",
    ),
    "spin_period_long": replace(
        hit_hk_base_attrs,
        validmax=65535,
        catdesc="Spin Period Long at T=0",
        fieldname="Spin Period Long at T=0",
        label_axis="Spin period long at T=0",
        format="I5",
    ),
    "leak_current": replace(
        hit_hk_base_attrs,
        catdesc="Leakage Current [V]",
        fieldname="Leakage Current [V]",
        label_axis="Leakage current V",
        format="I19",
    ),
    "phasic_status": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - Phasic Status",
        fieldname="Phasic Status",
        label_axis="Phasic status",
        format="I1",
    ),
    "active_heater": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - Active Heater",
        fieldname="Active Heater",
        label_axis="Active heater",
        format="I1",
    ),
    "heater_on": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - Heater On/Off",
        fieldname="Heater On/Off",
        label_axis="Heater On/Off",
        format="I1",
    ),
    "test_pulser_on": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - Test Pulser On/Off",
        fieldname="Test Pulser On/Off",
        label_axis="Test pulser on/off",
        format="I1",
    ),
    "dac0_enable": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - DAC 0 Enable",
        fieldname="DAC 0 Enable",
        label_axis="DAC 0 enable",
        format="I1",
    ),
    "dac1_enable": replace(
        hit_hk_base_attrs,
        validmax=1,
        catdesc="State - DAC 1 Enable",
        fieldname="DAC 1 Enable",
        label_axis="DAC 1 enable",
        format="I1",
    ),
}

# Dictionary of housekeeping attributes specific to L1A
l1a_hk_attrs_subset = {
    "preamp_l234a": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Preamp L234A",
        fieldname="Preamp L234A",
        label_axis="Preamp L234A",
        format="I4",
    ),
    "preamp_l1a": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Preamp L1A",
        fieldname="Preamp L1A",
        label_axis="Preamp L1A",
        format="I4",
    ),
    "preamp_l1b": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Preamp L1B",
        fieldname="Preamp L1B",
        label_axis="Preamp L1B",
        format="I4",
    ),
    "preamp_l234b": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Preamp L234B",
        fieldname="Preamp L234B",
        label_axis="Preamp L234B",
        format="I4",
    ),
    "temp0": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="FEE LDO Regulator Mounted on the Board Next to the Low-dropout "
        "Regulator",
        fieldname="FEE LDO Regulator",
        label_axis="Temp0 - FEE LDO regulator",
        format="I4",
    ),
    "temp1": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Primary Heater Mounted on the Board Next to the Primary Heater "
        "Circuit",
        fieldname="Primary Heater",
        label_axis="Temp1 - primary heater",
        format="I4",
    ),
    "temp2": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="FEE FPGA Mounted on the Board Next to the FPGA",
        fieldname="FEE FPGA",
        label_axis="Temp2 - FEE FPGA",
        format="I4",
    ),
    "temp3": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Secondary Heater",
        fieldname="Secondary Heater",
        label_axis="Temp3 - secondary heater",
        format="I4",
    ),
    "analog_temp": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Chassis Temp Mounted on the Analog Board Close to Thermostats, "
        "Heaters, and Chassis",
        fieldname="Analog Temp",
        label_axis="Analog temp",
        format="I4",
    ),
    "hvps_temp": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Board Temp Mounted Inside Faraday Cage in Middle of Board Near "
        "Connector Side",
        fieldname="Board Temp",
        label_axis="HVPS temp",
        format="I4",
    ),
    "idpu_temp": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="LDO Temp Mounted on Top of the Low-dropout Regulator",
        fieldname="LDO Temp",
        label_axis="IDPU temp",
        format="I4",
    ),
    "lvps_temp": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Board Temp Mounted in the Middle of Board on Opposite Side of "
        "Hottest Component",
        fieldname="Board Temp",
        label_axis="LVPS temp",
        format="I4",
    ),
    "ebox_3d4vd": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="3.4VD Ebox (Digital)",
        fieldname="3.4VD Ebox (Digital)",
        label_axis="3.4VD Ebox (Digital)",
        format="I4",
    ),
    "ebox_5d1vd": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="5.1VD Ebox (Digital)",
        fieldname="5.1VD Ebox (Digital)",
        label_axis="5.1VD Ebox",
        format="I4",
    ),
    "ebox_p12va": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="+12VA Ebox (Analog)",
        fieldname="+12VA Ebox (Analog)",
        label_axis="+12VA Ebox",
        format="I4",
    ),
    "ebox_m12va": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="-12VA Ebox (Analog)",
        fieldname="-12VA Ebox (Analog)",
        label_axis="-12VA Ebox",
        format="I4",
    ),
    "ebox_p5d7va": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="+5.7VA Ebox (Analog)",
        fieldname="+5.7VA Ebox (Analog)",
        label_axis="+5.7VA Ebox",
        format="I4",
    ),
    "ebox_m5d7va": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="-5.7VA Ebox (Analog)",
        fieldname="-5.7VA Ebox (Analog)",
        label_axis="-5.7VA Ebox",
        format="I4",
    ),
    "ref_p5v": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="+5V ref",
        fieldname="+5V ref",
        label_axis="+5V ref",
        format="I4",
    ),
    "l1ab_bias": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="L1A/B Bias",
        fieldname="L1A/B Bias",
        label_axis="L1A/B Bias",
        format="I4",
    ),
    "l2ab_bias": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="L2A/B Bias",
        fieldname="L2A/B Bias",
        label_axis="L2A/B Bias",
        format="I4",
    ),
    "l34a_bias": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="L3/4A Bias",
        fieldname="L3/4A Bias",
        label_axis="L3/4A Bias",
        format="I4",
    ),
    "l34b_bias": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="L3/4B Bias",
        fieldname="L3/4B Bias",
        label_axis="L3/4B Bias",
        format="I4",
    ),
    "ebox_p2d0vd": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="+2.0VD Ebox (Digital)",
        fieldname="+2.0VD Ebox (Digital)",
        label_axis="+2.0VD Ebox",
        format="I4",
    ),
}

# Dictionary of housekeeping attributes specific to L1B
# TODO Update data formats. Should be float values. Need more info from instrument team
l1b_hk_attrs_subset = {
    "preamp_l234a": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L234A Voltage",
        fieldname="Preamp L234A Voltage",
        label_axis="Preamp L234A V",
        units="V",
        format="I4",
    ),
    "preamp_l1a": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L1A Voltage",
        fieldname="Preamp L1A Voltage",
        label_axis="Preamp L1A V",
        units="V",
        format="I4",
    ),
    "preamp_l1b": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L1B Voltage",
        fieldname="Preamp L1B Voltage",
        label_axis="Preamp L1B V",
        units="V",
        format="I4",
    ),
    "preamp_l234b": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L234B Voltage",
        fieldname="Preamp L234B Voltage",
        label_axis="Preamp L234B V",
        units="V",
        format="I2",
    ),
    "temp0": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=57,
        catdesc="FEE LDO Regulator Mounted on the Board Next to the Low-dropout "
        "Regulator",
        fieldname="FEE LDO Regulator",
        label_axis="Temp0 - FEE LDO regulator",
        units="C",
        format="I2",
    ),
    "temp1": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Primary Heater Mounted on the Board Next to the Primary Heater "
        "Circuit",
        fieldname="Primary Heater",
        label_axis="Temp1 - primary heater",
        units="C",
        format="I2",
    ),
    "temp2": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="FEE FPGA Mounted on the Board Next to the FPGA",
        fieldname="FEE FPGA",
        label_axis="Temp2 - FEE FPGA",
        units="C",
        format="I2",
    ),
    "temp3": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Secondary Heater",
        fieldname="Secondary Heater",
        label_axis="Temp3 - secondary heater",
        units="C",
        format="I2",
    ),
    "analog_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Chassis Temp Mounted on the Analog Board Close to Thermostats, "
        "Heaters, and Chassis",
        fieldname="Analog Temp",
        label_axis="Analog temp",
        units="C",
        format="I2",
    ),
    "hvps_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Board Temp Mounted Inside Faraday Cage in Middle of Board Near "
        "Connector Side",
        fieldname="Board Temp",
        label_axis="HVPS temp",
        units="C",
        format="I2",
    ),
    "idpu_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=65,
        catdesc="LDO Temp Mounted on Top of the Low-dropout Regulator",
        fieldname="LDO Temp",
        label_axis="IDPU temp",
        units="C",
        format="I2",
    ),
    "lvps_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=80,
        catdesc="Board Temp Mounted in the Middle of Board on Opposite Side of "
        "Hottest Component",
        fieldname="Board Temp",
        label_axis="LVPS temp",
        units="C",
        format="I2",
    ),
    "ebox_3d4vd": replace(
        hit_hk_base_attrs,
        validmin=3,  # doc says min 3.1
        validmax=4,  # doc says max 3.6
        catdesc="3.4VD Ebox (Digital)",
        fieldname="3.4VD Ebox (Digital)",
        label_axis="3.4VD Ebox",
        units="V",
        format="F2.1",
    ),
    "ebox_5d1vd": replace(
        hit_hk_base_attrs,
        validmin=4,  # doc says min 4.85
        validmax=6,  # doc says max 5.45
        catdesc="5.1VD Ebox (Digital)",
        fieldname="5.1VD Ebox (Digital)",
        label_axis="5.1VD Ebox",
        units="V",
        format="F3.2",
    ),
    "ebox_p12va": replace(
        hit_hk_base_attrs,
        validmin=11,  # doc says min 11.2
        validmax=14,  # doc says max 13.1
        catdesc="+12VA Ebox (Analog)",
        fieldname="+12VA Ebox (Analog)",
        label_axis="+12VA Ebox",
        units="V",
        format="F3.1",
    ),
    "ebox_m12va": replace(
        hit_hk_base_attrs,
        validmin=-14,  # doc says min -13.1
        validmax=-12,  # doc says max -11.2
        catdesc="-12VA Ebox (Analog)",
        fieldname="-12VA Ebox (Analog)",
        label_axis="-12VA Ebox",
        units="V",
        format="F3.1",
    ),
    "ebox_p5d7va": replace(
        hit_hk_base_attrs,
        validmin=5,  # doc says min 5.3
        validmax=7,  # doc says max 6.3
        catdesc="+5.7VA Ebox (Analog)",
        fieldname="+5.7VA Ebox (Analog)",
        label_axis="+5.7VA Ebox",
        units="V",
        format="F2.1",
    ),
    "ebox_m5d7va": replace(
        hit_hk_base_attrs,
        validmin=-7,  # doc says min -6.4125
        validmax=-5,  # doc says max -5.25
        catdesc="-5.7VA Ebox (Analog)",
        fieldname="-5.7VA Ebox (Analog)",
        label_axis="-5.7VA Ebox",
        units="V",
        format="F5.4",
    ),
    "ref_p5v": replace(
        hit_hk_base_attrs,
        # validmin=,             Need more info from instrument team
        # validmax=,
        catdesc="+5V ref",
        fieldname="+5V ref",
        label_axis="+5V ref",
        units="V",
        format="I4",
    ),
    "l1ab_bias": replace(
        hit_hk_base_attrs,
        validmin=0,
        validmax=15,
        catdesc="L1A/B Bias",
        fieldname="L1A/B Bias",
        label_axis="L1A/B Bias",
        units="V",
        format="I2",
    ),
    "l2ab_bias": replace(
        hit_hk_base_attrs,
        validmin=0,
        validmax=25,
        catdesc="L2A/B Bias",
        fieldname="L2A/B Bias",
        label_axis="L2A/B Bias",
        units="V",
        format="I2",
    ),
    "l34a_bias": replace(
        hit_hk_base_attrs,
        validmin=0,
        validmax=255,
        catdesc="L3/4A Bias",
        fieldname="L3/4A Bias",
        label_axis="L3/4A Bias",
        units="V",
        format="I3",
    ),
    "l34b_bias": replace(
        hit_hk_base_attrs,
        validmin=0,
        validmax=225,
        catdesc="L3/4B Bias",
        fieldname="L3/4B Bias",
        label_axis="L3/4B Bias",
        units="V",
        format="I3",
    ),
    "ebox_p2d0vd": replace(
        hit_hk_base_attrs,
        validmin=1,  # doc says min 1.79
        validmax=3,  # doc says max 2.5
        catdesc="+2.0VD Ebox (Digital)",
        fieldname="+2.0VD Ebox (Digital)",
        label_axis="+2.0VD Ebox",
        units="V",
        format="F3.2",
    ),
}

# Dictionaries of complete L1A and L1B housekeeping attributes
l1a_hk_attrs = l1a_hk_attrs_subset.update(l1a_l1b_hk_attrs)
l1b_hk_attrs = l1b_hk_attrs_subset.update(l1a_l1b_hk_attrs)
