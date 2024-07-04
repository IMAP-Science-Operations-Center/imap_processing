"""IMAP-HIT CDF Attributes."""

from dataclasses import replace

from imap_processing.cdf.defaults import GlobalConstants
from imap_processing.cdf.global_attrs import (
    AttrBase,
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
    "See https://imap.princeton.edu/instruments/hit for more details. "
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
    var_type="data",
    variable_purpose="PRIMARY",
)

# TODO: add attributes for science packets
# TODO: the following may change as we look at different tools for
#  defining CDF attributes
# TODO: update label_axis with values provided by instrument team.
#  Waiting for info

# Define housekeeping data variable attributes

# Dictionary of housekeeping attributes that are common between
# L1A and L1B (modes, flags, states)
l1a_l1b_hk_attrs = {
    # adc_channels is a dependency for leak_i data variable
    "adc_channels": AttrBase(
        validmin=0,
        validmax=63,
        var_type="metadata",
        display_type="no_plot",
        catdesc="ADC Channel",
        fieldname="ADC Channel",
        label_axis="Channel",
        format="I2",
    ),
    "fsw_version_a": replace(
        hit_hk_base_attrs,
        validmax=3,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number A",
        label_axis="FSW A",
        format="I1",
    ),
    "fsw_version_b": replace(
        hit_hk_base_attrs,
        validmax=15,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number B",
        label_axis="FSW B",
        format="I2",
    ),
    "fsw_version_c": replace(
        hit_hk_base_attrs,
        validmax=63,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Flight Software Version Number (A.B.C bits)",
        fieldname="Flight Software Version Number C",
        label_axis="FSW C",
        format="I2",
    ),
    "num_good_cmds": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Number of Good Commands",
        fieldname="Number of Good Commands",
        label_axis="Counts",
        format="I3",
    ),
    "last_good_cmd": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Last Good Command",
        fieldname="Last Good Command",
        label_axis="Last cmd",
        format="I3",
    ),
    "last_good_seq_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Last Good Sequence Number",
        fieldname="Last Good Sequence Number",
        label_axis="Last num",
        format="I3",
    ),
    "num_bad_cmds": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Number of Bad Commands",
        fieldname="Number of Bad Commands",
        label_axis="Counts",
        format="I3",
    ),
    "last_bad_cmd": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Last Bad Command",
        fieldname="Last Bad Command",
        label_axis="Last cmd",
        format="I3",
    ),
    "last_bad_seq_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Last Bad Sequence Number",
        fieldname="Last Bad Sequence Number",
        label_axis="Last num",
        format="I3",
    ),
    "fee_running": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - FEE Running (1) or Reset (0)",
        fieldname="FEE Running (1) or Reset (0)",
        label_axis="State",
        format="I1",
    ),
    "mram_disabled": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - MRAM Disabled (1) or Enabled (0)",
        fieldname="MRAM Disabled (1) or Enabled (0)",
        label_axis="State",
        format="I1",
    ),
    "enable_50khz": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - 50kHz Enabled (1) or Disabled (0)",
        fieldname="50kHz Enabled (1) or Disabled (0)",
        label_axis="State",
        format="I1",
    ),
    "enable_hvps": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - HVPS Enabled (1) or Disabled (0)",
        fieldname="HVPS Enabled (1) or Disabled (0)",
        label_axis="State",
        format="I1",
    ),
    "table_status": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Table Status OK (1) or Error (0)",
        fieldname="Table Status OK (1) or Error (0)",
        label_axis="Status",
        format="I1",
    ),
    "heater_control": replace(
        hit_hk_base_attrs,
        validmax=2,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Heater Control (0=None, 1=Pri, 2=Sec)",
        fieldname="Heater Control (0=None, 1=Pri, 2=Sec)",
        label_axis="State",
        format="I1",
    ),
    "adc_mode": replace(
        hit_hk_base_attrs,
        validmax=3,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - ADC Mode (0=quiet, 1=normal, 2=adcstim, 3=adcThreshold",
        fieldname="ADC Mode (0=quiet, 1=normal, 2=adcstim, 3=adcThreshold)",
        label_axis="ADC mode",
        format="I1",
    ),
    "mode": replace(
        hit_hk_base_attrs,
        validmax=3,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Mode (0=Boot, 1=Maintenance, 2=Standby, 3=Science)",
        fieldname="Mode (0=Boot, 1=Maintenance, 2=Standby, 3=Science)",
        label_axis="Mode",
        format="I1",
    ),
    "dyn_thresh_lvl": replace(
        hit_hk_base_attrs,
        validmax=3,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Dynamic Threshold Level (0-3)",
        fieldname="Dynamic Threshold Level (0-3)",
        label_axis="Level",
        format="I1",
    ),
    "num_evnt_last_hk": replace(
        hit_hk_base_attrs,
        validmax=262143,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Number of Events Since Last HK Update",
        fieldname="Number of Events Since Last HK Update",
        label_axis="Num events",
        format="I6",
    ),
    "num_errors": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Number of Errors",
        fieldname="Number of Errors",
        label_axis="Num errors",
        format="I3",
    ),
    "last_error_num": replace(
        hit_hk_base_attrs,
        validmax=255,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Last Error Number",
        fieldname="Last Error Number",
        label_axis="Error num",
        format="I3",
    ),
    "code_checksum": replace(
        hit_hk_base_attrs,
        validmax=65535,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Code Checksum",
        fieldname="Code Checksum",
        label_axis="Checksum",
        format="I5",
    ),
    "spin_period_short": replace(
        hit_hk_base_attrs,
        validmax=65535,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Spin Period Short at T=0",
        fieldname="Spin Period Short at T=0",
        label_axis="Spin period short",
        format="I5",
    ),
    "spin_period_long": replace(
        hit_hk_base_attrs,
        validmax=65535,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="Spin Period Long at T=0",
        fieldname="Spin Period Long at T=0",
        label_axis="Spin period long",
        format="I5",
    ),
    "leak_i": replace(
        hit_hk_base_attrs,
        var_type="ignore_data",
        display_type="no_plot",
        depend_1="adc_channels",
        catdesc="Leakage Current [I]",
        fieldname="Leakage Current [I]",
        label_axis="Current I",
        labl_ptr="adc_channels",
        format="I19",
    ),
    "phasic_stat": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Phasic Status",
        fieldname="Phasic Status",
        label_axis="Status",
        format="I1",
    ),
    "active_heater": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Active Heater",
        fieldname="Active Heater",
        label_axis="State",
        format="I1",
    ),
    "heater_on": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Heater On/Off",
        fieldname="Heater On/Off",
        label_axis="State",
        format="I1",
    ),
    "test_pulser_on": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - Test Pulser On/Off",
        fieldname="Test Pulser On/Off",
        label_axis="State",
        format="I1",
    ),
    "dac0_enable": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - DAC 0 Enable",
        fieldname="DAC 0 Enable",
        label_axis="State",
        format="I1",
    ),
    "dac1_enable": replace(
        hit_hk_base_attrs,
        validmax=1,
        var_type="ignore_data",
        display_type="no_plot",
        catdesc="State - DAC 1 Enable",
        fieldname="DAC 1 Enable",
        label_axis="State",
        format="I1",
    ),
}

# Dictionary of housekeeping attributes specific to L1A
l1a_hk_attrs = {
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
        label_axis="Temp0",
        format="I4",
    ),
    "temp1": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Primary Heater Mounted on the Board Next to the Primary Heater "
        "Circuit",
        fieldname="Primary Heater",
        label_axis="Temp1",
        format="I4",
    ),
    "temp2": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="FEE FPGA Mounted on the Board Next to the FPGA",
        fieldname="FEE FPGA",
        label_axis="Temp2",
        format="I4",
    ),
    "temp3": replace(
        hit_hk_base_attrs,
        validmax=4095,
        catdesc="Secondary Heater",
        fieldname="Secondary Heater",
        label_axis="Temp3",
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
        label_axis="3.4VD Ebox",
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
l1b_hk_attrs = {
    "preamp_l234a": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L234A Voltage",
        fieldname="Preamp L234A Voltage",
        label_axis="Preamp L234A V",
        units="V",
        format="I4",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "preamp_l1a": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L1A Voltage",
        fieldname="Preamp L1A Voltage",
        label_axis="Preamp L1A V",
        units="V",
        format="I4",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "preamp_l1b": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L1B Voltage",
        fieldname="Preamp L1B Voltage",
        label_axis="Preamp L1B V",
        units="V",
        format="I4",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "preamp_l234b": replace(
        hit_hk_base_attrs,
        # validmax=4095,           # Need this info from instrument team
        catdesc="Preamp L234B Voltage",
        fieldname="Preamp L234B Voltage",
        label_axis="Preamp L234B V",
        units="V",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "temp0": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=57,
        catdesc="FEE LDO Regulator Mounted on the Board Next to the Low-dropout "
        "Regulator",
        fieldname="FEE LDO Regulator",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "temp1": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Primary Heater Mounted on the Board Next to the Primary Heater "
        "Circuit",
        fieldname="Primary Heater",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "temp2": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="FEE FPGA Mounted on the Board Next to the FPGA",
        fieldname="FEE FPGA",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "temp3": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Secondary Heater",
        fieldname="Secondary Heater",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "analog_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Chassis Temp Mounted on the Analog Board Close to Thermostats, "
        "Heaters, and Chassis",
        fieldname="Analog Temp",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "hvps_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=50,
        catdesc="Board Temp Mounted Inside Faraday Cage in Middle of Board Near "
        "Connector Side",
        fieldname="Board Temp",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "idpu_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=65,
        catdesc="LDO Temp Mounted on Top of the Low-dropout Regulator",
        fieldname="LDO Temp",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
    "lvps_temp": replace(
        hit_hk_base_attrs,
        validmin=-25,
        validmax=80,
        catdesc="Board Temp Mounted in the Middle of Board on Opposite Side of "
        "Hottest Component",
        fieldname="Board Temp",
        label_axis="Temp",
        units="C",
        format="I2",
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
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
        fill_val=GlobalConstants.DOUBLE_FILLVAL,
    ),
}

# Dictionaries of complete L1A and L1B housekeeping attributes
l1a_hk_attrs.update(l1a_l1b_hk_attrs)  # type: ignore[arg-type]
l1b_hk_attrs.update(l1a_l1b_hk_attrs)  # type: ignore[arg-type]

# TODO Fix mypy error above.
# Argument 1 to "update" of "MutableMapping" has incompatible type
# "dict[str, AttrBase]"; expected "SupportsKeysAndGetItem[str, ScienceAttrs]"
