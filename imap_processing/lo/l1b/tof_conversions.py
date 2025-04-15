"""Lo TOF EU Conversions."""

from collections import namedtuple

tof_conv = namedtuple("tof_conv", ["C0", "C1"])
# TOF conversion coefficients from Lo's TOF Conversion_annotated.docx
# TODO: Ask Lo to put these in the algorithm document for better reference
TOF0_CONV = tof_conv(C0=5.52524e-01, C1=1.68374e-01)
TOF1_CONV = tof_conv(C0=-7.20181e-01, C1=1.65124e-01)
TOF2_CONV = tof_conv(C0=3.74422e-01, C1=1.66409e-01)
TOF3_CONV = tof_conv(C0=4.41970e-01, C1=1.72024e-01)
