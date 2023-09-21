from enum import Enum


class HitAPID(Enum):
    HIT_AUT = 1250  # Autonomy
    HIT_HSKP = 1251  # Housekeeping
    HIT_SCIENCE = 1252  # Science
    HIT_IALRT = 1253  # I-ALiRT
    HIT_MEMDUMP = 1255  # Memory dump
