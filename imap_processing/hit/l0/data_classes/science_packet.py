"""L0 HIT Science Packet data class."""

from dataclasses import InitVar, dataclass

import numpy as np

from imap_processing.hit.l0.utils.hit_base import HITBase

# TODO: add methods to the SciencePacket data class to decom attributes with binary data
# TODO: add __post_init__ method to SciencePacket data class to handle InitVar
#       attributes


@dataclass
class SciencePacket(HITBase):
    """
    L0 HIT Science Package data.

    This data class handles the decommutation of the HIT Science Packet
    data.

    Attributes
    ----------
    SHCOARSE: int
        Spacecraft time
    HDR_UNIT_NUM: int
        Unit ID (e.g. EM)
    HDR_FRAME_VERSION: int
        Frame version number
    HDR_STATUS_BITS: int
        Status bits
    HDR_MINUTE_CNT: int
        Minute counter (minute mod 10 -> subcom for sectored rates)
    LIVE_TIME: int
        Livetime count (270=100%)
    NUM_TRIG: int
        Number of triggers
    NUM_REJECT: int
        Number of rejected events
    NUM_ACC_W_PHA: int
        Number of accepted events with PHA data
    NUM_ACC_NO_PHA: int
        Number of events without PHA data
    NUM_HAZ_TRIG: int
        Number of triggers with hazard flag
    NUM_HAZ_REJECT: int
        Number of rejected events with hazard flag
    NUM_HAZ_ACC_W_PHA: int
        Number of accepted hazard events with PHA data
    NUM_HAZ_ACC_NO_PHA: int
        Number of hazard events without PHA data
    SNGRATES_HG: np.ndarray (int, (64,1))
        Counts since last science frame for PHA (hi gain) formatted as an array
    SNGRATES_LG: np.ndarray (int, (64,1))
        Counts since last science frame for PHA (low gain) formatted as an array
    NREAD: int
        Events read from event fifo
    NHAZARD: int
        Events tagged with hazard flag
    NADSTIM: int
        adc-stim events
    NODD: int
        Odd events
    NODDFIX: int
        Odd events that were fixed in sw
    NMULTI: int
        Events with multiple hits in a single detector (may be crosstalk)
    NMULTIFIX: int
        Multi events that were fixed in sw
    NBADTRAJ: int
        Bad trajectory
    NL2: int
        Events sorted into L12 event category
    NL3: int
        Events sorted into L123 event category
    NL4: int
        Events sorted into L1423 event category
    NPEN: int
        Events sorted into PEN (penetrating) event category
    NFORMAT: int
        Nothing currently goes in this slot
    NASIDE: int
        A-side events
    NBSIDE: int
        B-side events
    NERROR: int
        Events that caused a processing error - should never happen
    NBADTAGS: int
        Events with inconsistent tags vs pulse heights
    COINRATES: np.ndarray (int, (26,1))
        Coincidence rates for all detectors formatted into an array
    BUFRATES: np.ndarray (int, (31,1))
        Priority Buffer: ADC cal events formatted into an array
    L2FGRATES: np.ndarray (int, (132,1))
        R2 foreground rates formatted into an array
    L2BGRATES: np.ndarray (int, (12,1))
        R2 background rates formatted into an array
    L3FGRATES: np.ndarray (int, (167,1))
        R3 foreground rates formatted into an array
    L3BGRATES: np.ndarray (int, (12,1))
        R3 background rates formatted into an array
    PENFGRATES: np.ndarray (int, (15,1))
        R4 foreground rates formatted into an array
    PENBGRATES: np.ndarray (int, (15,1))
        R4 foreground rates formatted into an array
    IALIRTRATES: np.ndarray (int, (20,1))
        I-ALiRT rates formatted into an array
    SECTORATES: np.ndarray (int, (120,1))
        R4 background rates formatted into an array
    L4FGRATES: np.ndarray (int, (48,1))
        L4 ions foreground rates formatted into an array
    L4BGRATES: np.ndarray (int, (24,1))
        L4 ions background rates formatted into an array
    PHA_RECORDS: np.ndarray (int, (917,1))
        Event PHA records, array of 4-byte fields, formatted into an array
    SNGRATES_RAW: InitVar[str]
        Raw binary for PHA high gain and low gain for all detectors
    COINRATES_RAW: InitVar[str]
        Raw binary for coincidence rates for all detectors
    BUFRATES_RAW: InitVar[str]
        Raw binary for ADC calibration events
    L2FGRATES_RAW: InitVar[str]
        Raw binary for R2 foreground rates
    L2BGRATES_RAW: InitVar[str]
        Raw binary for R2 background rates
    L3FGRATES_RAW: InitVar[str]
        Raw binary for R3 foreground rates
    L3BGRATES_RAW: InitVar[str]
        Raw binary for R3 background rates
    PENFGRATES_RAW: InitVar[str]
        Raw binary for R4 foreground rates
    PENBGRATES_RAW: InitVar[str]
        Raw binary for R4 background rates
    IALIRTRATES_RAW: InitVar[str]
        Raw binary for I-ALiRT rates
    SECTORATES_RAW: InitVar[str]
        Raw binary for sector rates
    L4FGRATES_RAW: InitVar[str]
        Raw binary for L4 Ions foreground rates
    L4BGRATES_RAW: InitVar[str]
        Raw binary for L4 Ions background rates
    PHA_RECORDS_RAW: InitVar[str]
        Raw binary for event PHA records
    """

    SHCOARSE: int
    HDR_UNIT_NUM: int
    HDR_FRAME_VERSION: int
    HDR_STATUS_BITS: int
    HDR_MINUTE_CNT: int
    LIVE_TIME: int
    NUM_TRIG: int
    NUM_REJECT: int
    NUM_ACC_W_PHA: int
    NUM_ACC_NO_PHA: int
    NUM_HAZ_TRIG: int
    NUM_HAZ_REJECT: int
    NUM_HAZ_ACC_W_PHA: int
    NUM_HAZ_ACC_NO_PHA: int
    SNGRATES_HG: np.ndarray
    SNGRATES_LG: np.ndarray
    NREAD: int
    NHAZARD: int
    NADSTIM: int
    NODD: int
    NODDFIX: int
    NMULTI: int
    NMULTIFIX: int
    NBADTRAJ: int
    NL2: int
    NL3: int
    NL4: int
    NPEN: int
    NFORMAT: int
    NASIDE: int
    NBSIDE: int
    NERROR: int
    NBADTAGS: int
    COINRATES: np.ndarray
    BUFRATES: np.ndarray
    L2FGRATES: np.ndarray
    L2BGRATES: np.ndarray
    L3FGRATES: np.ndarray
    L3BGRATES: np.ndarray
    PENFGRATES: np.ndarray
    PENBGRATES: np.ndarray
    IALIRTRATES: np.ndarray
    SECTORATES: np.ndarray
    L4FGRATES: np.ndarray
    L4BGRATES: np.ndarray
    PHA_RECORDS: np.ndarray
    SNGRATES_RAW: InitVar[str]
    COINRATES_RAW: InitVar[str]
    BUFRATES_RAW: InitVar[str]
    L2FGRATES_RAW: InitVar[str]
    L2BGRATES_RAW: InitVar[str]
    L3FGRATES_RAW: InitVar[str]
    L3BGRATES_RAW: InitVar[str]
    PENFGRATES_RAW: InitVar[str]
    PENBGRATES_RAW: InitVar[str]
    IALIRTRATES_RAW: InitVar[str]
    SECTORATES_RAW: InitVar[str]
    L4FGRATES_RAW: InitVar[str]
    L4BGRATES_RAW: InitVar[str]
    PHA_RECORDS_RAW: InitVar[str]


@dataclass
class SectorRates:
    """
    L0 PHA Record data.

    A data class for PHA Record data which will be stored
    in the SciencePacket PHA_RECORDS attribute.

    Attributes
    ----------
        TBD - talk to Eric for descriptions
    """

    RATE_TYPE: int
    DATA: int


@dataclass
class PHARecord:
    """
    L0 Sector Rates data.

    A data class for Sector Rate data which will be stored
    in the SciencePacket SECTORATES attribute.

    Attributes
    ----------
        TBD - talk to Eric for descriptions
    """

    particle_id: int
    priority_buffer_num: int
    stim_tag: int
    haz_tag: int
    time_tag: int
    a_b_side: int
    has_unread_adc: bool
    extended_header_flag: int
    culling_flag: int
    detector_flags: int
    de_index: int
    ep_index: int
    stim_block: int
    dac_value: int
    pha_number: int
    stim_step: int
    stim_gain: int
    astim: int
    adc_value: int
    detector_address: int
    gain_flag: int
    last_pha_in_event: int
