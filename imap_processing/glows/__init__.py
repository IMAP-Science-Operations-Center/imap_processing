__version__ = "v001"

BAD_TIME_FLAG_NAMES = (
    "is_pps_missing",
    "is_time_status_missing",
    "is_phase_missing",
    "is_spin_period_missing",
    "is_overexposed",
    "is_direct_event_non_monotonic",
    "is_night",
    "is_hv_test_in_progress",
    "is_test_pulse_in_progress",
    "is_memory_error_detected",
    "is_generated_on_ground",
    "is_beyond_daily_statistical_error",
    "is_temperature_std_dev_beyond_threshold",
    "is_hv_voltage_std_dev_beyond_threshold",
    "is_spin_period_std_dev_beyond_threshold",
    "is_pulse_length_std_dev_beyond_threshold",
    "is_spin_period_difference_beyond_threshold",
)

# Quality flag list length. Used in L1B and L2.
FLAG_LENGTH = len(BAD_TIME_FLAG_NAMES)
