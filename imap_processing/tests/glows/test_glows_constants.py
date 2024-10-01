import pytest

from imap_processing.glows.utils.constants import TimeTuple


@pytest.mark.parametrize(
    "time,expected_seconds",
    [
        (TimeTuple(54232214, 1997263), 54232214.9986315),
        (TimeTuple(100, 0), 100),
        (TimeTuple(100, 200), 100.0001),
    ],
)
def test_to_seconds(time, expected_seconds):
    assert abs(time.to_seconds() - expected_seconds) < 0.00001
