import os
import subprocess
import sys
from pathlib import Path


def test_data_dir():
    """Test that the data directory is set correctly."""
    command = [
        sys.executable,
        "-c",
        "import imap_processing; print(imap_processing.config['DATA_DIR'])",
    ]
    # Default import should be current working directory.
    proc = subprocess.run(
        command,
        capture_output=True,
        check=True,
        text=True,
    )
    expected = str(Path.cwd() / "imap-data")
    assert proc.stdout.strip() == expected

    # Setting the environment variable should change the data directory
    proc = subprocess.run(
        command,
        env={**os.environ, "IMAP_DATA_DIR": "/test/path"},
        capture_output=True,
        check=True,
        text=True,
    )
    assert proc.stdout.strip() == str(Path("/test/path"))
