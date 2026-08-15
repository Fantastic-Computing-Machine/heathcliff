# ABOUTME: Tests lazy optional imports from the core package
# ABOUTME: Keeps text-only startup free of audio imports

import subprocess
import sys


def test_core_does_not_import_audio_handler():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import core, sys; assert 'core.audio_handler' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
