"""Atomic publication tolerant of brief Windows/WSL file sharing conflicts."""
from __future__ import annotations

import os
import time
from pathlib import Path


def replace_with_retry(
    source: str | Path,
    destination: str | Path,
    *,
    timeout: float = 5.0,
) -> None:
    """Retry a denied rename briefly without deleting either file.

    A Windows reader can temporarily deny replacement of a file on /mnt/c.
    Retry only PermissionError; permanent denial and other filesystem errors
    remain fatal. The previous destination and prepared source remain intact
    if replacement never succeeds.
    """
    deadline = time.monotonic() + max(0.0, timeout)
    delay = 0.02
    while True:
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.2)
