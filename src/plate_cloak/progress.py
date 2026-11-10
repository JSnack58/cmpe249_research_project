"""A small progress bar for long stages, readable both live and in a log file.

In a terminal it redraws one line in place; when output goes to a file (a background job), it
writes a new line at most every `every` seconds, so `tail -1 <log>` always shows the latest state:

    attack    [##########----------]  865/1731  50%  1.6 img/s  elapsed 9:01  left 9:02
"""

from __future__ import annotations

import sys
import time
from typing import TextIO

WIDTH = 20  # characters in the bar


class Progress:
    def __init__(
        self,
        total: int,
        label: str,
        unit: str = "img",
        every: float = 10.0,
        stream: TextIO | None = None,
        clock=time.monotonic,
    ):
        self.total, self.label, self.unit, self.every = total, label, unit, every
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.live = self.stream.isatty()
        self.done = 0
        self.start = self.last = clock()

    def update(self, n: int = 1) -> None:
        """Count n more items; redraw if it's time, and always on the last item."""
        self.done += n
        now = self.clock()
        finished = self.done >= self.total
        if finished or now - self.last >= (0.2 if self.live else self.every):
            self.last = now
            end = "\n" if finished or not self.live else ""
            self.stream.write(("\r" if self.live else "") + self.line(now) + end)
            self.stream.flush()

    def line(self, now: float) -> str:
        elapsed = now - self.start
        frac = self.done / self.total if self.total else 1.0
        filled = round(frac * WIDTH)
        rate = self.done / elapsed if elapsed > 0 else 0.0
        left = (self.total - self.done) / rate if rate > 0 else None
        return (
            f"{self.label:<9} [{'#' * filled}{'-' * (WIDTH - filled)}] "
            f"{self.done:>{len(str(self.total))}}/{self.total} {frac:4.0%}  "
            f"{rate:.1f} {self.unit}/s  elapsed {_clock(elapsed)}  left {_clock(left)}"
        )


def _clock(seconds: float | None) -> str:
    """h:mm:ss or m:ss; '?' before the speed is known."""
    if seconds is None:
        return "?"
    s = round(seconds)
    h, m = divmod(s // 60, 60)
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m}:{s % 60:02d}"
