"""A small progress bar for long stages, readable both live and in a log file.

In a terminal it redraws one line in place, cut to the terminal's width; when output goes to a
file (a background job), it writes a new line at most every `every` seconds, so `tail -1 <log>`
always shows the latest state:

    attack    [##########----------]  865/1731  50%  1.6 img/s  elapsed 9:01  left 9:02
"""

from __future__ import annotations

import shutil
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
        columns=lambda: shutil.get_terminal_size().columns,
    ):
        self.total, self.label, self.unit, self.every = total, label, unit, every
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.columns = columns  # asked on every redraw, so resizing the window works
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
            line = self.line(now)
            if self.live:
                # A line wider than the terminal wraps, and \r would then redraw only its last
                # row; cut it to fit and clear whatever an earlier, longer line left behind.
                line = "\r" + _fit(line, self.columns() - 1) + "\x1b[K"
            self.stream.write(line + ("\n" if finished or not self.live else ""))
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


def _fit(line: str, width: int) -> str:
    """The line cut to `width`, dropping whole fields (they are separated by two spaces)."""
    if len(line) <= width:
        return line
    cut = line.rfind("  ", 0, width + 1)
    return line[:cut] if cut > 0 else line[: max(width, 0)]


def _clock(seconds: float | None) -> str:
    """h:mm:ss or m:ss; '?' before the speed is known."""
    if seconds is None:
        return "?"
    s = round(seconds)
    h, m = divmod(s // 60, 60)
    return f"{h}:{m:02d}:{s % 60:02d}" if h else f"{m}:{s % 60:02d}"
