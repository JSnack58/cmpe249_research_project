import io

from plate_cloak.progress import Progress


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Tty(io.StringIO):
    def isatty(self):
        return True


def test_log_file_gets_a_line_every_interval_and_at_the_end():
    clock, out = FakeClock(), io.StringIO()
    p = Progress(100, "attack", every=10, stream=out, clock=clock)
    for _ in range(100):
        clock.t += 0.5  # 2 images per second -> a line every 20 images, plus the last
        p.update()
    lines = out.getvalue().splitlines()
    assert len(lines) == 5 and "\r" not in out.getvalue()
    assert lines[0].startswith("attack    [####----------------]  20/100  20%  2.0 img/s")
    assert "left 0:40" in lines[0]
    assert lines[-1].startswith("attack    [####################] 100/100 100%")
    assert lines[-1].endswith("elapsed 0:50  left 0:00")


def test_terminal_redraws_one_line_in_place():
    clock, out = FakeClock(), Tty()
    p = Progress(3, "evaluate", stream=out, clock=clock)
    for _ in range(3):
        clock.t += 1
        p.update()
    text = out.getvalue()
    assert text.count("\r") == 3 and text.count("\n") == 1 and text.endswith("\n")


def test_hours_and_unknown_time_left():
    clock, out = FakeClock(), io.StringIO()
    p = Progress(10, "attack", every=0, stream=out, clock=clock)
    assert p.line(0.0).endswith("left ?")  # no speed yet
    clock.t = 3700
    p.update()
    assert "elapsed 1:01:40" in out.getvalue()
