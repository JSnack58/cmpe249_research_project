"""The blind two-alternative eye test (milestone 2a, Task 9)."""

import csv
import json
import threading
import urllib.request

import numpy as np
import pytest
from PIL import Image

from plate_cloak import cli
from plate_cloak.attack.stage import fingerprint
from plate_cloak.config import BASE_CONFIG, load_config, resolve_path
from plate_cloak.data.eval_images import eval_image_ids, load_by_id
from plate_cloak.visibility.eyetest import (
    MAX_CORRECT,
    N_TRIALS,
    Budget,
    ladder_ceiling,
    make_server,
    make_session,
    score,
)

N_IMAGES = 12  # two plates each: 24 candidate plates per budget


def data_overrides(tmp_path):
    root = tmp_path / "data"
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "test").mkdir(parents=True, exist_ok=True)
    ids = [f"img{i:02d}" for i in range(N_IMAGES)]
    for i, image_id in enumerate(ids):
        pixels = np.random.default_rng(i).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / "images" / "test" / f"{image_id}.jpg")
        labels = "0 0.3 0.3 0.3 0.2\n0 0.85 0.85 0.3 0.3\n"  # the second plate is at the edge
        (root / "labels" / "test" / f"{image_id}.txt").write_text(labels)
    eligible = tmp_path / "eligible.txt"
    eligible.write_text("\n".join(ids) + "\n")
    return [
        f"data.root={root}",
        f"data.eligible_list={eligible}",
        f"output.cache_root={tmp_path / 'cache'}",
        "data.imgsz=160",
        "data.min_plate_w=20",
        "data.min_plate_h=10",
    ]


def budget(tmp_path, ladder, rung, cb):
    """A finished cache for a YCbCr budget: +cb/255 blue on every pixel (enough for crops)."""
    cfg = load_config(BASE_CONFIG, [*data_overrides(tmp_path), f"attack.eps.cb={cb}"])
    ids = eval_image_ids(cfg)
    out = resolve_path(cfg.output.cache_root) / fingerprint(cfg, ids)
    out.mkdir(parents=True, exist_ok=True)
    for image_id in ids:
        img = np.array(load_by_id(cfg, image_id).image).astype(int)
        img[..., 2] += cb
        Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(out / f"{image_id}.png")
    (out / "done.json").write_text("{}")
    return Budget(ladder, rung, cfg)


@pytest.fixture
def session(tmp_path):
    budgets = [budget(tmp_path, "cbcr", "cb12", 12), budget(tmp_path, "cbcr", "cb16", 16)]
    out = tmp_path / "session"
    make_session(budgets, out, seed=42, n_trials=20)
    return out


def trials(session):
    return json.loads((session / "trials.json").read_text())["trials"]


def test_pass_rule_constants():
    # Fixed before any trial: 26 of 40 has a one-sided binomial p of about 0.04 under guessing.
    assert (N_TRIALS, MAX_CORRECT) == (40, 25)


def test_session_has_n_trials_per_budget_and_balanced_sides(session):
    t = trials(session)
    assert len(t) == 40
    for rung in ("cb12", "cb16"):
        mine = [x for x in t if x["rung"] == rung]
        assert len(mine) == 20
        assert len({(x["image_id"], x["plate_idx"]) for x in mine}) == 20  # no plate twice
        left = sum(x["attacked"] == "left" for x in mine)
        assert 0.3 * 20 <= left <= 0.7 * 20
    # Budgets are interleaved, so the viewer can't tell which budget a trial tests.
    assert [x["rung"] for x in t[:20]].count("cb12") not in (0, 20)


def test_crops_of_a_trial_have_the_same_size_even_at_the_edge(session):
    for x in trials(session):
        left = Image.open(session / "crops" / f"{x['id']}_left.png")
        right = Image.open(session / "crops" / f"{x['id']}_right.png")
        assert left.size == right.size, x


def test_too_few_plates_is_refused(tmp_path):
    with pytest.raises(ValueError, match="plates"):
        make_session([budget(tmp_path, "cbcr", "cb12", 12)], tmp_path / "s", n_trials=40)


def test_existing_session_is_never_overwritten(session):
    with pytest.raises(FileExistsError):
        make_session([], session)


# --- server ---------------------------------------------------------------------------------


def start(session):
    server = make_server(session, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def post(url, payload):
    req = urllib.request.Request(
        url, json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_page_and_crops_are_served(session):
    server, base = start(session)
    try:
        assert b"<title>" in get(base + "/")
        nxt = json.loads(get(base + "/next"))
        assert get(base + nxt["trial"]["left"])[:8] == b"\x89PNG\r\n\x1a\n"
    finally:
        server.shutdown()


def test_next_never_reveals_the_answer(session):
    server, base = start(session)
    try:
        nxt = json.loads(get(base + "/next"))
        text = json.dumps(nxt)
        assert "attacked" not in text and "rung" not in text and "cb1" not in text
        assert set(nxt["trial"]) == {"id", "left", "right"}
        assert nxt["trial"]["left"] == f"/crop/{nxt['trial']['id']}/left"
    finally:
        server.shutdown()


def test_answers_survive_a_restart_and_duplicates_are_ignored(session):
    server, base = start(session)
    first = [x["id"] for x in trials(session)]
    try:
        for tid in first[:3]:
            post(base + "/answer", {"trial": tid, "choice": "left"})
        post(base + "/answer", {"trial": first[2], "choice": "right"})  # a double click
    finally:
        server.shutdown()
        server.server_close()
    server, base = start(session)  # restarted: picks up where the viewer left off
    try:
        nxt = json.loads(get(base + "/next"))
        assert nxt["trial"]["id"] == first[3] and nxt["answered"] == 3 and nxt["total"] == 40
    finally:
        server.shutdown()
    with (session / "answers.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["trial"] for r in rows] == first[:3] and rows[2]["choice"] == "left"


def test_bad_answers_are_rejected(session):
    server, base = start(session)
    try:
        for payload in [{"trial": "nope", "choice": "left"}, {"trial": "t001", "choice": "up"}]:
            with pytest.raises(urllib.error.HTTPError) as e:
                post(base + "/answer", payload)
            assert e.value.code == 400
    finally:
        server.shutdown()


# --- scoring --------------------------------------------------------------------------------


def answer(session, correct_by_rung):
    """Answer every trial; the first `correct_by_rung[rung]` of each rung correctly."""
    seen = dict.fromkeys(correct_by_rung, 0)
    with (session / "answers.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "choice", "time"])
        for x in trials(session):
            right = seen[x["rung"]] < correct_by_rung[x["rung"]]
            seen[x["rung"]] += 1
            other = "right" if x["attacked"] == "left" else "left"
            w.writerow([x["id"], x["attacked"] if right else other, "t"])


def test_score_counts_correct_answers_per_budget(session):
    answer(session, {"cb12": 10, "cb16": 15})
    s = score([session], n_trials=20, max_correct=12)
    assert s["budgets"]["cbcr/cb12"] == {"correct": 10, "n": 20, "complete": True, "passed": True}
    assert s["budgets"]["cbcr/cb16"]["passed"] is False
    assert s["ladders"]["cbcr"] == {"rungs": ["cb12", "cb16"], "ceiling": "cb12", "extend": None}
    assert json.loads((session / "scores.json").read_text()) == s


def test_unfinished_budget_has_no_verdict(session):
    answer(session, {"cb12": 10, "cb16": 10})
    rows = (session / "answers.csv").read_text().splitlines()
    (session / "answers.csv").write_text("\n".join(rows[:-1]) + "\n")  # one trial unanswered
    s = score([session], n_trials=20, max_correct=12)
    incomplete = [b for b in s["budgets"].values() if not b["complete"]]
    assert len(incomplete) == 1 and incomplete[0]["passed"] is False
    assert s["ladders"]["cbcr"]["ceiling"] is None and s["ladders"]["cbcr"]["extend"] is None


def test_pass_rule_at_the_boundary():
    assert ladder_ceiling([True]) == (0, "up")
    # 25 of 40 passes, 26 fails: checked through the same rule score() uses.
    from plate_cloak.visibility.eyetest import passes

    assert passes(25, 40) and not passes(26, 40) and not passes(20, 39)


def test_ceiling_and_extension():
    assert ladder_ceiling([True, True, False, True]) == (1, None)  # a later pass doesn't count
    assert ladder_ceiling([True, True, True]) == (2, "up")
    assert ladder_ceiling([False, True]) == (None, "down")


def test_cli_make_and_score(tmp_path, monkeypatch):
    b = [budget(tmp_path, "cbcr", "cb12", 12)]
    monkeypatch.setattr(cli, "_eyetest_budgets", lambda sweeps, variants, overrides: b)
    monkeypatch.setattr("plate_cloak.visibility.eyetest.N_TRIALS", 20)
    root = tmp_path / "eyetest"
    args = ["--root", str(root)]
    assert cli.main(["eyetest", "make", "--session", "s1", "--sweeps", "x.yaml", *args]) == 0
    assert len(trials(root / "s1")) == 20
    assert cli.main(["eyetest", "score", "--session", "s1", *args]) == 0


def test_an_idle_browser_connection_does_not_block_the_page(session):
    # Browsers open spare connections ahead of time and may never send on them; a server that
    # handles one connection at a time would hang until that socket closes.
    import socket

    server, base = start(session)
    idle = socket.create_connection(server.server_address[:2])
    try:
        with urllib.request.urlopen(base + "/next", timeout=3) as r:
            assert json.loads(r.read())["total"] == 40
    finally:
        idle.close()
        server.shutdown()
