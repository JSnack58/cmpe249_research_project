"""The blind eye test: can the viewer tell which of two plate crops was attacked?

    python -m plate_cloak eyetest make  --session NAME --sweeps SWEEP.yaml [...] [--variants V ...]
    python -m plate_cloak eyetest serve --session NAME          (then open the printed address)
    python -m plate_cloak eyetest score --session NAME [NAME ...]

Each trial shows the clean and the attacked crop of one plate side by side, in random order, and
the viewer picks the one they think is attacked (two-alternative forced choice). Trials of every
budget are shuffled together, so the viewer never knows which budget a trial tests, and the page
never says whether an answer was right.

The pass rule was fixed before any trial (spec 2026-11-10, section 3): a budget passes if the
viewer picks the attacked crop in at most MAX_CORRECT of N_TRIALS trials. Guessing gets 20 of 40
on average; 26 or more happens by luck with probability of about 0.04.
On each ladder (one sweep file; rungs ordered by budget size) the ceiling is the largest rung
that passes with every smaller rung passing too. If the top rung passes, the ladder should be
extended upward; if the bottom rung fails, downward.

A session folder results/eyetest/<NAME>/ holds
    trials.json   the answer key: every trial's budget, plate and attacked side
    crops/        <trial>_left.png and <trial>_right.png (git-ignored; remade from the seed)
    answers.csv   trial, choice, time: appended on every click, so nothing is lost on a restart
    scores.json   written by `score`
"""

from __future__ import annotations

import csv
import json
import random
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from omegaconf import DictConfig

from plate_cloak.attack.stage import attack_settings, fingerprint
from plate_cloak.config import resolve_path
from plate_cloak.data.eval_images import eval_image_ids, load_by_id
from plate_cloak.progress import Progress
from plate_cloak.visibility.crops import plate_crops

N_TRIALS = 40
MAX_CORRECT = 25
DEFAULT_ROOT = "results/eyetest"
SIDES = ("left", "right")


@dataclass
class Budget:
    ladder: str  # the sweep file's stem, e.g. visibility_cbcr
    rung: str  # the variant, e.g. cb16
    cfg: DictConfig


def make_session(
    budgets: list[Budget], out: Path, seed: int = 42, n_trials: int | None = None
) -> None:
    """Draw the trials for every budget, save their crops and the answer key."""
    n_trials = N_TRIALS if n_trials is None else n_trials
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; pick another --session")
    trials, budget_info = [], []
    for b in budgets:
        ids = eval_image_ids(b.cfg)
        adv_dir = resolve_path(b.cfg.output.cache_root) / fingerprint(b.cfg, ids)
        if not (adv_dir / "done.json").exists():
            raise RuntimeError(f"{b.ladder}/{b.rung}: attack cache {adv_dir} is not finished")
        plates = [
            (image_id, int(idx))
            for image_id in ids
            for idx in (~load_by_id(b.cfg, image_id).ignore).nonzero()[0]
        ]
        if len(plates) < n_trials:
            raise ValueError(f"{b.ladder}/{b.rung}: only {len(plates)} plates, need {n_trials}")
        # Each budget draws its own plates, so the viewer can't learn a plate across budgets.
        rng = random.Random(f"{seed}:{b.ladder}:{b.rung}")
        for image_id, idx in rng.sample(plates, n_trials):
            trials.append(
                {
                    "ladder": b.ladder,
                    "rung": b.rung,
                    "image_id": image_id,
                    "plate_idx": idx,
                    "attacked": rng.choice(SIDES),
                    "_adv_dir": adv_dir,
                    "_cfg": b.cfg,
                }
            )
        eps = attack_settings(b.cfg.attack)["eps"]
        budget_info.append(
            {"ladder": b.ladder, "rung": b.rung, "eps": eps, "size": sum(eps.values())}
        )
    random.Random(f"{seed}:shuffle").shuffle(trials)

    (out / "crops").mkdir(parents=True)
    bar = Progress(len(trials), "crops", unit="trial")
    for i, t in enumerate(trials, start=1):
        t["id"] = f"t{i:03d}"
        (crop,) = [
            c
            for c in plate_crops(t.pop("_cfg"), t.pop("_adv_dir"), t["image_id"])
            if c.plate_idx == t["plate_idx"]
        ]
        shown = {t["attacked"]: crop.adv, _other(t["attacked"]): crop.clean}
        for side in SIDES:
            shown[side].save(out / "crops" / f"{t['id']}_{side}.png")
        bar.update()
    key = {"seed": seed, "n_per_budget": n_trials, "budgets": budget_info, "trials": trials}
    (out / "trials.json").write_text(json.dumps(key, indent=1) + "\n")
    print(f"eyetest: {len(trials)} trials in {out}")


def passes(correct: int, n: int, n_trials: int | None = None, max_correct: int | None = None):
    """The pre-registered rule: all trials answered and at most max_correct right."""
    n_trials = N_TRIALS if n_trials is None else n_trials
    max_correct = MAX_CORRECT if max_correct is None else max_correct
    return n == n_trials and correct <= max_correct


def ladder_ceiling(passed: list[bool]) -> tuple[int | None, str | None]:
    """(index of the ceiling rung or None, "up" / "down" / None) for rungs in budget order."""
    ceiling = None
    for i, ok in enumerate(passed):
        if not ok:
            break
        ceiling = i
    if ceiling == len(passed) - 1:
        return ceiling, "up"
    return ceiling, ("down" if ceiling is None else None)


def score(
    sessions: list[Path], n_trials: int | None = None, max_correct: int | None = None
) -> dict:
    """Score one or more sessions together (a ladder can be extended in a later session).

    Writes scores.json into the last session given and returns the same content.
    """
    n_trials = N_TRIALS if n_trials is None else n_trials
    budgets, sizes = {}, {}
    for session in map(Path, sessions):
        key = json.loads((session / "trials.json").read_text())
        answers = _answers(session)
        for b in key["budgets"]:
            sizes[(b["ladder"], b["rung"])] = b["size"]
        for t in key["trials"]:
            entry = budgets.setdefault(f"{t['ladder']}/{t['rung']}", {"correct": 0, "n": 0})
            if t["id"] in answers:
                entry["n"] += 1
                entry["correct"] += answers[t["id"]] == t["attacked"]
    for entry in budgets.values():
        entry["complete"] = entry["n"] == n_trials
        entry["passed"] = passes(entry["correct"], entry["n"], n_trials, max_correct)

    ladders = {}
    for ladder in dict.fromkeys(lad for lad, _ in sizes):
        rungs = sorted((r for lad, r in sizes if lad == ladder), key=lambda r: sizes[(ladder, r)])
        entries = [budgets[f"{ladder}/{r}"] for r in rungs]
        if all(e["complete"] for e in entries):
            idx, extend = ladder_ceiling([e["passed"] for e in entries])
            ceiling = None if idx is None else rungs[idx]
        else:
            ceiling, extend = None, None  # no verdict until every rung is fully answered
        ladders[ladder] = {"rungs": rungs, "ceiling": ceiling, "extend": extend}
    result = {"budgets": budgets, "ladders": ladders}
    (Path(sessions[-1]) / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def make_server(session: Path, port: int = 8765, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """The eye-test page and its tiny API, for one session (call serve_forever on it).

    Threaded, because browsers open spare connections they may never use, and a server that
    handles one connection at a time would hang on them; a lock keeps answers in order.
    """
    session = Path(session)
    trials = json.loads((session / "trials.json").read_text())["trials"]
    order = [t["id"] for t in trials]
    answers_csv = session / "answers.csv"
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif self.path == "/next":
                self._json(200, _next(order, _answers(session)))
            elif self.path.startswith("/crop/"):
                _, _, tid, side = self.path.split("/")
                path = session / "crops" / f"{tid}_{side}.png"
                if tid not in order or side not in SIDES or not path.exists():
                    self._json(404, {"error": "no such crop"})
                else:
                    self._send(200, "image/png", path.read_bytes())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/answer":
                return self._json(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            tid, choice = body.get("trial"), body.get("choice")
            if tid not in order or choice not in SIDES:
                return self._json(400, {"error": "bad trial or choice"})
            with lock:
                answered = _answers(session)
                if tid not in answered:  # a double click or a resent answer is stored once
                    new = not answers_csv.exists()
                    with answers_csv.open("a", newline="") as f:
                        w = csv.writer(f)
                        if new:
                            w.writerow(["trial", "choice", "time"])
                        w.writerow([tid, choice, datetime.now(UTC).isoformat()])
                    answered[tid] = choice
            self._json(200, _next(order, answered))

        def _json(self, code, payload):
            self._send(code, "application/json", json.dumps(payload).encode())

        def _send(self, code, kind, data):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):  # keep the terminal quiet while the viewer works
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True  # Ctrl+C must not wait for idle browser connections
    return server


def serve(session: Path, port: int = 8765) -> None:
    server = make_server(session, port)
    print(f"eyetest: open http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\neyetest: stopped; answers so far are saved")
    finally:
        server.server_close()


def _answers(session: Path) -> dict[str, str]:
    path = Path(session) / "answers.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        answers = {}
        for row in csv.DictReader(f):
            answers.setdefault(row["trial"], row["choice"])  # the first answer counts
        return answers


def _next(order: list[str], answered: dict) -> dict:
    remaining = [tid for tid in order if tid not in answered]
    trial = None
    if remaining:
        tid = remaining[0]
        trial = {"id": tid, "left": f"/crop/{tid}/left", "right": f"/crop/{tid}/right"}
    return {
        "done": not remaining,
        "answered": len(order) - len(remaining),
        "total": len(order),
        "trial": trial,
    }


def _other(side: str) -> str:
    return "right" if side == "left" else "left"


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plate eye test</title>
<style>
  :root { --bg: #7f7f7f; --ink: #111; --muted: #2b2b2b; --btn: #e9e9e9; --btn-ink: #111; }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; background: var(--bg); color: var(--ink);
         font: 16px/1.4 system-ui, sans-serif; display: flex; flex-direction: column;
         align-items: center; justify-content: center; gap: 28px; padding: 24px 16px; }
  header { text-align: center; }
  h1 { font-size: 18px; font-weight: 600; margin: 0 0 4px; }
  #progress { color: var(--muted); font-variant-numeric: tabular-nums; }
  #pair { display: flex; flex-wrap: wrap; gap: 32px 48px; align-items: center;
          justify-content: center; min-height: 160px; }
  #pair[hidden] { display: none; }  /* the flex rule above would otherwise override [hidden] */
  figure { margin: 0; display: flex; flex-direction: column; align-items: center; gap: 12px; }
  img { display: block; image-rendering: pixelated; }  /* native size: never scaled */
  button { font: inherit; min-width: 140px; padding: 10px 18px; border: 1px solid #555;
           border-radius: 6px; background: var(--btn); color: var(--btn-ink); cursor: pointer; }
  button:hover { background: #fff; }
  button:disabled { opacity: 0.5; cursor: default; }
  kbd { font: 13px ui-monospace, monospace; padding: 1px 6px; border: 1px solid #555;
        border-radius: 4px; background: #d6d6d6; }
  #note { color: var(--muted); font-size: 14px; text-align: center; max-width: 520px; }
</style>
</head>
<body>
<header>
  <h1>Which crop has the noise?</h1>
  <div id="progress">Loading&hellip;</div>
</header>
<div id="pair">
  <figure><img id="left" alt="crop 1"><button id="b-left">Left <kbd>1</kbd></button></figure>
  <figure><img id="right" alt="crop 2"><button id="b-right">Right <kbd>2</kbd></button></figure>
</div>
<p id="note">Pick the one you think was changed, even if you have to guess.
Answers are saved as you go: you can close this tab and come back later.</p>
<script>
let current = null, busy = false;
const $ = (id) => document.getElementById(id);
function show(state) {
  if (state.done) {
    $("pair").hidden = true;
    $("progress").textContent = `All ${state.total} trials answered. Thank you!`;
    $("note").textContent = "You can close this tab and stop the server (Ctrl+C).";
    return;
  }
  current = state.trial.id;
  $("progress").textContent = `Trial ${state.answered + 1} of ${state.total}`;
  // Show the pair only once both crops have loaded, so neither appears first.
  const imgs = [["left", state.trial.left], ["right", state.trial.right]];
  $("pair").style.visibility = "hidden";
  Promise.all(imgs.map(([id, src]) => new Promise((ok) => {
    const img = $(id); img.onload = ok; img.onerror = ok; img.src = src;
  }))).then(() => { $("pair").style.visibility = "visible"; busy = false; setButtons(true); });
}
function setButtons(on) { $("b-left").disabled = !on; $("b-right").disabled = !on; }
async function choose(choice) {
  if (busy || !current) return;
  busy = true; setButtons(false);
  const r = await fetch("/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trial: current, choice }),
  });
  show(await r.json());
}
$("b-left").onclick = () => choose("left");
$("b-right").onclick = () => choose("right");
document.addEventListener("keydown", (e) => {
  if (e.key === "1" || e.key === "ArrowLeft") choose("left");
  if (e.key === "2" || e.key === "ArrowRight") choose("right");
});
setButtons(false);
fetch("/next").then((r) => r.json()).then(show);
</script>
</body>
</html>
"""
