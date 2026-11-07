"""Attack stage: make the adversarial version of every evaluation image once, and cache it.

    python -m plate_cloak attack --config configs/base.yaml [--variant V] [key=value ...]

Output: cache/adv/<fingerprint>/ holding
    <image_id>.png   attacked image, 8-bit RGB, same size as the EvalImage
    audit.csv        largest actual change per channel for each image, in 1/255 units
    settings.yaml    what made this cache: attack settings, surrogate and its weights' SHA-256,
                     image count, code version
    done.json        written last; its presence means the cache is complete

The fingerprint covers every setting that changes the attacked pixels (and nothing that only
changes evaluation), so a cache is reused exactly when it would come out the same.
Each PNG is written to a temporary name and renamed, so a crash never leaves a cut-off image;
an interrupted run picks up where it stopped because the attack is deterministic per image.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from plate_cloak.attack.pgd import AttackSettings, audit, pgd, random_sign_noise, to_uint8
from plate_cloak.config import resolve_path
from plate_cloak.data.eval_images import eval_image_ids, load_by_id
from plate_cloak.runs import provenance

AUDIT_COLUMNS = ["image_id", "R", "G", "B", "Y", "Cb", "Cr"]


def fingerprint(cfg: DictConfig, image_ids: list[str]) -> str:
    """First 12 hex characters of a SHA-256 over everything that shapes the attacked pixels."""
    relevant = {
        "attack": OmegaConf.to_container(cfg.attack, resolve=True),
        "surrogate": OmegaConf.to_container(cfg.models.surrogate, resolve=True),
        "imgsz": cfg.data.imgsz,
        "min_plate_w": cfg.data.min_plate_w,
        "min_plate_h": cfg.data.min_plate_h,
        "seed": cfg.seed,
        "image_ids": list(image_ids),
    }
    blob = json.dumps(relevant, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def run_attack(cfg: DictConfig) -> Path:
    """Attack every evaluation image (or reuse a finished cache) and return the cache folder."""
    ids = eval_image_ids(cfg)
    out = resolve_path(cfg.output.cache_root) / fingerprint(cfg, ids)
    digest = weights_digest(cfg)
    if (out / "done.json").exists():
        _check_same_weights(out, digest)
        print(f"attack: reusing finished cache {out}")
        return out

    out.mkdir(parents=True, exist_ok=True)
    s = AttackSettings.from_cfg(cfg.attack)
    device = torch.device(cfg.device)
    net = load_surrogate(cfg).to(device)
    rows = []
    for n, image_id in enumerate(ids, start=1):
        ev = load_by_id(cfg, image_id)
        clean = np.array(ev.image)  # (H, W, 3) uint8 RGB; np.array copies, so it is writable
        png = out / f"{image_id}.png"
        if png.exists():  # finished before an interruption
            adv = np.array(Image.open(png).convert("RGB"))
        else:
            adv = _attack_one(net, clean, ev.boxes, s, cfg.seed, image_id, device)
            _save_png(adv, png)
        rows.append({"image_id": image_id, **audit(clean, adv)})
        if n % 50 == 0 or n == len(ids):
            print(f"attack: {n}/{len(ids)} images")

    worst = {k: max(r[k] for r in rows) for k in AUDIT_COLUMNS[1:]} if rows else {}
    _check_budget(worst, cfg.attack)
    _write_audit(rows, out / "audit.csv")
    settings = {
        "fingerprint": out.name,
        "attack": OmegaConf.to_container(cfg.attack, resolve=True),
        "surrogate": {**OmegaConf.to_container(cfg.models.surrogate), "weights_sha256": digest},
        "data": {k: cfg.data[k] for k in ("imgsz", "min_plate_w", "min_plate_h")},
        "seed": cfg.seed,
        "n_images": len(ids),
        "provenance": provenance(),
    }
    (out / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    (out / "done.json").write_text(json.dumps({"n_images": len(ids), "max_change": worst}) + "\n")
    print(f"attack: done -> {out}  (max change {worst})")
    return out


def load_surrogate(cfg: DictConfig) -> torch.nn.Module:
    """The surrogate's raw network: eval mode, float32, weights frozen."""
    from plate_cloak.models.zoo import load_detector

    net = load_detector(cfg.models.surrogate.name, cfg.models.surrogate.seed).model
    net = net.float().eval()
    for p in net.parameters():
        p.requires_grad_(False)  # PGD only needs gradients for the pixels
    return net


def weights_digest(cfg: DictConfig) -> str:
    """SHA-256 of the surrogate's weights file, so a retrained model can't reuse an old cache."""
    from plate_cloak.models.zoo import weights_path

    path = weights_path(cfg.models.surrogate.name, cfg.models.surrogate.seed)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_seed(seed: int, image_id: str) -> int:
    """A per-image seed for the random control: the same image gets the same noise in any order."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8], "big")


def _attack_one(net, clean_u8, boxes, s: AttackSettings, seed, image_id, device) -> np.ndarray:
    x = torch.from_numpy(clean_u8).permute(2, 0, 1)[None].float().div(255).to(device)
    if s.method == "pgd":
        adv = pgd(net, x, boxes, s)
    elif s.method == "random":
        generator = torch.Generator(device=device).manual_seed(image_seed(seed, image_id))
        adv = random_sign_noise(x, boxes, s, generator)
    else:
        raise ValueError(f"unknown attack.method {s.method!r}; options: pgd, random")
    return to_uint8(adv)


def _save_png(img: np.ndarray, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(img).save(tmp, format="PNG")
    os.replace(tmp, path)  # atomic: the PNG is either absent or complete


def _check_budget(worst: dict[str, float], attack: DictConfig) -> None:
    """Refuse to mark the cache done if any saved image breaks the budget (that would be a bug)."""
    if attack.space == "rgb":
        limits = {c: attack.eps.rgb for c in ("R", "G", "B")}
    else:
        limits = {"Y": attack.eps.y, "Cb": attack.eps.cb, "Cr": attack.eps.cr}
    over = {c: worst[c] for c, limit in limits.items() if c in worst and worst[c] > limit + 1e-6}
    if over:
        raise RuntimeError(f"saved images exceed the budget {limits}: {over}")


def _check_same_weights(out: Path, digest: str) -> None:
    recorded = yaml.safe_load((out / "settings.yaml").read_text())["surrogate"]["weights_sha256"]
    if recorded != digest:
        raise RuntimeError(
            f"{out} was made with different surrogate weights (sha256 {recorded[:12]}..., now "
            f"{digest[:12]}...). Move that folder aside and rerun the attack."
        )


def _write_audit(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if k != "image_id" else v) for k, v in row.items()})
