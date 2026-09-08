"""Vérification des exigences non fonctionnelles — §5.

    python -m facedit.benchmark --config configs/final.yaml

Les seuils du §5 sont des exigences, pas des vœux : ils se vérifient. Ce module les
mesure et rend un verdict par exigence, dans un JSON versionné joint au rapport.

| ID | Exigence | Seuil |
|----|----------|-------|
| NF-1 | Latence de génération d'une image, hors restauration | < 3 s |
| NF-2 | Empreinte VRAM à l'entraînement | < 6 Go |
| NF-3 | Empreinte VRAM à l'inférence | < 4 Go |
| NF-4 | Durée totale de l'entraînement final | < 8 h |
| NF-5 | Le dataset pré-encodé tient en RAM | < 1 Go |

NF-4 est *extrapolée* depuis le débit réellement mesuré sur quelques centaines de pas,
et non recopiée de l'estimation « 3 à 5 h » du §7.3 — que le PRD demande explicitement
de ne pas reprendre telle quelle.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from facedit.models.dit import build_dit
from facedit.utils.config import Config, load_config
from facedit.utils.repro import env_fingerprint, seed_everything


def _verdict(value: Optional[float], threshold: float, unit: str, lower_is_better: bool = True):
    if value is None:
        return {"value": None, "threshold": threshold, "unit": unit, "status": "non mesuré"}
    ok = value < threshold if lower_is_better else value > threshold
    return {
        "value": round(float(value), 3),
        "threshold": threshold,
        "unit": unit,
        "status": "conforme" if ok else "NON CONFORME",
    }


# --------------------------------------------------------------------------------------
# NF-1 / NF-3 — inférence
# --------------------------------------------------------------------------------------


@torch.no_grad()
def benchmark_inference(cfg: Config, device: torch.device, warmup: int = 3, runs: int = 10) -> Dict:
    """Latence et VRAM d'une génération complète, VAE compris, hors restauration.

    Le premier appel après le chargement du modèle inclut l'initialisation des noyaux
    CUDA et coûte typiquement plusieurs secondes de plus. On chauffe donc avant de
    mesurer : NF-1 porte sur l'usage réel de l'interface, où l'utilisateur enchaîne les
    générations, pas sur le tout premier clic.
    """
    from facedit.data.encode import load_vae
    from facedit.sample import Generator, build_ddim_scheduler

    model = build_dit(cfg.model).to(device).eval()
    vae = load_vae(cfg.data.vae_id, str(device), torch.float16 if device.type == "cuda" else torch.float32)
    generator = Generator(
        model=model, cfg=cfg, device=str(device),
        scheduler=build_ddim_scheduler(cfg), vae=vae,
    )

    results: Dict[str, object] = {}
    warmup_peak = 0.0
    steady_peak = 0.0

    for batch_size in (1, 8):
        # Pic de chauffe et pic de régime établi sont mesurés séparément. Les confondre
        # masquerait exactement le défaut qui a été trouvé ici : un espace de travail
        # transitoire, alloué une seule fois, dix fois supérieur au besoin réel. Un
        # utilisateur dont la carte est déjà à moitié occupée tombe en OOM à la première
        # génération, sur une empreinte de régime pourtant confortable.
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            generator.generate(35.0, 1, 30.0, batch_size=batch_size, seed=0)
        if device.type == "cuda":
            torch.cuda.synchronize()
            warmup_peak = max(warmup_peak, torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.reset_peak_memory_stats()

        timings = []
        for run in range(runs):
            start = time.perf_counter()
            generator.generate(35.0, 1, 30.0, batch_size=batch_size, seed=run)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)

        if device.type == "cuda":
            steady_peak = max(steady_peak, torch.cuda.max_memory_allocated() / 1e9)

        timings = np.asarray(timings)
        results[f"batch_{batch_size}"] = {
            "total_seconds_median": float(np.median(timings)),
            "seconds_per_image": float(np.median(timings) / batch_size),
            "p95_seconds": float(np.percentile(timings, 95)),
        }

    results["num_steps"] = cfg.sample.num_steps
    results["params_millions"] = round(model.num_parameters / 1e6, 2)
    if device.type == "cuda":
        results["peak_vram_warmup_gb"] = round(warmup_peak, 3)
        results["peak_vram_steady_gb"] = round(steady_peak, 3)
        results["cudnn_benchmark"] = torch.backends.cudnn.benchmark

    del model, vae, generator
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # NF-3 est jugée sur le pic le plus élevé des deux : c'est celui qui déclenche un
    # OOM, qu'il soit transitoire ou non.
    peak_gb = max(warmup_peak, steady_peak) if device.type == "cuda" else None

    return {
        "detail": results,
        "NF-1": _verdict(results["batch_1"]["total_seconds_median"], 3.0, "s"),
        "NF-3": _verdict(peak_gb, 4.0, "Go"),
    }


# --------------------------------------------------------------------------------------
# NF-2 / NF-4 — entraînement
# --------------------------------------------------------------------------------------


def benchmark_training(cfg: Config, device: torch.device, steps: int = 60) -> Dict:
    """VRAM de pointe et débit sur quelques dizaines de pas, sur des latents synthétiques.

    Des latents synthétiques suffisent : la VRAM et le débit ne dépendent que des formes
    des tenseurs, pas de leur contenu. Cela permet de vérifier NF-2 et NF-4 **avant**
    d'avoir encodé le dataset — donc avant de s'engager dans huit heures de calcul.
    """
    from diffusers import DDPMScheduler

    model = build_dit(cfg.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas),
        weight_decay=cfg.train.weight_decay,
        fused=cfg.train.fused_adam and device.type == "cuda",
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
        clip_sample=False,
    )
    use_bf16 = cfg.train.precision == "bf16" and device.type == "cuda"

    batch = cfg.train.batch_size
    size = cfg.model.latent_size
    latents = torch.randn(batch, 4, size, size, device=device)
    age = torch.rand(batch, device=device) * 50 + 20
    gender = torch.randint(0, 2, (batch,), device=device)
    ita = torch.rand(batch, device=device) * 100 - 40

    from facedit.train import sample_condition_dropout

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Les dix premiers pas incluent la compilation des noyaux et l'allocation du cache
    # CUDA : ils sont chronométrés à part et jetés. `start_window` est initialisé ici et
    # non dans la boucle, sinon un appel sur CPU — ou avec `steps <= 10` — sortirait de la
    # boucle sans l'avoir jamais lié, et le calcul de `elapsed` lèverait un NameError.
    warmup = min(10, max(0, steps - 1))
    start_window = time.perf_counter()
    for step in range(steps):
        if step == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_window = time.perf_counter()

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, cfg.train.num_train_timesteps, (batch,), device=device)
        noisy = scheduler.add_noise(latents, noise, timesteps)
        drop = sample_condition_dropout(
            batch, cfg.train.cond_dropout, cfg.train.independent_cond_dropout, device
        )

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            predicted = model(noisy, timesteps, age, gender, ita, drop=drop)
            loss = torch.nn.functional.mse_loss(predicted.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_window
    measured_steps = steps - warmup
    steps_per_second = measured_steps / elapsed
    projected_hours = cfg.train.steps / steps_per_second / 3600.0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None

    del model, optimizer, latents
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "detail": {
            "batch_size": batch,
            "precision": cfg.train.precision,
            "steps_measured": measured_steps,
            "steps_per_second": round(steps_per_second, 2),
            "images_per_second": round(steps_per_second * batch, 1),
            "planned_steps": cfg.train.steps,
        },
        "NF-2": _verdict(peak_gb, 6.0, "Go"),
        "NF-4": _verdict(projected_hours, 8.0, "h"),
        "note": (
            "NF-4 est extrapolée du débit mesuré ici, et remplace l'estimation « 3 à 5 h » "
            "du §7.3 que le PRD demande de ne pas reprendre telle quelle. Le débit réel de "
            "l'entraînement complet est journalisé dans runs/*/train_summary.json."
        ),
    }


# --------------------------------------------------------------------------------------
# NF-5 — empreinte mémoire du cache
# --------------------------------------------------------------------------------------


def benchmark_cache(cfg: Config) -> Dict:
    path = cfg.data.latents_path("train")
    if not path.exists():
        expected = None
        return {
            "detail": {"path": str(path), "exists": False},
            "NF-5": _verdict(expected, 1.0, "Go"),
        }
    size_gb = path.stat().st_size / 1e9
    array = np.load(path, mmap_mode="r")
    return {
        "detail": {
            "path": str(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "n_latents": int(array.shape[0]),
        },
        "NF-5": _verdict(size_gb, 1.0, "Go"),
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def run_benchmark(cfg: Config, skip_training: bool = False) -> Dict:
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    report: Dict[str, object] = {
        "config": cfg.name,
        "config_hash": cfg.hash(),
        "device": str(device),
        "env": env_fingerprint(),
        "reference": "§5 — exigences non fonctionnelles",
    }

    print("[bench] inférence (NF-1, NF-3)…")
    report["inference"] = benchmark_inference(cfg, device)
    if not skip_training:
        print("[bench] entraînement (NF-2, NF-4)…")
        report["training"] = benchmark_training(cfg, device)
    print("[bench] cache (NF-5)…")
    report["cache"] = benchmark_cache(cfg)

    verdicts = {}
    for section in ("inference", "training", "cache"):
        for key, value in (report.get(section) or {}).items():
            if key.startswith("NF-"):
                verdicts[key] = value
    report["verdicts"] = dict(sorted(verdicts.items()))

    print(f"\n{'=' * 62}\nEXIGENCES NON FONCTIONNELLES (§5)\n{'=' * 62}")
    labels = {
        "NF-1": "Latence de génération",
        "NF-2": "VRAM entraînement",
        "NF-3": "VRAM inférence",
        "NF-4": "Durée entraînement final",
        "NF-5": "Cache en RAM",
    }
    for key, entry in report["verdicts"].items():
        value = "—" if entry["value"] is None else f"{entry['value']:g} {entry['unit']}"
        print(f"  {key}  {labels[key]:<26} {value:>12}  (< {entry['threshold']:g} "
              f"{entry['unit']})  {entry['status']}")
    print("=" * 62)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérification des exigences §5")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    report = run_benchmark(cfg, args.skip_training)

    out = Path(args.out) if args.out else Path(cfg.eval.out_dir) / "benchmark_nf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[bench] → {out}")


if __name__ == "__main__":
    main()
