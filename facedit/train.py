"""Boucle d'entraînement — F-T1 à F-T7, §7.3.

    python -m facedit.train --config configs/final.yaml
    python -m facedit.train --config configs/final.yaml --resume auto

Le planning de bruit vient de `DDPMScheduler` (F-T2) : il n'est pas réimplémenté. Tout
le reste — modèle, dropout de condition, EMA, checkpointing — est écrit ici.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from facedit.data.dataset import build_loader
from facedit.models.dit import build_dit
from facedit.models.ema import EMA
from facedit.utils.config import Config, load_config
from facedit.utils.repro import env_fingerprint, seed_everything


# --------------------------------------------------------------------------------------
# Dropout de condition — F-M7, F-M8
# --------------------------------------------------------------------------------------


def sample_condition_dropout(
    batch_size: int,
    probability: float,
    independent: bool,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Masques de dropout de condition, un par attribut.

    Deux régimes :

    - **conjoint** (F-M7, défaut) : un unique tirage par échantillon, les trois attributs
      tombent ensemble. Le modèle apprend une vraie branche inconditionnelle p(x), qui
      est exactement ce dont le CFG standard a besoin.
    - **indépendant** (F-M8, prio C) : un tirage par attribut. Le modèle voit alors les
      2³ combinaisons de conditionnement partiel, ce qui est la condition nécessaire au
      guidage par attribut (F-S6) — sans quoi ε_A ou ε_AG seraient évalués sur des
      combinaisons jamais rencontrées à l'entraînement.

    Coût du régime indépendant : à p=0.1, la branche entièrement inconditionnelle n'est
    plus vue que dans 0.1³ = 0.1 % des cas au lieu de 10 %. On relève donc la probabilité
    par attribut pour conserver la même masse inconditionnelle : p_attr = p^(1/3).
    """
    if probability <= 0.0:
        zeros = torch.zeros(batch_size, dtype=torch.bool, device=device)
        return {"age": zeros, "gender": zeros.clone(), "ita": zeros.clone()}

    if not independent:
        mask = torch.rand(batch_size, device=device) < probability
        return {"age": mask, "gender": mask, "ita": mask}

    per_attribute = probability ** (1.0 / 3.0)
    return {
        name: torch.rand(batch_size, device=device) < per_attribute
        for name in ("age", "gender", "ita")
    }


# --------------------------------------------------------------------------------------
# Checkpoints — F-T4, F-T5
# --------------------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    step: int,
    model,
    ema: EMA,
    optimizer,
    cfg: Config,
    extra: Optional[Dict] = None,
) -> None:
    """Checkpoint complet : modèle, EMA, optimiseur, compteur de pas (F-T4).

    Écriture atomique via un fichier temporaire renommé : un redémarrage Windows au
    mauvais moment (risque §12, probabilité « élevée ») ne doit pas laisser derrière lui
    un checkpoint tronqué qui casserait la reprise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "config_hash": cfg.hash(),
        "env": env_fingerprint(),
        **(extra or {}),
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def numbered_checkpoints(out_dir: Path) -> List[Tuple[int, Path]]:
    """Checkpoints `ckpt_<pas>.pt`, triés par pas croissant.

    Le filtre sur `isdigit` n'est pas cosmétique : le dossier contient aussi
    `ckpt_last.pt` et `ckpt_final.pt`, que le motif `ckpt_*.pt` capture et dont le
    suffixe n'est pas un entier.
    """
    found = []
    for path in out_dir.glob("ckpt_*.pt"):
        suffix = path.stem.split("_")[-1]
        if suffix.isdigit():
            found.append((int(suffix), path))
    return sorted(found)


def find_latest_checkpoint(out_dir: Path) -> Optional[Path]:
    """Checkpoint le plus avancé, en préférant `ckpt_last.pt` s'il est au moins aussi
    récent : c'est lui qui est réécrit à chaque sauvegarde."""
    numbered = numbered_checkpoints(out_dir)
    last = out_dir / "ckpt_last.pt"
    if last.exists():
        return last
    return numbered[-1][1] if numbered else None


def prune_checkpoints(out_dir: Path, keep: int) -> None:
    """Ne conserve que les `keep` derniers checkpoints numérotés.

    `ckpt_last.pt` et `ckpt_final.pt` ne sont pas numérotés et survivent toujours — la
    reprise (F-T5) en dépend.
    """
    if keep <= 0:
        return
    for _, stale in numbered_checkpoints(out_dir)[:-keep]:
        stale.unlink(missing_ok=True)


# --------------------------------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------------------------------


def train(cfg: Config, resume: Optional[str] = None, max_steps: Optional[int] = None) -> Path:
    from diffusers import DDPMScheduler
    from torch.utils.tensorboard import SummaryWriter

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(out_dir / "config.yaml")

    # --- données ---------------------------------------------------------------------
    loader = build_loader(cfg, "train")
    dataset_stats = loader.dataset.stats()
    print(f"[data] {dataset_stats['n']} latents {dataset_stats['latent_size']}² · "
          f"âge [{dataset_stats['age_min']:.0f}, {dataset_stats['age_max']:.0f}] · "
          f"ITA μ={dataset_stats['ita_mean']:.1f}° σ={dataset_stats['ita_std']:.1f}° · "
          f"σ(latents)={dataset_stats['latent_std']:.3f}")

    # --- modèle ----------------------------------------------------------------------
    model = build_dit(cfg.model).to(device)
    ema = EMA(model, cfg.train.ema_decay, cfg.train.ema_start).to(device)
    # Le nom de la variante est dérivé de la config, jamais codé en dur : le palier
    # `final_v2` porte une profondeur et une largeur DiT-S, l'afficher « XS » induirait
    # le rapport en erreur.
    variant = f"DiT d{cfg.model.depth}/w{cfg.model.hidden_size}/p{cfg.model.patch_size}"
    print(f"[model] {variant} · {model.num_parameters / 1e6:.2f} M paramètres · "
          f"{model.num_patches} tokens")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
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
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # --- reprise (F-T5) ----------------------------------------------------------------
    start_step = 0
    if resume:
        path = find_latest_checkpoint(out_dir) if resume == "auto" else Path(resume)
        if path is None:
            print(f"[resume] aucun checkpoint dans {out_dir} — démarrage à zéro")
        else:
            payload = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"])
            ema.load_state_dict(payload["ema"])
            optimizer.load_state_dict(payload["optimizer"])
            start_step = int(payload["step"])
            if payload.get("config_hash") != cfg.hash():
                print(
                    "[resume] ATTENTION : la configuration a changé depuis ce checkpoint "
                    f"({payload.get('config_hash')} → {cfg.hash()}). La reprise se poursuit, "
                    "mais l'expérience n'est plus celle d'origine."
                )
            print(f"[resume] reprise depuis {path} au pas {start_step}")

    writer = SummaryWriter(out_dir / "tb")
    total_steps = max_steps or cfg.train.steps

    # --- boucle ------------------------------------------------------------------------
    model.train()
    iterator = iter(loader)
    loss_window: deque = deque(maxlen=cfg.train.log_every)
    started = time.time()
    last_log = started

    for step in range(start_step, total_steps):
        batch = next(iterator)
        latents = batch["latent"].to(device, non_blocking=True)
        age = batch["age"].to(device, non_blocking=True)
        gender = batch["gender"].to(device, non_blocking=True)
        ita = batch["ita"].to(device, non_blocking=True)
        batch_size = latents.shape[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, cfg.train.num_train_timesteps, (batch_size,), device=device, dtype=torch.long
        )
        noisy = scheduler.add_noise(latents, noise, timesteps)

        drop = sample_condition_dropout(
            batch_size, cfg.train.cond_dropout, cfg.train.independent_cond_dropout, device
        )

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_bf16):
            predicted = model(noisy, timesteps, age, gender, ita, drop=drop)
            # F-T1 : MSE sur le bruit prédit. Calculée en fp32 même sous autocast — en
            # bf16 la mantisse à 8 bits fait perdre les décimales de la loss bien avant
            # que les gradients ne s'en ressentent, et le suivi F-T6 devient illisible.
            loss = F.mse_loss(predicted.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        ema.update(model, step)

        loss_window.append(loss.detach().item())

        # --- journalisation (F-T6) ---
        if (step + 1) % cfg.train.log_every == 0:
            now = time.time()
            steps_per_second = cfg.train.log_every / (now - last_log)
            last_log = now
            mean_loss = float(np.mean(loss_window))
            remaining = (total_steps - step - 1) / max(steps_per_second, 1e-6)

            writer.add_scalar("train/loss", mean_loss, step + 1)
            writer.add_scalar("train/grad_norm", float(grad_norm), step + 1)
            writer.add_scalar("train/ema_decay", ema.current_decay, step + 1)
            writer.add_scalar("perf/steps_per_second", steps_per_second, step + 1)
            if device.type == "cuda":
                writer.add_scalar(
                    "perf/vram_gb", torch.cuda.max_memory_allocated() / 1e9, step + 1
                )
            print(
                f"step {step + 1:>6}/{total_steps} · loss {mean_loss:.4f} · "
                f"|g| {float(grad_norm):.2f} · {steps_per_second:.2f} it/s · "
                f"reste {remaining / 3600:.2f} h"
            )

        # --- grille d'échantillons (F-T6) ---
        if cfg.train.sample_every and (step + 1) % cfg.train.sample_every == 0:
            grid = _training_grid(cfg, ema, device, seed=cfg.seed)
            if grid is not None:
                writer.add_image("samples/attribute_sweep", grid, step + 1, dataformats="HWC")
            model.train()

        # --- checkpoint (F-T4) ---
        if (step + 1) % cfg.train.ckpt_every == 0:
            save_checkpoint(
                out_dir / f"ckpt_{step + 1:07d}.pt", step + 1, model, ema, optimizer, cfg,
                extra={"loss": float(np.mean(loss_window))},
            )
            save_checkpoint(out_dir / "ckpt_last.pt", step + 1, model, ema, optimizer, cfg)
            prune_checkpoints(out_dir, cfg.train.keep_last_ckpts)

    # --- fin ---------------------------------------------------------------------------
    elapsed = time.time() - started
    final_path = out_dir / "ckpt_final.pt"
    save_checkpoint(
        final_path, total_steps, model, ema, optimizer, cfg,
        extra={"loss": float(np.mean(loss_window)) if loss_window else None,
               "wall_clock_seconds": elapsed},
    )
    writer.close()

    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "steps": total_steps,
                "steps_run": total_steps - start_step,
                "wall_clock_hours": round(elapsed / 3600, 3),
                "final_loss": float(np.mean(loss_window)) if loss_window else None,
                "peak_vram_gb": (
                    round(torch.cuda.max_memory_allocated() / 1e9, 2)
                    if device.type == "cuda"
                    else None
                ),
                "params_millions": round(model.num_parameters / 1e6, 2),
                "dataset": dataset_stats,
                "config_hash": cfg.hash(),
                "env": env_fingerprint(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[train] terminé en {elapsed / 3600:.2f} h → {final_path}")
    return final_path


def _training_grid(cfg: Config, ema: EMA, device: torch.device, seed: int):
    """Planche de balayage d'attributs à partir des poids EMA courants.

    Le VAE est chargé puis relâché à chaque appel : le garder en mémoire coûterait
    ~160 Mo de VRAM en permanence pour un usage toutes les 2000 étapes, et NF-2 plafonne
    l'entraînement à 6 Go.
    """
    from facedit.sample import Generator, attribute_sweep_grid, build_ddim_scheduler
    from facedit.data.encode import load_vae

    try:
        vae = load_vae(cfg.data.vae_id, str(device), torch.float16)
        generator = Generator(
            model=ema.shadow.eval(),
            cfg=cfg,
            device=str(device),
            scheduler=build_ddim_scheduler(cfg),
            vae=vae,
        )
        grid = attribute_sweep_grid(generator, seed=seed)
        del vae
        torch.cuda.empty_cache()
        return grid
    except Exception as exc:  # une grille ratée ne doit jamais tuer un entraînement
        print(f"[sample] grille ignorée : {exc.__class__.__name__}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement du DiT FaceDiT")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="chemin d'un checkpoint, ou 'auto'")
    parser.add_argument("--max-steps", type=int, default=None, help="écrase train.steps")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    train(cfg, args.resume, args.max_steps)


if __name__ == "__main__":
    main()
