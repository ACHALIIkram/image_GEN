"""Pré-encodage VAE du dataset — F-D5, F-D6, et vérification de l'hypothèse H3.

Produit trois artefacts, conformes au contrat §6.3 :

    <split>_<size>_latents.npy      (N, 4, S, S) float16
    <split>_<size>_labels.npy       (N, 3)       float32  [age_years, gender_01, ita_deg]
    <split>_<size>_age_bounds.npy   (N, 2)       float32  [age_low, age_high]

`age_bounds` n'est pas au contrat mais lui est adjoint : le contrat fige `labels[:, 0]`
au centre de tranche (valeur déterministe), tandis que F-D3 demande un tirage uniforme
dans la tranche à chaque chargement. Stocker les bornes permet les deux — le contrat est
respecté, et `LatentDataset` re-tire l'âge à chaque pas quand les bornes sont présentes.

Le VAE est **gelé** (§14) : il n'est jamais entraîné, seulement appelé en inférence.
On encode par la moyenne de la postérieure (`latent_dist.mean`) et non par un tirage :
le cache est écrit une fois pour toutes, un tirage y figerait une réalisation de bruit
arbitraire au lieu de la régulariser.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from facedit.data.fairface import (
    FairFaceImages,
    age_bin_centers,
    collate_uint8,
    open_fairface,
    resize_uint8,
    to_signed_float,
    to_uint8,
)
from facedit.data.ita import compute_ita
from facedit.utils.config import Config


# --------------------------------------------------------------------------------------
# VAE
# --------------------------------------------------------------------------------------


def load_vae(vae_id: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Charge le VAE de Stable Diffusion, gelé et en mode évaluation.

    L'import explicite de `safetensors.torch` n'est pas décoratif : `diffusers` 0.32.2
    appelle `safetensors.torch.load_file` en n'ayant importé que le paquet racine. Selon
    ce que les autres dépendances ont déjà chargé, le sous-module peut ne pas être
    enregistré, et l'`AttributeError` qui en résulte est capturée puis re-levée par
    diffusers en un `OSError: Unable to load weights` qui désigne à tort le fichier de
    poids comme corrompu. Importer le sous-module ici rend le chargement déterministe.
    """
    import safetensors.torch  # noqa: F401  (voir docstring)

    from diffusers import AutoencoderKL

    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
    vae.to(device).eval()
    vae.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images(
    vae, images_uint8: np.ndarray, scale: float, device: str = "cuda"
) -> torch.Tensor:
    """(B, H, W, 3) uint8 → latents (B, 4, S, S) float32, déjà mis à l'échelle."""
    pixels = torch.from_numpy(to_signed_float(images_uint8)).to(device=device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mean * scale
    return latents.float()


@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor, scale: float) -> np.ndarray:
    """Latents (B, 4, S, S) mis à l'échelle → images (B, H, W, 3) uint8."""
    pixels = vae.decode((latents / scale).to(vae.dtype)).sample
    return to_uint8(pixels.clamp(-1.0, 1.0))


# --------------------------------------------------------------------------------------
# H3 : le VAE reconstruit-il correctement des visages en 128×128 ?
# --------------------------------------------------------------------------------------


@torch.no_grad()
def vae_reconstruction_report(
    cfg: Config, num_images: int = 256, split: str = "val"
) -> Dict[str, object]:
    """PSNR / LPIPS de reconstruction VAE (H3, jalon semaine 1, risque §12).

    Repères d'interprétation, à citer dans le rapport :
      - PSNR > 28 dB et LPIPS < 0.10 → le VAE n'est pas le facteur limitant ;
      - PSNR < 24 dB ou LPIPS > 0.20 → le plancher de FID est imposé par le VAE, il
        faut basculer en 256×256 (repli prévu au §12, même coût en tokens).
    """
    device = cfg.device if torch.cuda.is_available() else "cpu"
    vae = load_vae(cfg.data.vae_id, device, torch.float32)

    source = open_fairface(cfg.data, split)
    count = min(num_images, len(source))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(source), size=count, replace=False)

    lpips_model = None
    try:
        import lpips

        lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)
    except Exception as exc:
        print(f"[H3] LPIPS indisponible ({exc}) : seul le PSNR sera reporté.")

    psnr_values, lpips_values = [], []
    batch = 16
    for start in tqdm(range(0, count, batch), desc="H3 reconstruction VAE"):
        chunk = indices[start : start + batch]
        originals = np.stack(
            [resize_uint8(source.load_image(int(i)), cfg.data.image_size) for i in chunk]
        )
        latents = encode_images(vae, originals, cfg.data.vae_scale, device)
        recons = decode_latents(vae, latents, cfg.data.vae_scale)

        # PSNR sur l'échelle [0, 255], erreur moyenne par image.
        error = originals.astype(np.float64) - recons.astype(np.float64)
        mse = (error**2).mean(axis=(1, 2, 3))
        psnr_values.extend(10.0 * np.log10(255.0**2 / np.maximum(mse, 1e-12)))

        if lpips_model is not None:
            a = torch.from_numpy(to_signed_float(originals)).to(device)
            b = torch.from_numpy(to_signed_float(recons)).to(device)
            lpips_values.extend(lpips_model(a, b).flatten().cpu().numpy().tolist())

    report = {
        "hypothesis": "H3 — le VAE SD reconstruit correctement des visages à cette résolution",
        "vae_id": cfg.data.vae_id,
        "image_size": cfg.data.image_size,
        "latent_size": cfg.data.latent_size,
        "num_images": int(count),
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_p05": float(np.percentile(psnr_values, 5)),
        "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
        "lpips_p95": float(np.percentile(lpips_values, 95)) if lpips_values else None,
    }
    psnr_ok = report["psnr_mean"] > 28.0
    lpips_ok = report["lpips_mean"] is None or report["lpips_mean"] < 0.10
    report["verdict"] = (
        "H3 validée" if psnr_ok and lpips_ok else "H3 à discuter — voir repli 256×256 (§12)"
    )
    return report


# --------------------------------------------------------------------------------------
# Encodage du dataset
# --------------------------------------------------------------------------------------


def _temp_memmap(
    path: Path, shape: Tuple[int, ...], dtype, resume: bool = False
) -> np.ndarray:
    """Ouvre le tampon temporaire, en réutilisant son contenu si l'on reprend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if resume and path.exists():
        existing = np.lib.format.open_memmap(path, mode="r+")
        if existing.shape == shape and existing.dtype == dtype:
            return existing
        # Forme incompatible : le tampon vient d'une autre configuration, on repart de zéro
        # plutôt que de mélanger deux encodages.
        del existing
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _progress_path(cache: Path, split: str) -> Path:
    return cache / f".tmp_{split}_progress.npz"


def _save_progress(
    cache: Path, split: str, cursor: int, consumed: int, labels, bounds,
    rejected: int, region_sources, ita_iqr_values,
) -> None:
    """Point de reprise de l'encodage.

    Écrit dans un fichier temporaire puis renommé : une coupure pendant l'écriture du
    point de reprise ne doit pas détruire le précédent.
    """
    target = _progress_path(cache, split)
    staging = target.with_name(target.name + ".part")
    # Descripteur ouvert plutôt que chemin : `np.savez` ajoute silencieusement `.npz` à
    # un nom de fichier qui n'en a pas, et le renommage suivant échouerait sur un fichier
    # introuvable. Avec un descripteur, numpy écrit exactement où on lui demande.
    with open(staging, "wb") as handle:
        np.savez(
            handle, cursor=cursor, consumed=consumed, labels=labels, bounds=bounds,
            rejected=rejected,
            region_keys=np.array(list(region_sources.keys()), dtype=object),
            region_values=np.array(list(region_sources.values()), dtype=np.int64),
            ita_iqr=np.asarray(ita_iqr_values, dtype=np.float32),
        )
    staging.replace(target)


def encode_split(
    cfg: Config,
    split: str = "train",
    limit: Optional[int] = None,
    overwrite: bool = False,
    resume: bool = False,
    checkpoint_every: int = 50,
    keep_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Encode un split complet vers le cache `.npy` (F-D5).

    Ordre des opérations par image : redimensionnement → ITA (sur le uint8 final, donc
    exactement la même mesure que celle appliquée aux images générées en évaluation) →
    encodage VAE. Les images dont l'ITA échoue sont écartées : conditionner sur un ITA
    inventé serait pire que de perdre 1 % du jeu.

    Le flip (F-D6) est appliqué **avant** l'encodage et double le cache. L'ITA du miroir
    est réutilisé sans recalcul — un flip horizontal ne change aucune valeur de pixel,
    seulement leur position.

    `keep_mask` (F-D9) est un masque booléen indexé sur le dataset source, produit par
    `facedit.data.filter_faces`. Il est appliqué **avant** l'ITA et avant le VAE, de sorte
    qu'aucun calcul n'est dépensé sur une image écartée. Le filtrer ici plutôt que de
    sous-échantillonner le cache après coup garantit que latents, labels et bornes d'âge
    restent alignés par construction : il n'existe jamais d'index à faire correspondre.
    """
    data = cfg.data
    latents_path = data.latents_path(split)
    if latents_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"{latents_path} existe déjà. Relancer avec --overwrite pour le remplacer."
        )

    device = cfg.device if torch.cuda.is_available() else "cpu"
    vae = load_vae(data.vae_id, device, torch.float16)

    source = open_fairface(data, split)
    total = len(source) if limit is None else min(limit, len(source))
    print(f"[encode] {source.origin} — {total} images retenues après filtrage d'âge")

    size = data.latent_size
    cache = data.cache
    cache.mkdir(parents=True, exist_ok=True)

    # --- reprise (§12 : « perte suite à redémarrage Windows », probabilité élevée) ---
    cursor = 0
    consumed = 0
    rejected = 0
    filtered_out = 0
    unalignable = 0
    region_sources: Counter = Counter()
    ita_iqr_values: list = []
    labels = np.zeros((total, 3), dtype=np.float32)
    bounds = np.zeros((total, 2), dtype=np.float32)

    progress_file = _progress_path(cache, split)
    resuming = resume and progress_file.exists()
    if resuming:
        # `np.load` sur un .npz rend un objet paresseux qui garde le fichier ouvert.
        # Sous Windows, un fichier ouvert ne peut pas être supprimé : sans ce `with`,
        # le nettoyage de fin d'encodage échoue sur `PermissionError [WinError 32]` et
        # fait planter une exécution qui avait pourtant abouti.
        with np.load(progress_file, allow_pickle=True) as state:
            cursor = int(state["cursor"])
            consumed = int(state["consumed"])
            rejected = int(state["rejected"])
            saved_labels = np.array(state["labels"])
            saved_bounds = np.array(state["bounds"])
            saved_keys = state["region_keys"].tolist()
            saved_values = state["region_values"].tolist()
            saved_iqr = state["ita_iqr"].tolist()
        if saved_labels.shape == labels.shape:
            labels, bounds = saved_labels, saved_bounds
            region_sources = Counter(dict(zip(saved_keys, saved_values)))
            ita_iqr_values = saved_iqr
            print(f"[encode] reprise à {consumed}/{total} images ({cursor} retenues)")
        else:
            # Le point de reprise vient d'un autre périmètre : on ne mélange pas.
            print(f"[encode] point de reprise incompatible ({saved_labels.shape}) — ignoré")
            cursor = consumed = rejected = 0
            filtered_out = 0
            unalignable = 0
            resuming = False
    elif resume:
        print("[encode] aucun point de reprise trouvé — démarrage à zéro")

    dataset = FairFaceImages(source, data.image_size, flip=False, align=data.align_faces)
    # Les enregistrements déjà traités sont sautés côté dataset plutôt que côté boucle :
    # les sauter dans la boucle imposerait de décoder puis jeter chaque image.
    if consumed > 0 or limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(consumed, total))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=data.encode_batch_size,
        shuffle=False,
        num_workers=data.num_workers,
        collate_fn=collate_uint8,
        pin_memory=False,
        persistent_workers=data.num_workers > 0,
    )

    tmp_base = _temp_memmap(
        cache / f".tmp_{split}_base.npy", (total, 4, size, size), np.float16, resuming
    )
    tmp_flip = (
        _temp_memmap(
            cache / f".tmp_{split}_flip.npy", (total, 4, size, size), np.float16, resuming
        )
        if data.hflip
        else None
    )

    progress = tqdm(loader, desc=f"encode/{split}", unit="batch", initial=0)
    for batch_number, batch in enumerate(progress, start=1):
        images = batch["image"]
        # Les images non alignables ont déjà été retirées par le collate : on les compte
        # ici pour ne pas les perdre du décompte `consumed`, dont dépend la reprise.
        unalignable += int(batch.get("n_dropped", 0))
        consumed += images.shape[0] + int(batch.get("n_dropped", 0))
        if images.shape[0] == 0:
            continue

        # --- filtre visage unique + netteté (F-D9), avant tout calcul coûteux ---
        if keep_mask is not None:
            selected = keep_mask[batch["index"]]
            if not selected.any():
                filtered_out += int(images.shape[0])
                continue
            filtered_out += int((~selected).sum())
            images = images[selected]
            # `n_dropped` est un scalaire, pas une colonne : l'indexer lèverait TypeError.
            batch = {
                k: (images if k == "image" else v if np.isscalar(v) else v[selected])
                for k, v in batch.items()
            }

        # --- ITA image par image, avec rejet des mesures non fiables (F-D4) ---
        keep_rows, keep_ita = [], []
        for row in range(images.shape[0]):
            result = compute_ita(
                images[row],
                use_mediapipe=data.ita_use_mediapipe,
                min_pixels=data.ita_min_pixels,
            )
            region_sources[result.region_source] += 1
            if result.valid:
                keep_rows.append(row)
                keep_ita.append(result.ita)
                ita_iqr_values.append(result.ita_iqr)
            else:
                rejected += 1
        if not keep_rows:
            continue

        keep_rows = np.asarray(keep_rows)
        kept_images = images[keep_rows]
        count = len(keep_rows)
        end = cursor + count

        # --- encodage VAE, original puis miroir ---
        tmp_base[cursor:end] = (
            encode_images(vae, kept_images, data.vae_scale, device).cpu().numpy().astype(np.float16)
        )
        if tmp_flip is not None:
            mirrored = np.ascontiguousarray(kept_images[:, :, ::-1, :])
            tmp_flip[cursor:end] = (
                encode_images(vae, mirrored, data.vae_scale, device)
                .cpu()
                .numpy()
                .astype(np.float16)
            )

        bounds[cursor:end, 0] = batch["age_low"][keep_rows]
        bounds[cursor:end, 1] = batch["age_high"][keep_rows]
        labels[cursor:end, 0] = age_bin_centers(bounds[cursor:end])
        labels[cursor:end, 1] = batch["gender"][keep_rows].astype(np.float32)
        labels[cursor:end, 2] = np.asarray(keep_ita, dtype=np.float32)
        cursor = end

        if batch_number % checkpoint_every == 0:
            tmp_base.flush()
            if tmp_flip is not None:
                tmp_flip.flush()
            _save_progress(cache, split, cursor, consumed, labels, bounds,
                           rejected, region_sources, ita_iqr_values)

    n_base = cursor
    if n_base == 0:
        raise RuntimeError("Aucune image n'a survécu au calcul d'ITA — vérifier le masque de peau.")

    # --- compaction : [originaux ..., miroirs ...] ------------------------------------
    factor = 2 if tmp_flip is not None else 1
    final = np.lib.format.open_memmap(
        latents_path, mode="w+", dtype=np.float16, shape=(n_base * factor, 4, size, size)
    )
    final[:n_base] = tmp_base[:n_base]
    if tmp_flip is not None:
        final[n_base:] = tmp_flip[:n_base]
    final.flush()
    del final, tmp_base
    if tmp_flip is not None:
        del tmp_flip

    final_labels = np.concatenate([labels[:n_base]] * factor, axis=0)
    final_bounds = np.concatenate([bounds[:n_base]] * factor, axis=0)
    np.save(data.labels_path(split), final_labels)
    np.save(data.age_bounds_path(split), final_bounds)

    for name in (f".tmp_{split}_base.npy", f".tmp_{split}_flip.npy",
                 f".tmp_{split}_progress.npz", f".tmp_{split}_progress.npz.part"):
        (cache / name).unlink(missing_ok=True)

    ita_values = final_labels[:, 2]
    meta = {
        "split": split,
        "source": source.origin,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config_hash": cfg.hash(),
        "n_base": int(n_base),
        "n_total": int(n_base * factor),
        "hflip": bool(data.hflip),
        "image_size": data.image_size,
        "latent_size": size,
        "vae_id": data.vae_id,
        "vae_scale": data.vae_scale,
        "vae_encode_mode": "latent_dist.mean",
        "age_range": [float(cfg.data.age_min), float(cfg.data.age_max)],
        "age_bins_kept": sorted({r.age_bin for r in source.records}),
        "drop_partial_age_bins": data.drop_partial_age_bins,
        "effective_age_range": [
            float(final_bounds[:, 0].min()),
            float(final_bounds[:, 1].max()),
        ],
        "resumed": bool(resuming),
        "ita_rejected": int(rejected),
        "ita_reject_rate": float(rejected / max(total, 1)),
        # F-D9. Compté séparément du rejet ITA : les deux filtres répondent à des
        # questions différentes et le rapport doit pouvoir les citer distinctement.
        "face_filtered_out": int(filtered_out),
        "face_filter_applied": keep_mask is not None,
        # F-D10 : images où aucun visage exploitable n'a été trouvé, donc non alignables.
        # Elles sont écartées plutôt que redimensionnées sans alignement — mélanger des
        # images alignées et non alignées rétablirait la variance géométrique qu'on retire.
        "align_faces": bool(data.align_faces),
        "unalignable_dropped": int(unalignable),
        "ita_region_sources": dict(region_sources),
        "ita_mean_within_image_iqr": float(np.mean(ita_iqr_values)) if ita_iqr_values else None,
        "ita_stats": {
            "mean": float(ita_values.mean()),
            "std": float(ita_values.std()),
            "min": float(ita_values.min()),
            "p05": float(np.percentile(ita_values, 5)),
            "p95": float(np.percentile(ita_values, 95)),
            "max": float(ita_values.max()),
        },
        "gender_balance": {
            "male": int((final_labels[:, 1] == 0).sum()),
            "female": int((final_labels[:, 1] == 1).sum()),
        },
        "size_mb": round(latents_path.stat().st_size / 1e6, 1),
    }
    data.meta_path(split).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[encode] {meta['n_total']} latents → {latents_path} ({meta['size_mb']} Mo)")
    print(f"[encode] ITA : moyenne {meta['ita_stats']['mean']:.1f}°, "
          f"écart-type {meta['ita_stats']['std']:.1f}°, rejet {100 * meta['ita_reject_rate']:.2f} %")
    return meta


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Pré-encodage VAE de FairFace (F-D5)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reprend un encodage interrompu à partir de son dernier point de reprise.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=50,
                        help="Fréquence des points de reprise, en lots.")
    parser.add_argument(
        "--check-vae",
        action="store_true",
        help="N'encode rien : produit uniquement le rapport de reconstruction H3.",
    )
    parser.add_argument(
        "--keep-mask", default=None,
        help="Masque booléen .npy produit par `facedit.data.filter_faces` (F-D9). "
             "Indexé sur le dataset source ; appliqué avant l'ITA et avant le VAE.",
    )
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    from facedit.utils.config import load_config
    from facedit.utils.repro import seed_everything

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg.seed)

    if args.check_vae:
        report = vae_reconstruction_report(cfg, split=args.split)
        out = cfg.data.cache / f"vae_reconstruction_{cfg.data.image_size}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    mask = None
    if args.keep_mask:
        mask = np.load(args.keep_mask)
        if mask.dtype != bool:
            raise ValueError(f"{args.keep_mask} doit contenir un masque booléen.")
        print(f"[encode] masque F-D9 : {int(mask.sum())}/{mask.size} conservées "
              f"({mask.mean() * 100:.1f} %)")

    encode_split(cfg, args.split, args.limit, args.overwrite, args.resume,
                 args.checkpoint_every, keep_mask=mask)


if __name__ == "__main__":
    main()
