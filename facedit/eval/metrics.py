"""Métriques de distribution — F-E1, F-E5, F-E8, F-E9.

FID via `clean-fid` (F-E1) : l'implémentation historique de `pytorch-fid` dépend du
redimensionnement PIL de la bibliothèque appelante, ce qui rend les valeurs
incomparables d'un dépôt à l'autre. `clean-fid` fige le redimensionnement — c'est la
raison pour laquelle le PRD l'impose nommément.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Utilitaires
# --------------------------------------------------------------------------------------


def save_images_to_dir(images_uint8: np.ndarray, directory: str | Path, prefix: str = "img") -> Path:
    """Écrit un lot en PNG. Le FID passe par le disque, `clean-fid` ne prend que des dossiers."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images_uint8):
        Image.fromarray(image).save(directory / f"{prefix}_{index:06d}.png")
    return directory


def frechet_distance(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """Distance de Fréchet entre deux nuages gaussiens ajustés. Utilisée par F-E8.

    `sqrtm` de SciPy renvoie une petite partie imaginaire sur des matrices mal
    conditionnées ; on la vérifie avant de la jeter plutôt que de la jeter en silence.
    """
    from scipy import linalg

    mu_a, mu_b = features_a.mean(axis=0), features_b.mean(axis=0)
    sigma_a = np.cov(features_a, rowvar=False)
    sigma_b = np.cov(features_b, rowvar=False)

    diff = mu_a - mu_b
    covmean, _ = linalg.sqrtm(sigma_a.dot(sigma_b), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("sqrtm a produit une composante imaginaire non négligeable")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma_a) + np.trace(sigma_b) - 2.0 * np.trace(covmean))


# --------------------------------------------------------------------------------------
# FID — F-E1
# --------------------------------------------------------------------------------------


def compute_fid(
    generated_dir: str | Path, reference_dir: str | Path, batch_size: int = 32
) -> Optional[float]:
    """FID `clean-fid` entre deux dossiers d'images."""
    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        print("[fid] clean-fid absent — FID ignoré")
        return None

    num_generated = len(list(Path(generated_dir).glob("*.png")))
    num_reference = len(list(Path(reference_dir).glob("*.png")))
    if min(num_generated, num_reference) < 2:
        return None
    if min(num_generated, num_reference) < 2048:
        # Le FID est biaisé à la hausse sur de petits échantillons, et son estimateur de
        # covariance est singulier en dessous de la dimension des features (2048).
        # On calcule quand même — la grille de sous-groupes (F-E7) n'a que 300 images par
        # cellule — mais la valeur doit être lue comme comparative, jamais absolue.
        print(
            f"[fid] échantillon réduit ({num_generated} vs {num_reference}) : "
            f"valeur biaisée, comparable entre cellules mais non absolue"
        )

    return float(
        cleanfid.compute_fid(
            str(generated_dir), str(reference_dir), mode="clean",
            batch_size=batch_size, verbose=False, num_workers=0,
        )
    )


def build_reference_set(
    cfg,
    num_images: int,
    out_dir: str | Path,
    split: str = "val",
    keep_mask: Optional[np.ndarray] = None,
) -> Path:
    """Extrait `num_images` images réelles du split de référence vers un dossier PNG.

    Elles servent de dénominateur au FID (F-E1 : « 10k générées vs 10k réelles du val
    set »). Le dossier est mis en cache : le régénérer à chaque évaluation coûterait
    plusieurs minutes et surtout changerait l'échantillon, donc la valeur du FID.

    `keep_mask` (F-D9) restreint le tirage aux images que le filtre visage-unique/netteté
    aurait conservées. Il sert à construire la **seconde** référence, celle qui correspond
    à la distribution que le modèle filtré vise réellement.

    Les deux références sont nécessaires et aucune ne suffit seule. Contre la référence
    non filtrée, on pénalise le modèle pour un écart qu'on a créé exprès en retirant les
    images floues du jeu d'entraînement. Contre la seule référence filtrée, on ne peut
    plus distinguer un gain du générateur d'un simple assouplissement du dénominateur —
    on aurait rendu la cible plus facile et appelé cela un progrès. L'écart entre les deux
    valeurs mesure l'effet du filtre, et rien d'autre.
    """
    from facedit.data.fairface import open_fairface, resize_uint8

    out_dir = Path(out_dir)
    existing = len(list(out_dir.glob("*.png"))) if out_dir.exists() else 0
    if existing >= num_images:
        return out_dir

    source = open_fairface(cfg.data, split)
    candidates = np.arange(len(source))
    if keep_mask is not None:
        if len(keep_mask) != len(source):
            raise ValueError(
                f"masque de longueur {len(keep_mask)} pour un split de {len(source)} "
                "images : le masque ne vient pas du même split."
            )
        candidates = candidates[keep_mask]
    count = min(num_images, len(candidates))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(candidates, size=count, replace=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for position, index in enumerate(tqdm(indices, desc=f"référence FID ({split})")):
        image = resize_uint8(source.load_image(int(index)), cfg.data.image_size)
        Image.fromarray(image).save(out_dir / f"real_{position:06d}.png")
    return out_dir


# --------------------------------------------------------------------------------------
# Diversité — F-E5
# --------------------------------------------------------------------------------------


def lpips_diversity(
    generator,
    conditions: Sequence[Tuple[float, int, float]],
    num_pairs: int = 500,
    seed: int = 0,
    device: str = "cuda",
    sample_cfg=None,
) -> Dict[str, float]:
    """LPIPS moyen entre paires d'images générées **sous condition identique** (F-E5).

    C'est la mesure d'effondrement de mode qui compte pour ce projet : un modèle qui
    obéit parfaitement aux attributs mais rend le même visage pour toute condition
    donnée serait excellent sur F-E2/F-E3 et inutile. Seuil d'échec du §8.1 : < 0.20.

    Le CFG dégrade mécaniquement cette métrique — c'est le compromis que le balayage
    F-E6 met en évidence.
    """
    import lpips

    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    pairs_per_condition = max(1, num_pairs // max(len(conditions), 1))
    distances: List[float] = []

    for condition_index, (age, gender, ita) in enumerate(
        tqdm(conditions, desc="diversité LPIPS")
    ):
        # Deux tirages de bruit distincts, même condition : la variabilité mesurée est
        # exactement celle que le modèle produit à condition fixée.
        batch = pairs_per_condition * 2
        images = generator.generate(
            age, gender, ita, batch_size=batch,
            sample_cfg=sample_cfg, seed=seed + condition_index * 1000,
        )["images"]

        from facedit.data.fairface import to_signed_float

        tensor = torch.from_numpy(to_signed_float(images)).to(device)
        with torch.no_grad():
            values = loss_fn(tensor[0::2], tensor[1::2])
        distances.extend(values.flatten().cpu().numpy().tolist())

    distances_array = np.asarray(distances)
    return {
        "lpips_mean": float(distances_array.mean()),
        "lpips_std": float(distances_array.std()),
        "lpips_p05": float(np.percentile(distances_array, 5)),
        "n_pairs": int(len(distances_array)),
        "target": "> 0.35 (§8.1)",
        "fail_threshold": "< 0.20 (§8.1)",
    }


# --------------------------------------------------------------------------------------
# FD-DINOv2 — F-E8
# --------------------------------------------------------------------------------------


def _dinov2_features(images_uint8: np.ndarray, device: str, batch_size: int = 32):
    """Features DINOv2 ViT-S/14. Renvoie None si le backbone n'est pas récupérable."""
    try:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)
    except Exception as exc:
        print(f"[fd-dinov2] backbone indisponible ({exc.__class__.__name__}) — métrique ignorée")
        return None

    model.to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    features = []
    with torch.no_grad():
        for start in range(0, len(images_uint8), batch_size):
            chunk = images_uint8[start : start + batch_size]
            tensor = torch.from_numpy(
                np.ascontiguousarray(chunk.transpose(0, 3, 1, 2))
            ).to(device).float() / 255.0
            # DINOv2 exige un côté multiple de 14 ; 224 est la résolution d'entraînement.
            tensor = torch.nn.functional.interpolate(
                tensor, size=(224, 224), mode="bicubic", align_corners=False
            )
            features.append(model((tensor - mean) / std).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(features)


def fd_dinov2(
    generated: np.ndarray, reference: np.ndarray, device: str = "cuda"
) -> Optional[float]:
    """Distance de Fréchet en espace DINOv2 (F-E8).

    Complément au FID, pas remplacement : les features InceptionV3 du FID sont
    entraînées sur ImageNet en classification supervisée et saturent sur un domaine
    mono-classe comme les visages. DINOv2, auto-supervisé, discrimine mieux à l'intérieur
    d'une classe unique — précisément notre cas.
    """
    features_generated = _dinov2_features(generated, device)
    if features_generated is None:
        return None
    features_reference = _dinov2_features(reference, device)
    if features_reference is None:
        return None
    return frechet_distance(features_generated, features_reference)


# --------------------------------------------------------------------------------------
# Mémorisation — F-E9, E-3
# --------------------------------------------------------------------------------------


def _nearest_neighbour_lpips(
    generator,
    query_latents: torch.Tensor,
    reference,
    flat_reference: torch.Tensor,
    loss_fn,
    top_k: int,
    device: str,
    exclude: Optional[np.ndarray] = None,
    desc: str = "plus proches voisins",
) -> Tuple[np.ndarray, np.ndarray]:
    """LPIPS au plus proche voisin, en deux temps.

    1. **filtrage** par similarité cosinus dans l'espace latent VAE déjà en cache — pas
       de décodage, pas de disque, une seule multiplication matricielle ;
    2. **vérification** par LPIPS sur les `top_k` candidats décodés, qui est la mesure
       perceptuelle réellement pertinente pour dire « c'est la même personne ».

    `exclude` donne, par requête, une liste d'indices de référence à ignorer. Elle est
    indispensable pour calibrer sur des images réelles : le cache contient chaque image
    **deux fois** (original et miroir, F-D6), donc le plus proche voisin d'une image
    réelle serait son propre reflet, à distance quasi nulle.
    """
    from facedit.data.fairface import to_signed_float
    from facedit.sample import decode

    count = query_latents.shape[0]
    flat_query = torch.nn.functional.normalize(query_latents.reshape(count, -1), dim=1)
    similarity = flat_query @ flat_reference.T

    if exclude is not None:
        rows = np.repeat(np.arange(count), exclude.shape[1])
        similarity[rows, exclude.reshape(-1)] = -2.0

    candidates = similarity.topk(top_k, dim=1).indices.cpu().numpy()

    distances_out, matched = [], []
    for row in tqdm(range(count), desc=desc, leave=False):
        neighbour_latents = torch.from_numpy(
            np.asarray(reference[candidates[row]], dtype=np.float32)
        ).to(device)
        neighbours = decode(generator, neighbour_latents)

        query_image = decode(generator, query_latents[row : row + 1])
        query = torch.from_numpy(to_signed_float(query_image)).to(device)
        others = torch.from_numpy(to_signed_float(neighbours)).to(device)
        with torch.no_grad():
            values = loss_fn(query.expand_as(others), others).flatten().cpu().numpy()
        best = int(values.argmin())
        distances_out.append(float(values[best]))
        matched.append(int(candidates[row][best]))

    return np.asarray(distances_out), np.asarray(matched)


def memorization_test(
    generator,
    images_uint8: np.ndarray,
    latents_path: str | Path,
    top_k: int = 16,
    device: str = "cuda",
    max_reference: int = 40_000,
    num_calibration: int = 128,
) -> Dict[str, object]:
    """Plus proche voisin des images générées dans le set d'entraînement (F-E9, E-3).

    **Le seuil est calibré, pas codé en dur.** Une première version comparait la distance
    LPIPS à une constante (0.15) : sur des images 32×32 elle déclarait 100 % des sorties
    « quasi-copies », parce que LPIPS se contracte quand la résolution baisse et que deux
    visages flous quelconques sont perceptuellement proches. Un seuil absolu mesure donc
    autant la résolution que la mémorisation.

    La référence correcte est le jeu lui-même : on mesure la distance au plus proche
    voisin pour des images **réelles** (en excluant l'image elle-même et son miroir), ce
    qui donne la distance typique entre deux personnes distinctes à cette résolution.
    Une image générée n'est suspecte que si elle est plus proche d'une image
    d'entraînement que ne le sont deux personnes différentes — concrètement, sous le 1ᵉʳ
    centile de la distribution de calibration.

    Ce que le test ne peut pas conclure : l'absence de mémorisation en général. Il ne
    porte que sur les images effectivement générées ici.
    """
    import lpips

    reference = np.load(latents_path, mmap_mode="r")
    total = min(reference.shape[0], max_reference)
    flat_reference = torch.from_numpy(
        np.asarray(reference[:total], dtype=np.float32).reshape(total, -1)
    ).to(device)
    flat_reference = torch.nn.functional.normalize(flat_reference, dim=1)

    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)

    # --- distribution de calibration, sur des images réelles ---------------------------
    # Le cache est bâti comme [originaux…, miroirs…] : la moitié `n_base` sépare les deux.
    n_base = total // 2
    rng = np.random.default_rng(0)
    probes = rng.choice(n_base, size=min(num_calibration, n_base), replace=False)
    probe_latents = torch.from_numpy(
        np.asarray(reference[probes], dtype=np.float32)
    ).to(device)
    # Exclure l'image elle-même et son miroir.
    exclusions = np.stack([probes, (probes + n_base) % max(total, 1)], axis=1)

    calibration, _ = _nearest_neighbour_lpips(
        generator, probe_latents, reference, flat_reference, loss_fn, top_k, device,
        exclude=exclusions, desc="calibration (réel↔réel)",
    )

    # --- distances des images générées --------------------------------------------------
    query_latents = torch.from_numpy(_encode_for_search(generator, images_uint8)).to(device)
    generated, matched = _nearest_neighbour_lpips(
        generator, query_latents, reference, flat_reference, loss_fn, top_k, device,
        desc="mémorisation (généré↔réel)",
    )

    threshold = float(np.percentile(calibration, 1))
    suspicious = int((generated < threshold).sum())

    return {
        "n_generated": int(len(images_uint8)),
        "n_reference": int(total),
        "calibration": {
            "n_probes": int(len(calibration)),
            "lpips_nn_mean": float(calibration.mean()),
            "lpips_nn_p01": threshold,
            "lpips_nn_min": float(calibration.min()),
            "meaning": "distance au plus proche voisin entre personnes réelles distinctes, "
                       "à cette résolution — l'étalon du test",
        },
        "generated": {
            "lpips_nn_mean": float(generated.mean()),
            "lpips_nn_min": float(generated.min()),
            "lpips_nn_p01": float(np.percentile(generated, 1)),
        },
        "threshold_used": threshold,
        "n_below_threshold": suspicious,
        "fraction_below_threshold": float(suspicious / max(len(generated), 1)),
        "closest_train_indices": matched[:32].tolist(),
        "verdict": (
            f"Aucune quasi-copie détectée : aucune image générée n'est plus proche d'une "
            f"image d'entraînement (min {generated.min():.3f}) que ne le sont deux "
            f"personnes réelles distinctes (1ᵉʳ centile {threshold:.3f})."
            if suspicious == 0
            else f"{suspicious} image(s) sous le 1ᵉʳ centile de la distribution réelle "
                 f"({threshold:.3f}) — à inspecter visuellement et à signaler dans la "
                 f"section éthique (E-3)."
        ),
    }


def _encode_for_search(generator, images_uint8: np.ndarray) -> np.ndarray:
    """Ré-encode les images générées en latents VAE, pour comparer dans le même espace."""
    from facedit.data.encode import encode_images

    outputs = []
    for start in range(0, len(images_uint8), 32):
        chunk = images_uint8[start : start + 32]
        outputs.append(
            encode_images(
                generator.vae, chunk, generator.cfg.data.vae_scale, generator.device
            ).cpu().numpy()
        )
    return np.concatenate(outputs)


class TemporaryImageDir:
    """Dossier temporaire d'images, nettoyé même en cas d'exception."""

    def __init__(self, images_uint8: np.ndarray, prefix: str = "facedit"):
        self.images = images_uint8
        self.prefix = prefix
        self.path: Optional[Path] = None

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        save_images_to_dir(self.images, self.path)
        return self.path

    def __exit__(self, *exc) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)
