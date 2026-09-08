"""Filtrage du jeu source : un seul visage, suffisamment net — F-D9.

Motivation, constatée sur les générations du palier final. FairFace est un jeu « dans la
nature » : les vignettes contiennent parfois deux personnes serrées dans le cadre (un
couple qui s'embrasse, une main sur une épaule), parfois un profil bougé au point que les
traits sont illisibles. Le modèle apprend ces modes parce qu'ils sont dans les données, et
les restitue fidèlement — d'où les visages dédoublés observés à l'échantillonnage. Ce n'est
pas un artefact du générateur, c'est le jeu de données qui remonte à la surface.

Deux critères, tous deux mesurés et journalisés plutôt que devinés :

* **unicité** — MediaPipe FaceDetection ne doit remonter qu'un seul visage au-dessus de
  `min_detection_confidence`. C'est ce critère qui supprime le mode « deux visages ».
* **netteté** — variance du laplacien sur la vignette 128×128, comparée à un quantile de
  la distribution du jeu lui-même. Le seuil est donc relatif : on écarte le bas de la
  distribution observée, on n'impose pas une valeur absolue arbitraire qui dépendrait de
  la résolution ou du contraste du corpus.

Le masque produit est indexé sur les images **source** (avant miroir). Le cache de latents
range les miroirs à la suite (`[base_0..base_{n-1}, flip_0..flip_{n-1}]`), donc le même
masque appliqué aux deux moitiés suffit : aucun ré-encodage VAE n'est nécessaire.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from facedit.data.fairface import FairFaceImages, open_fairface, to_uint8
from facedit.utils.config import Config, load_config

# Noyau laplacien 4-connexe. La variance de la réponse est le proxy de netteté usuel
# (Pech-Pacheco et al., 2000) : peu coûteux, sans apprentissage, et monotone en flou.
_LAPLACIAN = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)


def laplacian_variance(image_uint8: np.ndarray) -> float:
    """Variance de la réponse laplacienne sur la luminance — proxy de netteté."""
    gray = image_uint8.astype(np.float32).mean(axis=-1)
    # Convolution 'valid' écrite à la main : évite une dépendance à scipy pour 9 termes.
    response = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
        - 4.0 * gray[1:-1, 1:-1]
    )
    return float(response.var())


_DETECTOR_STATE: Dict[str, object] = {}


def _get_detector(min_confidence: float):
    """Instancie FaceDetection une seule fois. Renvoie None si MediaPipe est indisponible."""
    if "detector" not in _DETECTOR_STATE:
        try:
            import mediapipe as mp

            _DETECTOR_STATE["detector"] = mp.solutions.face_detection.FaceDetection(
                # model_selection=0 (« short range », sujet à moins de 2 m). Mesuré sur
                # 1000 vignettes FairFace : le modèle 0 trouve un visage dans 92.7 % des
                # cas contre 38.5 % pour le modèle 1. Le modèle « full range » est conçu
                # pour des visages petits dans une scène large et échoue précisément sur
                # ce que FairFace contient — des visages recadrés serré qui remplissent le
                # cadre. Avec le modèle 1, le critère d'unicité ne rejetait plus que 57
                # images sur 64 599 : il était inerte, pas sélectif.
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        except Exception:  # pragma: no cover - dépend de l'environnement
            _DETECTOR_STATE["detector"] = None
    return _DETECTOR_STATE["detector"]


def count_faces(image_uint8: np.ndarray, min_confidence: float) -> Optional[int]:
    """Nombre de visages détectés, ou None si le détecteur est indisponible."""
    detector = _get_detector(min_confidence)
    if detector is None:
        return None
    result = detector.process(np.ascontiguousarray(image_uint8))
    return 0 if not result.detections else len(result.detections)


def detect_faces_and_yaw(
    image_uint8: np.ndarray, min_confidence: float
) -> Tuple[Optional[int], float]:
    """Renvoie (nombre de visages, |lacet| normalisé) en une seule passe de détection.

    Le lacet (rotation gauche-droite de la tête) est estimé à partir des points-clés que
    MediaPipe fournit déjà avec la boîte : on mesure de combien le nez s'écarte du milieu
    des deux yeux, rapporté à l'écartement oculaire. La grandeur est sans dimension, donc
    insensible à la taille du visage dans le cadre.

        0.0  ─ nez centré entre les yeux, visage de face
        0.25 ─ trois quarts marqué
        0.5+ ─ profil

    Ce n'est pas une estimation de pose 3D et cela ne prétend pas l'être ; c'est un
    ordonnancement monotone suffisant pour trier. Le calcul est gratuit : le détecteur
    tourne de toute façon pour le critère d'unicité.
    """
    detector = _get_detector(min_confidence)
    if detector is None:
        return None, 0.0
    result = detector.process(np.ascontiguousarray(image_uint8))
    if not result.detections:
        return 0, 0.0
    points = result.detections[0].location_data.relative_keypoints
    eye_gap = abs(points[1].x - points[0].x)
    if eye_gap < 1e-3:
        # Yeux confondus : le visage est vu de si loin de profil que la mesure n'a plus de
        # sens. On renvoie une valeur franchement au-dessus de tout seuil raisonnable
        # plutôt qu'un 0 qui le ferait passer pour frontal.
        return len(result.detections), 1.0
    nose_offset = points[2].x - (points[0].x + points[1].x) / 2.0
    return len(result.detections), abs(nose_offset / eye_gap)


def build_keep_mask(
    cfg: Config,
    split: str = "train",
    limit: Optional[int] = None,
    sharpness_quantile: float = 0.10,
    min_confidence: float = 0.5,
    max_yaw: Optional[float] = None,
    indices: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Calcule le masque de conservation sur les images source du split.

    Note sur `min_confidence` : mesuré sur 1000 vignettes, faire varier ce seuil de 0.10 à
    0.50 ne change aucun comptage (92.7 % à un visage, 0.8 % à deux ou plus, invariants).
    MediaPipe applique sa propre porte de score en amont dans cette version. Le paramètre
    est conservé pour l'explicitation et la reproductibilité, mais il ne faut pas compter
    sur lui pour régler la sévérité du filtre.

    `sharpness_quantile` est la fraction basse de la distribution de netteté écartée. À
    0.10, on retire le décile le plus flou. Le seuil absolu correspondant est journalisé
    pour que le rapport puisse citer une valeur, pas seulement un quantile.

    `max_yaw` (None = critère désactivé) écarte les têtes trop tournées. Justification
    mesurée sur le modèle 120k pas, à effectif de référence strictement égal (120 contre
    120, moyenné sur 5 tirages) : FID 98.5 ± 2.1 sur les visages frontaux contre
    126.7 ± 2.6 sur les visages tournés. La comparaison oppose des générations à des
    images réelles de MÊME pose : ce n'est donc pas « les profils sont durs à imiter »,
    c'est bien le modèle qui les rend moins bien. À 0.35 le critère ne retire que ~7 % du
    jeu, mesuré sur 800 vignettes.

    Attention à ne pas confondre ce seuil avec une censure de la diversité : on retire des
    POSES, pas des personnes. Aucune corrélation avec l'âge, le genre ou le teint n'est
    attendue, et le rapport de sortie journalise la composition avant/après pour qu'on
    puisse le vérifier plutôt que le supposer.
    """
    images = FairFaceImages(open_fairface(cfg.data, split), cfg.data.image_size, flip=False)
    full_size = len(images) if limit is None else min(limit, len(images))
    scan = np.arange(full_size) if indices is None else np.asarray(indices)
    total = len(scan)

    sharpness = np.zeros(total, dtype=np.float32)
    faces = np.full(total, -1, dtype=np.int16)
    yaws = np.zeros(total, dtype=np.float32)

    for position, index in enumerate(tqdm(scan, desc=f"filtre {split}")):
        index = int(index)
        array = np.asarray(images[index]["image"])
        if array.dtype != np.uint8:
            array = to_uint8(array)
        sharpness[position] = laplacian_variance(array)
        detected, yaw_value = detect_faces_and_yaw(array, min_confidence)
        faces[position] = -1 if detected is None else detected
        yaws[position] = yaw_value

    detector_available = bool((faces >= 0).any())
    # Un visage exactement. Les vignettes où le détecteur ne trouve rien sont conservées :
    # FairFace est cadré sur un visage par construction, une non-détection signale surtout
    # un cas difficile (forte occlusion, profil extrême) et non l'absence de sujet. Les
    # écarter reviendrait à filtrer sur la difficulté, ce qui biaiserait le jeu.
    unique = (faces <= 1) if detector_available else np.ones(total, bool)

    threshold = float(np.quantile(sharpness, sharpness_quantile))
    sharp = sharpness >= threshold

    # Les vignettes sans détection ont un lacet inconnu (0.0 par défaut) : on ne peut pas
    # les juger sur ce critère, donc on ne les écarte pas pour cette raison. Le critère de
    # netteté, lui, s'applique à toutes.
    if max_yaw is None:
        frontal = np.ones(total, bool)
    else:
        frontal = (yaws <= max_yaw) | (faces == 0)

    keep = unique & sharp & frontal

    report = {
        "split": split,
        "total": int(total),
        "kept": int(keep.sum()),
        "kept_ratio": float(keep.mean()),
        "rejected_multiface": int((~unique).sum()),
        "rejected_blurry": int((unique & ~sharp).sum()),
        "detector_available": detector_available,
        "no_detection": int((faces == 0).sum()) if detector_available else None,
        "sharpness_quantile": sharpness_quantile,
        "sharpness_threshold": threshold,
        "sharpness_median_kept": float(np.median(sharpness[keep])) if keep.any() else None,
        "sharpness_median_rejected": (
            float(np.median(sharpness[~keep])) if (~keep).any() else None
        ),
        "rejected_turned": int((unique & sharp & ~frontal).sum()),
        "max_yaw": max_yaw,
        "yaw_median": float(np.median(yaws[faces > 0])) if (faces > 0).any() else None,
        "yaw_p90": float(np.percentile(yaws[faces > 0], 90)) if (faces > 0).any() else None,
        "min_detection_confidence": min_confidence,
        "image_size": int(cfg.data.image_size),
    }

    # Composition démographique avant / après. Un filtre de POSE ne doit pas déplacer la
    # répartition par genre, âge ou origine ; s'il le fait, c'est qu'il capture autre chose
    # que la pose et il faut le savoir avant d'entraîner dessus, pas après.
    records = [images.source.records[int(i)] for i in scan]
    report["composition"] = {
        "gender_female_before": float(np.mean([r.gender == 1 for r in records])),
        "gender_female_after": float(np.mean([r.gender == 1 for r in np.array(records)[keep]]))
        if keep.any() else None,
        "age_bins_before": _fraction_by(records, lambda r: r.age_bin),
        "age_bins_after": _fraction_by(list(np.array(records)[keep]), lambda r: r.age_bin),
        "race_before": _fraction_by(records, lambda r: int(r.race)),
        "race_after": _fraction_by(list(np.array(records)[keep]), lambda r: int(r.race)),
    }
    return {
        "keep": keep, "sharpness": sharpness, "faces": faces, "yaws": yaws,
        "indices": scan, "full_size": full_size, "report": report,
    }


def _fraction_by(records, key) -> Dict[str, float]:
    """Répartition en fractions selon une clé — pour comparer avant / après filtrage."""
    counts: Dict[str, int] = {}
    for record in records:
        label = str(key(record))
        counts[label] = counts.get(label, 0) + 1
    total = max(len(records), 1)
    return {k: round(v / total, 4) for k, v in sorted(counts.items())}


def mask_path(cfg: Config, split: str, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return Path(cfg.data.cache_dir) / f"{split}_{cfg.data.image_size}_keep{suffix}.npy"


def shard_path(cfg: Config, split: str, index: int, count: int) -> Path:
    return (
        Path(cfg.data.cache_dir)
        / f".filterraw_{split}_{cfg.data.image_size}_{index}of{count}.npz"
    )


def merge_shards(
    cfg: Config, split: str, count: int, sharpness_quantile: float, max_yaw: Optional[float]
) -> Dict[str, object]:
    """Assemble les mesures brutes des shards puis applique les seuils GLOBALEMENT.

    Le seuil de netteté est un quantile de la distribution complète. Le calculer shard par
    shard donnerait N seuils légèrement différents et un filtre qui dépend du découpage —
    donc irreproductible dès qu'on change le nombre de processus. Les shards ne remontent
    donc que des mesures brutes ; la décision est prise ici, une seule fois.
    """
    sharpness = faces = yaws = None
    for index in range(count):
        path = shard_path(cfg, split, index, count)
        if not path.exists():
            raise FileNotFoundError(f"shard manquant : {path}")
        with np.load(path) as payload:
            if sharpness is None:
                size = int(payload["full_size"])
                sharpness = np.zeros(size, np.float32)
                faces = np.full(size, -1, np.int16)
                yaws = np.zeros(size, np.float32)
            where = payload["indices"]
            sharpness[where] = payload["sharpness"]
            faces[where] = payload["faces"]
            yaws[where] = payload["yaws"]

    unique = faces <= 1
    threshold = float(np.quantile(sharpness, sharpness_quantile))
    sharp = sharpness >= threshold
    frontal = np.ones(len(sharpness), bool) if max_yaw is None else (yaws <= max_yaw) | (faces == 0)
    keep = unique & sharp & frontal

    return {
        "keep": keep,
        "report": {
            "split": split, "total": int(len(keep)), "kept": int(keep.sum()),
            "kept_ratio": float(keep.mean()),
            "rejected_multiface": int((~unique).sum()),
            "rejected_blurry": int((unique & ~sharp).sum()),
            "rejected_turned": int((unique & sharp & ~frontal).sum()),
            "sharpness_quantile": sharpness_quantile,
            "sharpness_threshold": threshold,
            "max_yaw": max_yaw,
            "yaw_median": float(np.median(yaws[faces > 0])),
            "yaw_p90": float(np.percentile(yaws[faces > 0], 90)),
            "shards": count,
            "image_size": int(cfg.data.image_size),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filtre visage unique + netteté (F-D9).")
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sharpness-quantile", type=float, default=0.10)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--max-yaw", type=float, default=None,
        help="Seuil de lacet au-delà duquel la tête est jugée trop tournée (ex. 0.35). "
             "Non fourni : critère de pose désactivé.",
    )
    parser.add_argument("--shard", default=None, help="i/n : ne traite que la tranche i sur n.")
    parser.add_argument("--merge", type=int, default=None, help="Assemble n shards.")
    parser.add_argument("--tag", default="", help="Suffixe du fichier de masque produit.")
    parser.add_argument("overrides", nargs="*", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)

    if args.merge:
        merged = merge_shards(
            cfg, args.split, args.merge, args.sharpness_quantile, args.max_yaw
        )
        out = mask_path(cfg, args.split, args.tag)
        np.save(out, merged["keep"])
        out.with_name(out.stem + "_report.json").write_text(
            json.dumps(merged["report"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        r = merged["report"]
        print(f"[merge] {r['kept']}/{r['total']} conservées ({r['kept_ratio']*100:.1f} %)")
        print(f"        multi-visages {r['rejected_multiface']} · flou {r['rejected_blurry']} "
              f"· pose tournée {r['rejected_turned']}")
        print(f"[merge] masque -> {out}")
        return

    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        images_total = args.limit
        from facedit.data.fairface import open_fairface as _open
        full = len(_open(cfg.data, args.split).records)
        if images_total is not None:
            full = min(full, images_total)
        mine = np.arange(index, full, count)
        result = build_keep_mask(
            cfg, args.split, args.limit, args.sharpness_quantile, args.min_confidence,
            args.max_yaw, indices=mine,
        )
        path = shard_path(cfg, args.split, index, count)
        np.savez(
            path, indices=mine, sharpness=result["sharpness"], faces=result["faces"],
            yaws=result["yaws"], full_size=full,
        )
        print(f"[shard {index}/{count}] {len(mine)} images -> {path}")
        return
    result = build_keep_mask(
        cfg, args.split, args.limit, args.sharpness_quantile, args.min_confidence,
        args.max_yaw,
    )

    out = mask_path(cfg, args.split, args.tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, result["keep"])
    report_path = out.with_name(out.stem + "_report.json")
    report_path.write_text(
        json.dumps(result["report"], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = result["report"]
    print(f"[filtre] {report['kept']}/{report['total']} conservées "
          f"({report['kept_ratio'] * 100:.1f} %)")
    print(f"         rejet multi-visages : {report['rejected_multiface']}")
    print(f"         rejet flou          : {report['rejected_blurry']} "
          f"(seuil laplacien {report['sharpness_threshold']:.1f})")
    print(f"         rejet pose tournee  : {report['rejected_turned']} "
          f"(seuil lacet {report['max_yaw']})")
    composition = report["composition"]
    print(f"[filtre] part de femmes {composition['gender_female_before']:.3f} -> "
          f"{composition['gender_female_after']:.3f}  (doit rester stable)")
    print(f"[filtre] masque -> {out}")


if __name__ == "__main__":
    main()
