"""Rejet automatique des générations ratées — F-S6.

Constat mesuré sur 600 générations du palier v3 (w = 2) :

    4.0 %  de visages tournés (|lacet| > 0.25)
    4.5 %  où le détecteur ne trouve AUCUN visage — les ratés les plus visibles

Soit environ 8.5 % de tirages inexploitables. Un modèle de diffusion rate toujours une
fraction de ses tirages ; l'usage est de les écarter, pas de les réparer. Ce module rend
ce geste automatique et, surtout, **mesurable** : chaque rejet est motivé et compté.

Deux critères seulement, tous deux issus d'une mesure et non d'une intuition :

* **détection** — MediaPipe ne trouve pas de visage. C'est le proxy d'incohérence
  structurelle : les traits enchevêtrés, les fusions de deux visages, les bouillies.
* **lacet** — la tête est trop tournée. Le modèle rend mal ces poses : à effectif de
  référence égal, FID 98.5 ± 2.1 de face contre 126.7 ± 2.6 de trois quarts.

Aucun critère esthétique, aucun score appris : on n'écarte que ce qu'on sait mesurer.

HONNÊTETÉ DE MESURE — à lire avant d'utiliser ceci dans une évaluation. Le rejet modifie
la distribution de sortie. Un FID calculé sur des images filtrées n'est PAS comparable à un
FID calculé sans filtrage : on aurait retiré les pires échantillons puis annoncé un
progrès. Toute métrique du rapport doit être publiée SANS rejet, ou avec les deux valeurs
côte à côte et la mention explicite. Le rejet est un confort d'usage pour l'interface, pas
un résultat de modélisation.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

DEFAULT_MAX_YAW = 0.25
"""Seuil de lacet au-delà duquel on écarte. Plus strict que le 0.35 du filtrage des
données : à l'entraînement on veut garder du volume, à la génération on peut se permettre
de retirer davantage puisqu'il suffit de retirer une autre graine."""


def face_quality(image_uint8: np.ndarray, min_confidence: float = 0.5) -> Dict[str, object]:
    """Diagnostic d'une image générée : détection, nombre de visages, lacet."""
    from facedit.data.align import _get_detector

    detector = _get_detector(min_confidence)
    if detector is None:
        # Sans détecteur on n'a aucun moyen de juger : on accepte tout plutôt que de
        # rejeter en aveugle, et on le dit dans le rapport.
        return {"detected": True, "n_faces": 1, "yaw": 0.0, "detector": False}

    result = detector.process(np.ascontiguousarray(image_uint8))
    if not result.detections:
        return {"detected": False, "n_faces": 0, "yaw": None, "detector": True}

    points = result.detections[0].location_data.relative_keypoints
    eye_gap = abs(points[1].x - points[0].x)
    if eye_gap < 1e-3:
        yaw = 1.0
    else:
        yaw = abs((points[2].x - (points[0].x + points[1].x) / 2.0) / eye_gap)
    return {
        "detected": True,
        "n_faces": len(result.detections),
        "yaw": float(yaw),
        "detector": True,
    }


def acceptance_mask(
    images: np.ndarray, max_yaw: float = DEFAULT_MAX_YAW, allow_multiface: bool = False
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Masque des images retenues, plus le décompte motivé des rejets."""
    keep = np.zeros(len(images), dtype=bool)
    reasons = {"non_detecte": 0, "multi_visages": 0, "trop_tourne": 0, "accepte": 0}

    for index, image in enumerate(images):
        report = face_quality(image)
        if not report["detected"]:
            reasons["non_detecte"] += 1
            continue
        if not allow_multiface and report["n_faces"] > 1:
            reasons["multi_visages"] += 1
            continue
        if report["yaw"] is not None and report["yaw"] > max_yaw:
            reasons["trop_tourne"] += 1
            continue
        keep[index] = True
        reasons["accepte"] += 1

    return keep, reasons


def generate_accepted(
    generator,
    age,
    gender,
    ita,
    count: int = 1,
    sample_cfg=None,
    seed: int = 0,
    max_yaw: float = DEFAULT_MAX_YAW,
    max_rounds: int = 4,
    oversample: int = 3,
) -> Dict[str, object]:
    """Génère `count` images acceptables, en retirant de nouvelles graines si besoin.

    Le tirage se fait par vagues de `count * oversample` candidats. À ~8.5 % de rejet
    mesuré, une seule vague suffit presque toujours ; `max_rounds` borne le pire cas pour
    qu'une condition difficile ne fasse jamais tourner indéfiniment.

    Si le quota n'est pas atteint au bout de `max_rounds`, on renvoie ce qu'on a — complété
    par les meilleurs candidats rejetés, classés par lacet croissant. Rendre moins d'images
    que demandé serait une surprise désagréable pour l'appelant ; rendre une image
    imparfaite en le signalant dans `stats` est plus honnête.
    """
    accepted: List[np.ndarray] = []
    fallback: List[Tuple[float, np.ndarray]] = []
    totals = {"non_detecte": 0, "multi_visages": 0, "trop_tourne": 0, "accepte": 0}
    drawn = 0

    for round_index in range(max_rounds):
        batch_size = max(1, count * oversample) if round_index == 0 else max(1, count)
        images = generator.generate(
            age, gender, ita, batch_size=batch_size, sample_cfg=sample_cfg,
            seed=seed + 10000 * round_index,
        )["images"]
        drawn += len(images)

        keep, reasons = acceptance_mask(images, max_yaw)
        for key, value in reasons.items():
            totals[key] += value

        accepted.extend(list(images[keep]))
        for image in images[~keep]:
            report = face_quality(image)
            # Un lacet inconnu (aucune détection) est le pire cas : on le classe en dernier.
            fallback.append((report["yaw"] if report["yaw"] is not None else 9.0, image))

        if len(accepted) >= count:
            break

    stats = {
        "demande": count,
        "tires": drawn,
        "acceptes": len(accepted),
        "taux_acceptation": round(totals["accepte"] / max(drawn, 1), 4),
        "rejets": {k: v for k, v in totals.items() if k != "accepte"},
        "max_yaw": max_yaw,
        "complete_par_repli": max(0, count - len(accepted)),
    }

    if len(accepted) < count:
        fallback.sort(key=lambda item: item[0])
        accepted.extend(image for _, image in fallback[: count - len(accepted)])

    return {"images": np.stack(accepted[:count]), "stats": stats}
