"""Matrice de fuite (entanglement) — F-E4, protocole §8.2.

Pour chaque attribut cible A parmi {âge, genre, teint} :

1. fixer 100 seeds et une condition de base (35 ans, femme, ITA 30°) ;
2. générer avec A à sa valeur minimale, puis à sa valeur maximale, les autres attributs
   **et la seed** constants ;
3. mesurer les trois attributs sur les deux séries ;
4. Δ_B = |moyenne(B_max) − moyenne(B_min)| pour chaque attribut B ;
5. normaliser chaque Δ_B par l'écart-type de B sur le set réel.

La diagonale doit être forte (l'attribut piloté bouge), les termes hors-diagonale
faibles (les autres ne bougent pas). C'est la mesure du problème n°2 du §1.1.

Le partage de la seed entre les deux séries est ce qui donne sa puissance au test : la
différence observée ne peut pas venir du bruit initial, seulement du changement de
condition. Sans cela, il faudrait des milliers d'échantillons pour distinguer une fuite
réelle de la variance d'échantillonnage.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from facedit.data.ita import compute_ita_batch

# Valeurs extrêmes utilisées pour chaque attribut piloté.
ATTRIBUTE_EXTREMES: Dict[str, Tuple[float, float]] = {
    "age": (22.0, 68.0),
    "gender": (0.0, 1.0),
    "ita": (-35.0, 55.0),
}

BASE_CONDITION = {"age": 35.0, "gender": 1, "ita": 30.0}


def _measure_series(
    generator, oracle, age, gender, ita, seeds: np.ndarray, batch_size: int, sample_cfg
) -> Dict[str, np.ndarray]:
    """Génère une série à condition constante et seeds imposées, puis mesure."""
    count = len(seeds)
    out = {
        "age": np.zeros(count, np.float32),
        "gender": np.zeros(count, np.float32),
        "ita": np.full(count, np.nan, np.float32),
        "valid": np.zeros(count, bool),
    }

    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        # Les seeds doivent être identiques entre la série min et la série max : on
        # génère donc une image par seed, en re-tirant le bruit seed par seed plutôt que
        # de laisser un lot partager un unique tirage.
        images = []
        for seed in seeds[start:stop]:
            images.append(
                generator.generate(
                    age, gender, ita, batch_size=1, sample_cfg=sample_cfg, seed=int(seed)
                )["images"][0]
            )
        images = np.stack(images)

        prediction = oracle.predict(images)
        out["age"][start:stop] = prediction["age"]
        # Probabilité de « femme » plutôt que la classe argmax : un déplacement de genre
        # de 0.30 à 0.45 est une fuite réelle qu'un argmax arrondirait à zéro.
        out["gender"][start:stop] = np.where(
            prediction["gender"] == 1, prediction["gender_conf"], 1.0 - prediction["gender_conf"]
        )
        # Même instrument que les labels d'entraînement (voir `generate_and_measure`) :
        # un ITA mesuré sur une autre région de peau se décale de ~20°.
        analytic, valid = compute_ita_batch(images, use_mediapipe=True)
        out["ita"][start:stop] = analytic
        out["valid"][start:stop] = valid

    return out


def real_attribute_std(cfg) -> Dict[str, float]:
    """Écarts-types des trois attributs sur le set réel — dénominateurs de l'étape 5."""
    labels = np.load(cfg.data.labels_path("train"))
    return {
        "age": float(labels[:, 0].std()),
        # Écart-type d'une Bernoulli(p) : la « distance » entre homme et femme vaut
        # environ 0.5 dans l'unité de la probabilité prédite.
        "gender": float(labels[:, 1].std()),
        "ita": float(labels[:, 2].std()),
    }


def leakage_matrix(
    generator,
    oracle,
    cfg,
    num_seeds: int = 100,
    seed: int = 0,
    sample_cfg=None,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Matrice 3×3 normalisée (F-E4). Lignes = attribut piloté, colonnes = attribut mesuré."""
    seeds = np.arange(seed, seed + num_seeds, dtype=np.int64)
    stds = real_attribute_std(cfg)
    names = ("age", "gender", "ita")

    matrix = np.zeros((3, 3), dtype=np.float64)
    raw = np.zeros((3, 3), dtype=np.float64)
    details: List[Dict[str, object]] = []

    for row, driven in enumerate(tqdm(names, desc="matrice de fuite")):
        low_value, high_value = ATTRIBUTE_EXTREMES[driven]

        low_condition = dict(BASE_CONDITION)
        high_condition = dict(BASE_CONDITION)
        low_condition[driven] = int(low_value) if driven == "gender" else low_value
        high_condition[driven] = int(high_value) if driven == "gender" else high_value

        series_low = _measure_series(
            generator, oracle, low_condition["age"], low_condition["gender"],
            low_condition["ita"], seeds, batch_size, sample_cfg,
        )
        series_high = _measure_series(
            generator, oracle, high_condition["age"], high_condition["gender"],
            high_condition["ita"], seeds, batch_size, sample_cfg,
        )

        for col, measured in enumerate(names):
            if measured == "ita":
                mask = series_low["valid"] & series_high["valid"]
                if mask.sum() < 5:
                    matrix[row, col] = np.nan
                    raw[row, col] = np.nan
                    continue
                delta = abs(
                    float(series_high["ita"][mask].mean()) - float(series_low["ita"][mask].mean())
                )
            else:
                delta = abs(
                    float(series_high[measured].mean()) - float(series_low[measured].mean())
                )
            raw[row, col] = delta
            matrix[row, col] = delta / max(stds[measured], 1e-6)

        details.append({
            "driven": driven,
            "low": low_condition,
            "high": high_condition,
            "measured_low": {
                "age": float(series_low["age"].mean()),
                "gender_p_female": float(series_low["gender"].mean()),
                "ita": float(np.nanmean(series_low["ita"])),
            },
            "measured_high": {
                "age": float(series_high["age"].mean()),
                "gender_p_female": float(series_high["gender"].mean()),
                "ita": float(np.nanmean(series_high["ita"])),
            },
        })

    diagonal = np.array([matrix[i, i] for i in range(3)])
    off_diagonal = matrix[~np.eye(3, dtype=bool)]

    return {
        "attributes": list(names),
        "matrix_normalized": matrix.tolist(),
        "matrix_raw": raw.tolist(),
        "raw_units": {"age": "années", "gender": "probabilité de femme", "ita": "degrés"},
        "real_std": stds,
        "num_seeds": num_seeds,
        "extremes": {k: list(v) for k, v in ATTRIBUTE_EXTREMES.items()},
        "base_condition": BASE_CONDITION,
        "details": details,
        "summary": {
            "diagonal_mean": float(np.nanmean(diagonal)),
            "diagonal_min": float(np.nanmin(diagonal)),
            "off_diagonal_mean": float(np.nanmean(off_diagonal)),
            "off_diagonal_max": float(np.nanmax(off_diagonal)),
            # Rapport diagonale / hors-diagonale : un unique nombre pour résumer le
            # désenchevêtrement. > 5 est bon, < 2 signifie que les axes ne sont pas
            # séparés et que le contrôle « par attribut » est une illusion.
            "disentanglement_ratio": float(
                np.nanmean(diagonal) / max(np.nanmean(off_diagonal), 1e-6)
            ),
        },
        "reading": (
            "Ligne = attribut piloté, colonne = attribut mesuré. Chaque case est le "
            "déplacement de la moyenne, en écarts-types du set réel, quand l'attribut de "
            "la ligne passe de son minimum à son maximum à seed constante. La diagonale "
            "doit être grande, les termes hors-diagonale proches de zéro."
        ),
    }
