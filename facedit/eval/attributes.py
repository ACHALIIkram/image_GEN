"""Fidélité aux attributs — F-E2, F-E3, F-E7.

Distinction méthodologique importante entre les deux instruments de mesure :

- l'**oracle** (ResNet-18) mesure l'âge et le genre. Il est faillible et sa propre erreur
  est reportée en regard de chaque valeur (F-O4) ;
- le **calcul analytique d'ITA** mesure le teint, sans réseau (F-E3 : « ITA analytique,
  sans oracle »). Il n'a pas d'erreur d'apprentissage, seulement une sensibilité à
  l'éclairage quantifiée par `ita_dispersion_report`.

C'est délibéré : si les trois attributs étaient mesurés par le même réseau entraîné sur
les mêmes données que le générateur, un biais partagé rendrait le système d'apparence
obéissante sans l'être. L'ITA est le point d'ancrage indépendant du protocole.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from facedit.data.ita import compute_ita_batch, ita_to_monk


# --------------------------------------------------------------------------------------
# Tirage de conditions
# --------------------------------------------------------------------------------------


def sample_conditions(
    count: int,
    age_range: Tuple[float, float] = (20.0, 70.0),
    ita_range: Tuple[float, float] = (-40.0, 60.0),
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Conditions tirées **uniformément**, pas selon la distribution du jeu réel.

    Un tirage suivant la distribution de FairFace sur-représenterait les 20-40 ans et les
    ITA médians, c'est-à-dire exactement le régime où le modèle est le plus à l'aise. La
    MAE moyenne y paraîtrait excellente en masquant l'effondrement aux extrêmes — que la
    grille de sous-groupes (F-E7) est justement chargée d'exposer.
    """
    rng = np.random.default_rng(seed)
    return {
        "age": rng.uniform(age_range[0], age_range[1], count).astype(np.float32),
        "gender": rng.integers(0, 2, count).astype(np.int64),
        "ita": rng.uniform(ita_range[0], ita_range[1], count).astype(np.float32),
    }


# --------------------------------------------------------------------------------------
# Génération + mesure
# --------------------------------------------------------------------------------------


def generate_and_measure(
    generator,
    oracle,
    conditions: Dict[str, np.ndarray],
    batch_size: int = 64,
    seed: int = 0,
    sample_cfg=None,
    use_mediapipe: bool = True,
    keep_images: bool = False,
    desc: str = "génération",
) -> Dict[str, np.ndarray]:
    """Génère selon `conditions` puis mesure les trois attributs obtenus.

    **L'ITA des images générées doit être mesuré exactement comme celui des labels
    d'entraînement.** C'est la leçon d'une erreur commise puis corrigée sur ce projet :
    mesurer les images générées avec le gabarit géométrique alors que les labels avaient
    été construits avec MediaPipe faisait apparaître un ΔITA de 18.4°, dont −17.5° de
    biais pur. Vérification faite sur les *mêmes* images réelles, les deux régions de peau
    donnent des ITA médians de +1.8° et −21.1° : l'écart mesuré n'était pas de la
    désobéissance du modèle, c'était un désaccord d'instrument. Avec l'instrument
    cohérent, ΔITA vaut 6.1°.

    Le risque qui avait motivé le gabarit — un échec de détection écartant sélectivement
    les images les moins réalistes — reste réel, mais il se gère en le mesurant :
    `ita_measurable_fraction` remonte la proportion effectivement mesurée (~90 % sur nos
    images générées), et `compute_ita` retombe sur le gabarit plutôt que d'abandonner.
    """
    count = len(conditions["age"])
    measured = {
        "age": np.zeros(count, np.float32),
        "gender": np.zeros(count, np.int64),
        "gender_conf": np.zeros(count, np.float32),
        "ita_oracle": np.zeros(count, np.float32),
        "ita_analytic": np.full(count, np.nan, np.float32),
        "ita_valid": np.zeros(count, bool),
    }
    kept_images: List[np.ndarray] = []

    for start in tqdm(range(0, count, batch_size), desc=desc):
        stop = min(start + batch_size, count)
        images = generator.generate(
            conditions["age"][start:stop],
            conditions["gender"][start:stop],
            conditions["ita"][start:stop],
            batch_size=stop - start,
            sample_cfg=sample_cfg,
            seed=seed + start,
        )["images"]

        prediction = oracle.predict(images)
        measured["age"][start:stop] = prediction["age"]
        measured["gender"][start:stop] = prediction["gender"]
        measured["gender_conf"][start:stop] = prediction["gender_conf"]
        measured["ita_oracle"][start:stop] = prediction["ita"]

        analytic, valid = compute_ita_batch(images, use_mediapipe=use_mediapipe)
        measured["ita_analytic"][start:stop] = analytic
        measured["ita_valid"][start:stop] = valid

        if keep_images:
            kept_images.append(images)

    if keep_images:
        measured["images"] = np.concatenate(kept_images)
    return measured


# --------------------------------------------------------------------------------------
# Synthèse — F-E2, F-E3
# --------------------------------------------------------------------------------------


def attribute_fidelity(
    conditions: Dict[str, np.ndarray],
    measured: Dict[str, np.ndarray],
    oracle_metrics: Optional[Dict] = None,
) -> Dict[str, object]:
    """Précision de genre, MAE d'âge (F-E2) et ΔITA (F-E3), avec les barres d'erreur."""
    requested_age = conditions["age"]
    requested_gender = conditions["gender"]
    requested_ita = conditions["ita"]

    age_error = measured["age"] - requested_age
    gender_match = measured["gender"] == requested_gender

    valid = measured["ita_valid"]
    ita_delta = np.abs(measured["ita_analytic"][valid] - requested_ita[valid])

    baseline = (oracle_metrics or {}).get("metrics", oracle_metrics) or {}

    result: Dict[str, object] = {
        "n": int(len(requested_age)),
        # --- F-E2 ---
        "gender_accuracy": float(gender_match.mean()),
        "gender_mean_confidence": float(measured["gender_conf"].mean()),
        "gender_accuracy_target": "> 90 % (§8.1)",
        "age_mae": float(np.abs(age_error).mean()),
        "age_bias": float(age_error.mean()),
        "age_mae_target": "< 8 ans (§8.1)",
        # Régression vers la moyenne : le symptôme n°1 décrit au §1.1. Une pente
        # nettement inférieure à 1 signifie que le modèle compresse la plage d'âge vers
        # le centre de sa distribution d'entraînement, même si la MAE paraît correcte.
        "age_slope": float(np.polyfit(requested_age, measured["age"], 1)[0]),
        "age_pearson_r": float(np.corrcoef(requested_age, measured["age"])[0, 1]),
        # --- F-E3 ---
        "ita_delta_mean": float(ita_delta.mean()) if ita_delta.size else float("nan"),
        "ita_delta_median": float(np.median(ita_delta)) if ita_delta.size else float("nan"),
        "ita_delta_target": "< 10° (§8.1)",
        "ita_measurable_fraction": float(valid.mean()),
        "ita_slope": (
            float(np.polyfit(requested_ita[valid], measured["ita_analytic"][valid], 1)[0])
            if valid.sum() > 2 else float("nan")
        ),
        "ita_pearson_r": (
            float(np.corrcoef(requested_ita[valid], measured["ita_analytic"][valid])[0, 1])
            if valid.sum() > 2 else float("nan")
        ),
        # Accord entre les deux instruments (F-O5) : si la tête ITA de l'oracle et le
        # calcul analytique divergent sur les images générées alors qu'ils s'accordaient
        # sur les images réelles, c'est le signe d'un artefact de génération (peau
        # aplatie, texture absente) que ni l'un ni l'autre ne mesure correctement.
        "ita_oracle_vs_analytic_mae": (
            float(np.abs(measured["ita_oracle"][valid] - measured["ita_analytic"][valid]).mean())
            if valid.any() else float("nan")
        ),
    }

    if baseline:
        result["oracle_baseline"] = {
            "note": "Erreur de l'oracle sur des visages RÉELS (F-O4). "
                    "L'écart imputable au générateur est ce qui dépasse ces valeurs.",
            "gender_accuracy_on_real": baseline.get("gender_accuracy"),
            "age_mae_on_real": baseline.get("age_mae"),
            "ita_mae_on_real": baseline.get("ita_mae"),
        }
        if baseline.get("age_mae") is not None:
            result["age_mae_net_of_oracle"] = float(
                max(0.0, result["age_mae"] - baseline["age_mae"])
            )
    return result


# --------------------------------------------------------------------------------------
# Grille de sous-groupes — F-E7, E-2, §8.3
# --------------------------------------------------------------------------------------


def _bin_index(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


def subgroup_grid(
    generator,
    oracle,
    cfg,
    samples_per_cell: int = 300,
    seed: int = 0,
    sample_cfg=None,
    reference_split: str = "val",
    compute_cell_fid: bool = True,
) -> Dict[str, object]:
    """`4 tranches d'âge × 2 genres × 4 bins de teint = 32 cellules` (§8.3).

    Pour chaque cellule : précision de genre, MAE d'âge, ΔITA, et FID contre les images
    **réelles de la même cellule**. C'est la figure qui documente le biais (E-2) et
    remplit la section « edge cases ».

    Le FID par cellule est calculé sur ~300 images et est donc fortement biaisé à la
    hausse en valeur absolue. Il reste comparable **entre cellules**, puisque toutes
    subissent le même biais d'échantillon — et c'est le contraste entre cellules qui
    porte l'information de biais, pas le niveau.
    """
    from facedit.eval.metrics import TemporaryImageDir, compute_fid

    age_edges = cfg.eval.age_bins
    ita_edges = cfg.eval.ita_bins
    rng = np.random.default_rng(seed)

    # Le FID par cellule est comparé entre cellules, jamais dans l'absolu. On aligne donc
    # le nombre d'images réelles sur le nombre d'images générées : au-delà de ~2× le FID
    # ne bouge plus, alors que le coût de décodage PNG, lui, continue de croître — et il
    # est payé 32 fois. Un plafond fixe à 512 rendait aussi les cellules bien pourvues
    # plus lentes que les autres sans les rendre plus informatives.
    reference_cap = max(256, 2 * samples_per_cell)
    reference_index, reference_source = (
        _real_cell_index(cfg, age_edges, ita_edges, reference_split, reference_cap)
        if compute_cell_fid
        else ({}, None)
    )

    cells: List[Dict[str, object]] = []
    for age_index in range(len(age_edges) - 1):
        for gender in (0, 1):
            for ita_index in range(len(ita_edges) - 1):
                age_low, age_high = age_edges[age_index], age_edges[age_index + 1]
                ita_low, ita_high = ita_edges[ita_index], ita_edges[ita_index + 1]

                conditions = {
                    "age": rng.uniform(age_low, age_high, samples_per_cell).astype(np.float32),
                    "gender": np.full(samples_per_cell, gender, dtype=np.int64),
                    "ita": rng.uniform(ita_low, ita_high, samples_per_cell).astype(np.float32),
                }
                measured = generate_and_measure(
                    generator, oracle, conditions,
                    batch_size=cfg.eval.batch_size,
                    seed=seed + 7919 * len(cells),
                    sample_cfg=sample_cfg,
                    keep_images=compute_cell_fid,
                    desc=f"cellule {len(cells) + 1}/32",
                )
                fidelity = attribute_fidelity(conditions, measured)

                cell: Dict[str, object] = {
                    "age_bin": f"{age_low:.0f}-{age_high:.0f}",
                    "gender": "Female" if gender else "Male",
                    "ita_bin": f"{ita_low:.0f}..{ita_high:.0f}",
                    "monk_approx": str(ita_to_monk(float((ita_low + ita_high) / 2))),
                    "n": samples_per_cell,
                    "gender_accuracy": fidelity["gender_accuracy"],
                    "age_mae": fidelity["age_mae"],
                    "age_bias": fidelity["age_bias"],
                    "ita_delta_mean": fidelity["ita_delta_mean"],
                    "ita_measurable_fraction": fidelity["ita_measurable_fraction"],
                }

                key = (age_index, gender, ita_index)
                rows = reference_index.get(key, [])
                cell["n_real"] = len(rows)

                if compute_cell_fid and len(rows) >= 32:
                    # Les images réelles de la cellule ne sont chargées qu'ici, puis
                    # relâchées. Les garder toutes en mémoire coûterait ~800 Mo au palier
                    # final (32 cellules × 512 images × 128²×3), en pure attente.
                    real_images = _load_rows(reference_source, rows, cfg.data.image_size)
                    with TemporaryImageDir(measured["images"], "cell_gen") as gen_dir:
                        with TemporaryImageDir(real_images, "cell_real") as real_dir:
                            cell["fid"] = compute_fid(gen_dir, real_dir)
                    del real_images
                else:
                    cell["fid"] = None
                    if compute_cell_fid:
                        # Une cellule vide du côté réel est en soi un résultat : elle dit
                        # que le jeu d'entraînement ne couvre pas ce sous-groupe.
                        cell["note"] = "trop peu d'images réelles dans cette cellule"

                cells.append(cell)

    valid_fids = [c["fid"] for c in cells if c["fid"] is not None]
    accuracies = [c["gender_accuracy"] for c in cells]
    age_maes = [c["age_mae"] for c in cells]
    ita_deltas = [c["ita_delta_mean"] for c in cells if not np.isnan(c["ita_delta_mean"])]

    return {
        "cells": cells,
        "axes": {
            "age_bins": list(age_edges),
            "genders": ["Male", "Female"],
            "ita_bins": list(ita_edges),
        },
        "summary": {
            "n_cells": len(cells),
            "n_cells_without_real_reference": sum(1 for c in cells if c["n_real"] < 32),
            "gender_accuracy_worst": float(np.min(accuracies)),
            "gender_accuracy_spread": float(np.max(accuracies) - np.min(accuracies)),
            "age_mae_worst": float(np.max(age_maes)),
            "age_mae_spread": float(np.max(age_maes) - np.min(age_maes)),
            "ita_delta_worst": float(np.max(ita_deltas)) if ita_deltas else None,
            "fid_worst": float(np.max(valid_fids)) if valid_fids else None,
            "fid_spread": (
                float(np.max(valid_fids) - np.min(valid_fids)) if len(valid_fids) > 1 else None
            ),
        },
        "interpretation": (
            "L'étendue (spread) importe plus que le pire cas : un écart de précision de "
            "genre supérieur à ~10 points entre la meilleure et la pire cellule signale "
            "un biais de sous-groupe hérité de la distribution d'entraînement, pas un "
            "aléa d'échantillonnage."
        ),
    }


def _load_rows(source, rows: Sequence[int], image_size: int) -> np.ndarray:
    from facedit.data.fairface import resize_uint8

    return np.stack([resize_uint8(source.load_image(int(r)), image_size) for r in rows])


def _real_cell_index(cfg, age_edges, ita_edges, split: str, cap: int = 512):
    """Indexe les images réelles par cellule (âge × genre × ITA), **sans les charger**.

    Renvoie `(index, source)` : un dictionnaire cellule → liste d'indices, et la source
    permettant de charger ces images au moment voulu. Les pixels ne sont lus que cellule
    par cellule, au moment du FID correspondant.
    """
    from facedit.data.fairface import open_fairface
    from facedit.oracle.train_oracle import build_oracle_labels

    source = open_fairface(cfg.data, split)
    labels = build_oracle_labels(cfg, split)

    ages = (labels["age_low"] + labels["age_high"]) / 2.0
    age_index = _bin_index(ages, age_edges)
    ita_index = _bin_index(np.nan_to_num(labels["ita"], nan=-999.0), ita_edges)

    # Les labels peuvent avoir été construits sur un sous-ensemble (option --limit de
    # l'oracle) : on borne le parcours sur le plus court des deux, plutôt que de sortir
    # du tableau.
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for row in range(min(len(source), len(labels["valid"]))):
        if not labels["valid"][row]:
            continue
        key = (int(age_index[row]), int(labels["gender"][row]), int(ita_index[row]))
        buckets.setdefault(key, []).append(row)

    return {key: rows[:cap] for key, rows in buckets.items()}, source
