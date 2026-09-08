"""Orchestrateur d'évaluation — F-E10, contrat §6.3 `run_eval(checkpoint, config) -> results.json`.

    python -m facedit.eval.run_eval --ckpt runs/final/ckpt_final.pt
    python -m facedit.eval.run_eval --ckpt runs/final/ckpt_final.pt --quick
    python -m facedit.eval.run_eval --ckpt runs/final/ckpt_final.pt --only fid attributes

Le même harnais évalue notre modèle **et** les baselines (§8.4) : c'est la condition
d'une comparaison honnête. `--baseline` branche un générateur externe à la place du DiT.

Chaque section est isolée : un échec (dépendance absente, VRAM insuffisante) est capturé,
consigné dans `results.json` sous forme d'erreur, et l'évaluation continue. Une
évaluation de 3 heures ne doit pas être perdue parce que DINOv2 n'était pas téléchargeable.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from facedit.utils.config import Config, load_config
from facedit.utils.repro import env_fingerprint, file_sha256, seed_everything

RESULTS_SCHEMA_VERSION = "1.0"

# Seuils du §8.1. `higher_is_better` évite d'inverser la comparaison au cas par cas.
TARGETS = {
    "fid": {"target": 30.0, "fail": 60.0, "higher_is_better": False, "label": "FID"},
    "gender_accuracy": {
        "target": 0.90, "fail": 0.75, "higher_is_better": True, "label": "Précision genre",
    },
    "age_mae": {"target": 8.0, "fail": 15.0, "higher_is_better": False, "label": "MAE âge"},
    "ita_delta_mean": {"target": 10.0, "fail": 20.0, "higher_is_better": False, "label": "ΔITA"},
    "lpips_mean": {
        "target": 0.35, "fail": 0.20, "higher_is_better": True, "label": "LPIPS intra-condition",
    },
}


def _grade(name: str, value: Optional[float]) -> Dict[str, object]:
    spec = TARGETS[name]
    if value is None or not np.isfinite(value):
        return {"metric": spec["label"], "value": None, "status": "non mesuré"}

    if spec["higher_is_better"]:
        status = "cible atteinte" if value >= spec["target"] else (
            "échec" if value <= spec["fail"] else "entre cible et seuil"
        )
    else:
        status = "cible atteinte" if value <= spec["target"] else (
            "échec" if value >= spec["fail"] else "entre cible et seuil"
        )
    return {
        "metric": spec["label"],
        "value": round(float(value), 4),
        "target": spec["target"],
        "fail_threshold": spec["fail"],
        "status": status,
    }


class Section:
    """Exécute une section d'évaluation en isolant ses erreurs."""

    def __init__(self, results: Dict, verbose: bool = True):
        self.results = results
        self.verbose = verbose
        self.timings: Dict[str, float] = {}

    def run(self, name: str, function: Callable[[], object]) -> None:
        started = time.time()
        if self.verbose:
            print(f"\n{'=' * 70}\n[eval] {name}\n{'=' * 70}")
        try:
            self.results[name] = function()
        except Exception as exc:
            message = f"{exc.__class__.__name__}: {exc}"
            print(f"[eval] section '{name}' en échec — {message}")
            self.results[name] = {"error": message, "traceback": traceback.format_exc()}
        self.timings[name] = round(time.time() - started, 1)


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


def _section_fid(generator, cfg, sample_cfg, num_samples: int) -> Dict[str, object]:
    """F-E1 (et F-E8 au passage, sur le même lot d'images)."""
    from facedit.eval.attributes import sample_conditions
    from facedit.eval.metrics import (
        TemporaryImageDir, build_reference_set, compute_fid, fd_dinov2,
    )

    reference_dir = build_reference_set(cfg, num_samples, cfg.eval.real_stats_dir, "val")

    # Seconde référence, restreinte aux images que le filtre F-D9 aurait gardées. Elle
    # n'existe que si le masque du split val a été calculé ; sinon le FID filtré est
    # simplement absent du rapport plutôt qu'approximé.
    from facedit.data.filter_faces import mask_path

    filtered_reference_dir = None
    val_mask_path = mask_path(cfg, "val")
    if val_mask_path.exists():
        filtered_reference_dir = build_reference_set(
            cfg,
            num_samples,
            Path(cfg.eval.real_stats_dir).with_name(
                Path(cfg.eval.real_stats_dir).name + "_filtered"
            ),
            "val",
            keep_mask=np.load(val_mask_path),
        )

    # Conditions tirées uniformément : le FID mesure alors le réalisme sur toute la plage
    # contrôlable, y compris les extrêmes, et non seulement là où le modèle est à l'aise.
    conditions = sample_conditions(num_samples, seed=cfg.seed + 1)
    batch = cfg.eval.batch_size
    chunks: List[np.ndarray] = []

    from tqdm import tqdm

    for start in tqdm(range(0, num_samples, batch), desc="génération FID"):
        stop = min(start + batch, num_samples)
        chunks.append(
            generator.generate(
                conditions["age"][start:stop],
                conditions["gender"][start:stop],
                conditions["ita"][start:stop],
                batch_size=stop - start,
                sample_cfg=sample_cfg,
                seed=cfg.seed + start,
            )["images"]
        )
    images = np.concatenate(chunks)

    with TemporaryImageDir(images, "facedit_fid") as generated_dir:
        fid_value = compute_fid(generated_dir, reference_dir, cfg.eval.batch_size)
        fid_filtered = (
            compute_fid(generated_dir, filtered_reference_dir, cfg.eval.batch_size)
            if filtered_reference_dir is not None
            else None
        )

    result: Dict[str, object] = {
        "fid": fid_value,
        # Référence filtrée : la distribution que le modèle entraîné sur jeu filtré vise
        # réellement. À lire CONJOINTEMENT avec `fid` — voir `build_reference_set`. Un
        # écart important entre les deux ne dit rien sur le générateur, seulement sur la
        # sévérité du filtre.
        "fid_filtered_reference": fid_filtered,
        "n_generated": int(len(images)),
        "n_reference": len(list(Path(reference_dir).glob("*.png"))),
        "reference_split": "val",
        "implementation": "clean-fid, mode=clean",
        "note": "Calculé sur les images 128×128 brutes, AVANT restauration (F-S5) : un "
                "restaurateur entraîné sur des visages réels ferait baisser le FID sans "
                "que le générateur y soit pour rien.",
    }

    # F-E8 : même lot, autre espace de features.
    from PIL import Image

    reference_files = sorted(Path(reference_dir).glob("*.png"))[: min(2048, len(images))]
    reference_images = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in reference_files])
    result["fd_dinov2"] = fd_dinov2(
        images[: len(reference_images)], reference_images, generator.device
    )
    return result


def _section_attributes(generator, oracle, cfg, sample_cfg, num_samples: int) -> Dict[str, object]:
    """F-E2 et F-E3, plus les données du nuage de points demandé/mesuré."""
    from facedit.eval.attributes import attribute_fidelity, generate_and_measure, sample_conditions

    conditions = sample_conditions(num_samples, seed=cfg.seed + 2)
    measured = generate_and_measure(
        generator, oracle, conditions,
        batch_size=cfg.eval.batch_size, seed=cfg.seed + 100, sample_cfg=sample_cfg,
        desc="génération attributs",
    )
    result = attribute_fidelity(conditions, measured, getattr(oracle, "metrics", None))
    result["_scatter"] = {
        "age_requested": conditions["age"].tolist(),
        "age_measured": measured["age"].tolist(),
        "ita_requested": conditions["ita"].tolist(),
        "ita_measured": measured["ita_analytic"].tolist(),
    }
    return result


def _section_w_sweep(generator, oracle, cfg, base_sample_cfg) -> Dict[str, object]:
    """F-E6 : courbes FID et fidélité contre w, sur le même graphique.

    Le balayage utilise moins d'images que la mesure principale (FID sur 2000 au lieu de
    10 000) : sept points de balayage à 10 000 images coûteraient plusieurs heures pour
    une courbe dont seule la **forme** importe. Le biais d'échantillon est identique pour
    tous les points, donc la comparaison entre valeurs de w reste valide — mais les FID
    du balayage ne sont pas comparables au FID principal, et le JSON le dit.
    """
    import copy

    from facedit.eval.attributes import attribute_fidelity, generate_and_measure, sample_conditions
    from facedit.eval.metrics import TemporaryImageDir, build_reference_set, compute_fid

    sweep_samples = min(2000, cfg.eval.attr_num_samples)
    reference_dir = build_reference_set(cfg, max(sweep_samples, 2048), cfg.eval.real_stats_dir, "val")
    conditions = sample_conditions(sweep_samples, seed=cfg.seed + 3)

    points: List[Dict[str, object]] = []
    for w in cfg.eval.w_sweep:
        sample_cfg = copy.deepcopy(base_sample_cfg)
        sample_cfg.guidance = float(w)
        sample_cfg.guidance_age = sample_cfg.guidance_gender = sample_cfg.guidance_ita = None

        measured = generate_and_measure(
            generator, oracle, conditions,
            batch_size=cfg.eval.batch_size, seed=cfg.seed + 200, sample_cfg=sample_cfg,
            keep_images=True, desc=f"w={w:g}",
        )
        fidelity = attribute_fidelity(conditions, measured)
        with TemporaryImageDir(measured["images"], f"facedit_w{w:g}") as generated_dir:
            fid_value = compute_fid(generated_dir, reference_dir, cfg.eval.batch_size)

        points.append({
            "w": float(w),
            "fid": fid_value,
            "gender_accuracy": fidelity["gender_accuracy"],
            "age_mae": fidelity["age_mae"],
            "age_slope": fidelity["age_slope"],
            "ita_delta_mean": fidelity["ita_delta_mean"],
        })
        print(f"  w={w:g} · FID {fid_value if fid_value is None else round(fid_value, 2)} · "
              f"MAE âge {fidelity['age_mae']:.2f} · ΔITA {fidelity['ita_delta_mean']:.2f}°")

    return {
        "points": points,
        "n_samples_per_point": sweep_samples,
        "recommended_w": _recommend_w(points),
        "note": "Les FID de ce balayage sont calculés sur un échantillon réduit et ne "
                "sont pas comparables au FID principal ; seule leur évolution avec w l'est.",
        "answers_open_question": "n°4 — quelle valeur de w retenir comme défaut de l'interface",
    }


def _recommend_w(points: List[Dict[str, object]]) -> Optional[float]:
    """Choisit w par un score composite normalisé, plutôt qu'à l'œil.

    Chaque métrique est ramenée sur [0, 1] par min-max **à l'intérieur du balayage**, puis
    le réalisme (FID) et l'obéissance (les trois erreurs d'attributs) pèsent chacun pour
    moitié. Ce n'est pas une vérité : c'est une règle explicite, reproductible et
    critiquable — ce qui vaut mieux qu'un choix implicite justifié après coup.
    """
    usable = [p for p in points if p.get("fid") is not None]
    if len(usable) < 2:
        return None

    def normalize(values: List[float]) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        span = array.max() - array.min()
        return np.zeros_like(array) if span == 0 else (array - array.min()) / span

    realism = normalize([p["fid"] for p in usable])  # 0 = meilleur
    obedience = (
        normalize([p["age_mae"] for p in usable])
        + normalize([p["ita_delta_mean"] for p in usable])
        + normalize([1.0 - p["gender_accuracy"] for p in usable])
    ) / 3.0

    score = 0.5 * realism + 0.5 * obedience
    return float(usable[int(np.argmin(score))]["w"])


def _section_memorization(generator, cfg, sample_cfg, num_samples: int) -> Dict[str, object]:
    """F-E9 et E-3."""
    from facedit.eval.attributes import sample_conditions
    from facedit.eval.metrics import memorization_test

    conditions = sample_conditions(num_samples, seed=cfg.seed + 5)
    chunks = []
    for start in range(0, num_samples, cfg.eval.batch_size):
        stop = min(start + cfg.eval.batch_size, num_samples)
        chunks.append(
            generator.generate(
                conditions["age"][start:stop], conditions["gender"][start:stop],
                conditions["ita"][start:stop], batch_size=stop - start,
                sample_cfg=sample_cfg, seed=cfg.seed + 500 + start,
            )["images"]
        )
    images = np.concatenate(chunks)
    return memorization_test(
        generator, images, cfg.data.latents_path("train"), device=generator.device
    )


# --------------------------------------------------------------------------------------
# Point d'entrée
# --------------------------------------------------------------------------------------


def run_eval(
    checkpoint_path: str | Path,
    cfg: Config,
    out_dir: Optional[str | Path] = None,
    quick: bool = False,
    only: Optional[List[str]] = None,
    generator=None,
    label: str = "facedit-dit",
) -> Dict[str, object]:
    """Contrat §6.3. Renvoie le dictionnaire écrit dans `results.json`."""
    from facedit.eval.plots import render_all
    from facedit.oracle.model import load_oracle
    from facedit.sample import load_generator

    seed_everything(cfg.seed, deterministic=True)
    out_dir = Path(out_dir or cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if generator is None:
        generator = load_generator(checkpoint_path, cfg.device)
    oracle = load_oracle(cfg.eval.oracle_ckpt, generator.device)
    sample_cfg = cfg.sample

    # Les baselines (§8.4) n'ont ni checkpoint local ni modèle PyTorch de notre côté :
    # le harnais ne doit rien exiger au-delà de `generate()` et `device`.
    inner_model = getattr(generator, "model", None)
    has_checkpoint = checkpoint_path is not None and Path(str(checkpoint_path)).exists()

    scale = 0.05 if quick else 1.0
    counts = {
        "fid": max(256, int(cfg.eval.fid_num_samples * scale)),
        "attr": max(128, int(cfg.eval.attr_num_samples * scale)),
        "leakage": max(16, int(cfg.eval.leakage_num_seeds * scale)),
        "diversity": max(16, int(cfg.eval.diversity_num_pairs * scale)),
        "subgroup": max(16, int(cfg.eval.subgroup_samples_per_cell * scale)),
        "memorization": max(32, int(cfg.eval.memorization_num_samples * scale)),
    }

    results: Dict[str, object] = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "label": label,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "quick_mode": quick,
        "checkpoint": {
            "path": str(checkpoint_path),
            # F-E10 : le hash du checkpoint évalué. Sans lui, un `results.json` ne peut
            # pas être rattaché de façon certaine aux poids qui l'ont produit.
            "sha256": file_sha256(checkpoint_path) if has_checkpoint else None,
            "step": getattr(generator, "step", None),
            "params_millions": (
                round(inner_model.num_parameters / 1e6, 2)
                if inner_model is not None and hasattr(inner_model, "num_parameters")
                else None
            ),
            # Architecture RÉELLEMENT évaluée, lue dans le checkpoint. Elle n'a aucune
            # raison de coïncider avec `config.model` ci-dessous : le fichier
            # d'évaluation (eval_fast.yaml) hérite de base.yaml et transporte donc la
            # section `model` du prototype, qui ne décrit pas les poids chargés. Un
            # lecteur du JSON qui se fierait à `config.model` conclurait à un modèle de
            # 9,95 M paramètres alors que 32,96 M ont été mesurés.
            "model": (
                generator.cfg.model.__dict__.copy()
                if getattr(generator, "cfg", None) is not None
                else None
            ),
        },
        "config_hash": cfg.hash(),
        "config": cfg.to_dict(),
        "config_note": (
            "Configuration de L'ÉVALUATION. Sa section `model` est héritée et ne décrit "
            "pas les poids évalués : voir `checkpoint.model`."
        ),
        "env": env_fingerprint(),
        "sampling": {
            "num_steps": sample_cfg.num_steps,
            "guidance": sample_cfg.guidance,
            "eta": sample_cfg.eta,
            "restore": sample_cfg.restore,
        },
        "oracle": {
            "checkpoint": cfg.eval.oracle_ckpt,
            "metrics_on_real": getattr(oracle, "metrics", {}),
            "role": "F-O4 — barre d'erreur des mesures d'attributs",
        },
    }

    all_sections = ["fid", "attributes", "leakage", "diversity", "w_sweep", "subgroups", "memorization"]
    selected = only or all_sections
    section = Section(results)

    if "fid" in selected:
        section.run("fid", lambda: _section_fid(generator, cfg, sample_cfg, counts["fid"]))
    if "attributes" in selected:
        section.run(
            "attributes", lambda: _section_attributes(generator, oracle, cfg, sample_cfg, counts["attr"])
        )
    if "leakage" in selected:
        from facedit.eval.leakage import leakage_matrix

        section.run(
            "leakage",
            lambda: leakage_matrix(generator, oracle, cfg, counts["leakage"], cfg.seed, sample_cfg),
        )
    if "diversity" in selected:
        from facedit.eval.attributes import sample_conditions
        from facedit.eval.metrics import lpips_diversity

        conditions = sample_conditions(max(4, counts["diversity"] // 20), seed=cfg.seed + 4)
        pairs = list(zip(conditions["age"], conditions["gender"], conditions["ita"]))
        section.run(
            "diversity",
            lambda: lpips_diversity(
                generator, pairs, counts["diversity"], cfg.seed, generator.device, sample_cfg
            ),
        )
    if "w_sweep" in selected:
        section.run("w_sweep", lambda: _section_w_sweep(generator, oracle, cfg, sample_cfg))
    if "subgroups" in selected:
        from facedit.eval.attributes import subgroup_grid

        section.run(
            "subgroups",
            lambda: subgroup_grid(
                generator, oracle, cfg, counts["subgroup"], cfg.seed, sample_cfg,
                compute_cell_fid=not quick,
            ),
        )
    if "memorization" in selected:
        if getattr(generator, "vae", None) is None:
            # Le test cherche le plus proche voisin dans l'espace latent de NOTRE VAE.
            # Sur une baseline externe, cet espace n'existe pas et la question n'a de
            # toute façon pas de sens : elle porte sur notre jeu d'entraînement.
            results["memorization"] = {
                "skipped": "générateur sans VAE — test non applicable à une baseline externe"
            }
        else:
            section.run(
                "memorization",
                lambda: _section_memorization(generator, cfg, sample_cfg, counts["memorization"]),
            )

    results["timings_seconds"] = section.timings

    # --- confrontation aux cibles du §8.1 ---------------------------------------------
    def dig(path: List[str]):
        cursor: object = results
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                return None
            cursor = cursor[key]
        return cursor if isinstance(cursor, (int, float)) else None

    results["targets"] = {
        "reference": "§8.1 — cibles indicatives pour un modèle de 12M paramètres, "
                     "PAS des promesses. Un écart est un résultat à analyser.",
        "fid": _grade("fid", dig(["fid", "fid"])),
        "gender_accuracy": _grade("gender_accuracy", dig(["attributes", "gender_accuracy"])),
        "age_mae": _grade("age_mae", dig(["attributes", "age_mae"])),
        "ita_delta_mean": _grade("ita_delta_mean", dig(["attributes", "ita_delta_mean"])),
        "lpips_mean": _grade("lpips_mean", dig(["diversity", "lpips_mean"])),
    }

    # Le nuage de points est volumineux : on le sort de la section pour garder
    # `results.json` lisible, mais on le conserve — les figures en dépendent.
    if isinstance(results.get("attributes"), dict) and "_scatter" in results["attributes"]:
        results["scatter"] = results["attributes"].pop("_scatter")

    results_path = out_dir / f"results_{label}.json"
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"\n[eval] → {results_path}")

    try:
        figures = render_all(results, out_dir / "figures")
        results["figures"] = figures
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        for figure in figures:
            print(f"[eval] figure → {figure}")
    except Exception as exc:
        print(f"[eval] figures non générées : {exc}")

    _print_summary(results)
    return results


def _print_summary(results: Dict) -> None:
    print(f"\n{'=' * 70}\nRÉSUMÉ — {results['label']} (pas {results['checkpoint']['step']})\n{'=' * 70}")
    for entry in results.get("targets", {}).values():
        if isinstance(entry, dict) and "metric" in entry:
            value = "—" if entry["value"] is None else f"{entry['value']:g}"
            print(f"  {entry['metric']:<26} {value:>10}   {entry['status']}")

    leakage = results.get("leakage")
    if isinstance(leakage, dict) and "summary" in leakage:
        print(f"  {'Désenchevêtrement':<26} "
              f"{leakage['summary']['disentanglement_ratio']:>10.2f}   "
              f"(diagonale/hors-diagonale, > 5 souhaitable)")

    sweep = results.get("w_sweep")
    if isinstance(sweep, dict) and sweep.get("recommended_w") is not None:
        print(f"  {'w recommandé':<26} {sweep['recommended_w']:>10g}   (question ouverte n°4)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harnais d'évaluation FaceDiT (§8)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default=None,
                        help="par défaut, la configuration embarquée dans le checkpoint")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default="facedit-dit")
    parser.add_argument("--quick", action="store_true",
                        help="5 %% des échantillons : validation du harnais, pas des résultats")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    if args.config:
        cfg = load_config(args.config, args.overrides)
    else:
        import torch

        from facedit.utils.config import _from_dict

        payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = _from_dict(Config, payload["config"])
        print(f"[eval] configuration reprise du checkpoint (hash {cfg.hash()})")

    run_eval(args.ckpt, cfg, args.out, args.quick, args.only, label=args.label)


if __name__ == "__main__":
    main()
