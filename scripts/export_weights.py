"""Réduit un checkpoint d'entraînement à un fichier de poids distribuable.

    python scripts/export_weights.py runs/final_v3/ckpt_final.pt poids/facedit_v3.pt

Un checkpoint d'entraînement pèse 527 Mo parce qu'il transporte de quoi *reprendre* :
les poids bruts, les poids EMA, et les deux moments d'AdamW — soit quatre copies des
33 M paramètres, plus l'état du planificateur. Pour *générer*, une seule copie suffit :
les poids EMA, que `load_generator` est le seul à accepter (F-T3).

En ne gardant que ceux-ci, en float16, on passe de 527 Mo à 67 Mo, soit un facteur 7,9.
Le fichier tient alors dans un dépôt Git sans stockage annexe.

Pourquoi le float16 ne coûte rien ici. Les poids sont conservés en float16 puis reconvertis
en float32 au chargement. La perte est bornée par la précision relative du format,
2**-11 ≈ 4,9e-4, alors que le modèle est *entraîné* en bf16 — un format qui n'a que
8 bits de mantisse, donc strictement moins précis que le float16 sur cette plage. Le
bruit d'arrondi introduit ici est inférieur à celui que l'entraînement a déjà toléré.
Ce script le vérifie plutôt que de l'affirmer : il compare les sorties du modèle avant
et après conversion et refuse d'écrire si l'écart dépasse le seuil.

Ce que le fichier produit ne permet PAS : reprendre l'entraînement. C'est délibéré et
sans perte pour qui veut seulement faire tourner la démonstration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Ce fichier vit dans scripts/, donc `python scripts/export_weights.py` place scripts/ en
# tête de sys.path et non la racine du projet : `import facedit` échouerait. On ajoute la
# racine explicitement pour que le script s'appelle par son chemin, sans `-m` ni variable
# d'environnement à poser au préalable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Écart relatif maximal toléré entre la prédiction du modèle en float32 et celle du
# modèle passé par float16. Le seuil est vérifié sur du bruit gaussien, pire cas pour
# une comparaison relative puisque la sortie n'a aucune structure à laquelle se raccrocher.
MAX_RELATIVE_DEVIATION = 1e-3


def _predict(state_dict: dict, config: dict, seed: int = 0) -> torch.Tensor:
    """Une passe avant déterministe, pour comparer deux jeux de poids."""
    from facedit.models.dit import build_dit
    from facedit.utils.config import Config, _from_dict

    cfg = _from_dict(Config, config)
    model = build_dit(cfg.model)
    model.load_state_dict({k: v.float() for k, v in state_dict.items()})
    model.eval()

    generator = torch.Generator().manual_seed(seed)
    latents = torch.randn(4, cfg.model.in_channels, cfg.model.latent_size,
                          cfg.model.latent_size, generator=generator)
    timesteps = torch.tensor([10, 300, 600, 900])
    age = torch.tensor([25.0, 40.0, 55.0, 68.0])
    gender = torch.tensor([0, 1, 0, 1])
    ita = torch.tensor([-30.0, 10.0, 40.0, 60.0])

    with torch.no_grad():
        return model(latents, timesteps, age, gender, ita)


def export(source: Path, destination: Path, dtype: torch.dtype = torch.float16) -> dict:
    payload = torch.load(source, map_location="cpu", weights_only=False)

    if "ema" not in payload or "shadow" not in payload.get("ema", {}):
        raise ValueError(
            f"{source} ne contient pas de poids EMA. La génération les exige (F-T3) ; "
            f"exporter les poids bruts produirait un modèle mesurablement différent de "
            f"celui qui a été évalué."
        )

    shadow = payload["ema"]["shadow"]
    config = payload["config"]

    reference = _predict(shadow, config)
    converted = {key: value.to(dtype) for key, value in shadow.items()}
    candidate = _predict(converted, config)

    deviation = (candidate - reference).abs().max() / reference.abs().max()
    deviation = float(deviation)
    if deviation > MAX_RELATIVE_DEVIATION:
        raise ValueError(
            f"La conversion en {dtype} déplace la sortie de {deviation:.2e} en relatif, "
            f"au-delà du seuil {MAX_RELATIVE_DEVIATION:.0e}. Fichier non écrit : "
            f"exportez en float32 avec --dtype float32."
        )

    slim = {
        "ema": {"shadow": converted},
        "config": config,
        "step": int(payload.get("step", 0)),
        "config_hash": payload.get("config_hash"),
        "loss": payload.get("loss"),
        # Trace de provenance : sans elle, un fichier de poids isolé ne dit plus de quel
        # entraînement il sort, et une mesure publiée devient impossible à rattacher.
        "exported_from": source.name,
        "export_dtype": str(dtype),
        "export_max_relative_deviation": deviation,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, destination)

    return {
        "source_mb": source.stat().st_size / 1e6,
        "destination_mb": destination.stat().st_size / 1e6,
        "deviation": deviation,
        "step": slim["step"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="checkpoint d'entraînement (ckpt_*.pt)")
    parser.add_argument("destination", type=Path, help="fichier de poids à écrire")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    args = parser.parse_args()

    report = export(args.source, args.destination, getattr(torch, args.dtype))
    print(
        f"{args.source} ({report['source_mb']:.0f} Mo, pas {report['step']})\n"
        f"  -> {args.destination} ({report['destination_mb']:.0f} Mo, "
        f"facteur {report['source_mb'] / report['destination_mb']:.1f})\n"
        f"  écart relatif maximal sur la sortie : {report['deviation']:.2e}"
    )


if __name__ == "__main__":
    main()
