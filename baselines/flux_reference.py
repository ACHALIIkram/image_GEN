"""Baseline 2 — modèle texte-image généraliste piloté par prompt. §8.4, prio S.

Rôle double :

1. **borne supérieure de photoréalisme** — FLUX.1 ou Z-Image produisent des portraits
   qu'aucun modèle de 10 M paramètres entraîné cinq heures n'approchera ;
2. **illustration de la faible obéissance du prompt** — c'est le problème n°1 du §1.1.
   Demander « une femme de 62 ans » à un modèle T2I donne typiquement un visage perçu
   entre 45 et 50 ans. Cette baseline le **chiffre** au lieu de l'affirmer.

C'est la comparaison la plus parlante du rapport : un modèle mille fois plus gros, plus
réaliste, et pourtant moins obéissant sur la seule chose qu'on lui demande. Le résultat
attendu est que la MAE d'âge du T2I dépasse la nôtre malgré un FID bien meilleur.

Portage du workflow ComfyUI vers `diffusers`, conformément à §6.2 et §11.
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch

from baselines.base import BaselineGenerator
from facedit.data.ita import ita_to_category

# Le teint est décrit par sa catégorie dermatologique ITA, pas par une catégorie
# ethnique (E-1). C'est la traduction la plus fidèle possible d'un ITA vers du langage
# naturel, et sa perte d'information fait partie de ce que la baseline démontre.
ITA_TO_PHRASE = {
    "very light": "very light skin",
    "light": "light skin",
    "intermediate": "medium skin tone",
    "tan": "tan skin tone",
    "brown": "brown skin tone",
    "dark": "deep dark skin tone",
}


def build_prompt(age: float, gender: int, ita: float) -> str:
    """Prompt canonique. Figé : le changer invaliderait la comparaison."""
    who = "woman" if gender == 1 else "man"
    skin = ITA_TO_PHRASE[str(ita_to_category(float(ita)))]
    return (
        f"a photorealistic frontal portrait photograph of a {int(round(age))} year old "
        f"{who} with {skin}, neutral expression, plain background, "
        f"studio lighting, sharp focus, 50mm lens"
    )


NEGATIVE_PROMPT = "cartoon, illustration, painting, 3d render, blurry, deformed, multiple faces"


class TextToImageBaseline(BaselineGenerator):
    label = "t2i-prompt"

    def __init__(
        self,
        cfg,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
    ):
        super().__init__(cfg, label=f"t2i-{model_id.split('/')[-1]}")
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompts_used: List[str] = []

        from diffusers import AutoPipelineForText2Image

        self.pipe = AutoPipelineForText2Image.from_pretrained(model_id, torch_dtype=dtype)
        if enable_cpu_offload:
            # 8 Go de VRAM ne suffisent pas à FLUX en résident. Le déchargement séquentiel
            # est lent (~10 s/image) mais c'est la seule option sur la machine cible ;
            # cette baseline est donc évaluée sur moins d'images, ce que le JSON consigne.
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def _render(self, age, gender, ita, seed: int) -> np.ndarray:
        images = []
        for index in range(len(age)):
            prompt = build_prompt(float(age[index]), int(gender[index]), float(ita[index]))
            self.prompts_used.append(prompt)
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            kwargs = dict(
                prompt=prompt,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator,
                height=512,
                width=512,
            )
            if "negative_prompt" in self.pipe.__call__.__code__.co_varnames:
                kwargs["negative_prompt"] = NEGATIVE_PROMPT
            output = self.pipe(**kwargs)
            images.append(np.asarray(output.images[0].convert("RGB"), dtype=np.uint8))
        return np.stack(images)


def main() -> None:
    from facedit.eval.run_eval import run_eval
    from facedit.utils.config import load_config

    parser = argparse.ArgumentParser(description="Baseline texte-image (§8.4)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance", type=float, default=0.0)
    parser.add_argument("--no-offload", action="store_true")
    parser.add_argument(
        "--only", nargs="*", default=["attributes", "fid"],
        help="Par défaut, uniquement les sections abordables : chaque image coûte "
             "plusieurs secondes avec déchargement CPU.",
    )
    parser.add_argument("--n", type=int, default=300,
                        help="Écrase eval.attr_num_samples et eval.fid_num_samples")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.eval.attr_num_samples = args.n
    cfg.eval.fid_num_samples = args.n
    cfg.eval.batch_size = 4

    generator = TextToImageBaseline(
        cfg, args.model_id, args.steps, args.guidance, cfg.device,
        enable_cpu_offload=not args.no_offload,
    )
    results = run_eval(
        checkpoint_path=args.model_id, cfg=cfg, only=args.only,
        generator=generator, label=generator.label,
    )

    attributes = results.get("attributes", {})
    if isinstance(attributes, dict) and "age_mae" in attributes:
        print(
            f"\n[baseline T2I] MAE d'âge {attributes['age_mae']:.2f} ans · "
            f"pente {attributes['age_slope']:.2f}\n"
            f"Une pente nettement inférieure à 1 est la démonstration chiffrée du "
            f"problème n°1 du §1.1 : le modèle régresse vers la moyenne de sa "
            f"distribution d'entraînement au lieu d'obéir au prompt."
        )


if __name__ == "__main__":
    main()
