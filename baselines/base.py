"""Contrat commun aux baselines — §8.4.

« Toutes les baselines passent par **le même harnais d'évaluation** que notre modèle.
C'est la condition d'une comparaison honnête. »

Concrètement, le harnais (`facedit.eval.run_eval`) n'appelle jamais qu'une seule méthode :

    generator.generate(age, gender, ita, batch_size, sample_cfg, seed) -> {"images": ...}

où `images` est un tableau (B, H, W, 3) uint8 **à la même résolution que notre modèle**.
Tout objet exposant cette méthode, plus les attributs `device` et `label`, est évaluable
à l'identique — StyleGAN2, un modèle texte-image, ou un tirage aléatoire de contrôle.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class BaselineGenerator:
    """Classe de base. Les sous-classes n'ont qu'à implémenter `_render`."""

    label: str = "baseline"
    device: str = "cuda"
    step: Optional[int] = None
    vae = None  # aucun espace latent partagé : le test de mémorisation sera sauté

    def __init__(self, cfg, label: Optional[str] = None):
        self.cfg = cfg
        self.image_size = cfg.data.image_size
        if label:
            self.label = label

    # ------------------------------------------------------------------ à implémenter
    def _render(self, age: np.ndarray, gender: np.ndarray, ita: np.ndarray, seed: int) -> np.ndarray:
        """Produit (B, H, W, 3) uint8 pour les conditions demandées."""
        raise NotImplementedError

    # ---------------------------------------------------------------------- interface
    def generate(
        self,
        age,
        gender,
        ita,
        batch_size: int = 1,
        sample_cfg=None,
        seed: Optional[int] = None,
        decode_images: bool = True,
    ) -> Dict[str, object]:
        age = np.broadcast_to(np.atleast_1d(np.asarray(age, dtype=np.float32)), (batch_size,))
        gender = np.broadcast_to(np.atleast_1d(np.asarray(gender, dtype=np.int64)), (batch_size,))
        ita = np.broadcast_to(np.atleast_1d(np.asarray(ita, dtype=np.float32)), (batch_size,))

        images = self._render(np.array(age), np.array(gender), np.array(ita), int(seed or 0))

        if images.shape[1] != self.image_size:
            # Comparer un FID calculé en 512 à un FID calculé en 128 n'a aucun sens :
            # toute baseline est ramenée à la résolution d'évaluation.
            from facedit.data.fairface import resize_uint8

            images = np.stack([resize_uint8(img, self.image_size) for img in images])
        return {"images": images}
