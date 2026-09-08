"""Embeddings de conditionnement — F-M3 à F-M6.

Le vecteur de condition du DiT est la somme de quatre termes (§7.2) :

    c = emb_t(t) + emb_age(age) + emb_gender(gender) + emb_ita(ita)      dim hidden_size

Une **somme** et non une concaténation : chaque terme vit dans le même espace, ce qui
rend l'interpolation d'attributs (F-S4) linéaire et bien définie, et permet de retirer
un attribut isolément en le remplaçant par son jeton nul — la brique du guidage par
attribut (F-S6).

Chaque attribut porte son propre jeton nul appris. C'est ce qui rend le CFG possible
sans jamais présenter au modèle une valeur numérique « neutre » inventée : un âge
« absent » n'est pas un âge de 0 an ni de 40 ans, c'est un symbole distinct.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulation adaLN : x · (1 + scale) + shift, diffusée sur la dimension token.

    Le `1 +` est essentiel : avec `scale` initialisé à zéro, la modulation est l'identité.
    """
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------------------------------------
# Pas de diffusion
# --------------------------------------------------------------------------------------


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Embedding sinusoïdal du pas de diffusion (Transformer, Vaswani et al.).

    `t` : (B,) entiers ou flottants. Renvoie (B, dim).
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / half
    )
    args = t.float()[:, None] * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Sinusoïdal → MLP 2 couches SiLU (§7.2)."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_dim).to(self.mlp[0].weight.dtype))


# --------------------------------------------------------------------------------------
# Attributs continus : âge (F-M4) et ITA (F-M6)
# --------------------------------------------------------------------------------------


class FourierAttributeEmbedder(nn.Module):
    """Features de Fourier sur un scalaire normalisé, puis MLP.

    Les fréquences sont **fixes et log-espacées** (2^0 … 2^log2(max_freq)), pas tirées
    au hasard ni apprises. Trois raisons :

      - reproductibilité : aucune dépendance à l'état d'un RNG au moment de la
        construction du modèle ;
      - interpolabilité : `emb(x)` est une fonction lisse et déterministe de `x`, donc
        l'interpolation linéaire entre deux conditions (F-S4) parcourt une trajectoire
        continue dans l'espace de condition ;
      - stabilité : des fréquences apprises dérivent pendant l'entraînement et l'échelle
        d'un attribut cesse d'être comparable d'un checkpoint à l'autre.

    `value` est fourni en unités physiques (années, degrés) et normalisé ici, une seule
    fois, à partir des bornes de la configuration. Normaliser en amont exposerait le
    système à un décalage silencieux entre entraînement et inférence.
    """

    def __init__(
        self,
        hidden_size: int,
        num_freqs: int = 64,
        max_freq: float = 64.0,
        offset: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        if scale == 0:
            raise ValueError("`scale` de normalisation nul")
        self.offset = float(offset)
        self.scale = float(scale)

        freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), num_freqs)
        self.register_buffer("freqs", freqs * math.pi, persistent=True)

        self.mlp = nn.Sequential(
            nn.Linear(2 * num_freqs, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        # Jeton nul : représentation apprise de « cet attribut n'est pas spécifié ».
        self.null_token = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null_token, std=0.02)

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() + self.offset) / self.scale

    def forward(
        self, value: torch.Tensor, drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """`value` : (B,) en unités physiques. `drop_mask` : (B,) booléen, True = nul."""
        normalized = self.normalize(value)
        args = normalized[:, None] * self.freqs[None, :]
        features = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        embedding = self.mlp(features.to(self.mlp[0].weight.dtype))

        if drop_mask is not None and drop_mask.any():
            null = self.null_token.to(embedding.dtype).expand_as(embedding)
            embedding = torch.where(drop_mask[:, None], null, embedding)
        return embedding


# --------------------------------------------------------------------------------------
# Attribut catégoriel : genre (F-M5)
# --------------------------------------------------------------------------------------


class GenderEmbedder(nn.Module):
    """Table apprise à 3 entrées : 0 = homme, 1 = femme, 2 = inconnu (§6.3).

    Le troisième jeton n'est pas une troisième catégorie de genre : c'est le symbole
    « non spécifié » du CFG. E-5 : le caractère binaire des deux premiers jetons est une
    limitation des labels de FairFace, pas une position de l'équipe.
    """

    UNKNOWN = 2

    def __init__(self, hidden_size: int):
        super().__init__()
        self.table = nn.Embedding(3, hidden_size)
        nn.init.normal_(self.table.weight, std=0.02)

    def forward(
        self, gender: torch.Tensor, drop_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        index = gender.long()
        if drop_mask is not None:
            index = torch.where(drop_mask, torch.full_like(index, self.UNKNOWN), index)
        return self.table(index)
