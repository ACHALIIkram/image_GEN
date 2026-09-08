"""Diffusion Transformer — F-M1 à F-M8, §7.2.

**Exigence académique (§1.3) : ce fichier est écrit à la main.** Aucune classe modèle
n'est importée d'une bibliothèque ; seuls des primitives `torch.nn` (Linear, LayerNorm,
Embedding) et `F.scaled_dot_product_attention` sont utilisés. Le planning de bruit, lui,
vient de `diffusers` — c'est explicitement autorisé par F-T2.

Architecture — la profondeur, la largeur et la taille de patch viennent de la config :

    latent 16×16×4
      → patchify (patch_size)                    → T tokens de dim hidden_size
      → + embedding positionnel appris
      → depth × DiTBlock(adaLN-Zero)             conditionnés par c
      → FinalLayer(adaLN)                        → T × (patch_size²·4)
      → unpatchify                               → ε̂ 16×16×4

Deux points de fonctionnement, à ne pas confondre :

  * `configs/base.yaml` — patch 2, depth 8, hidden 256 → 64 jetons, 9,95 M paramètres.
    C'est le prototype, conservé pour les tests de fumée et le palier v1.
  * `configs/final_v2.yaml` et ses héritiers — patch 1, depth 12, hidden 384 → 256
    jetons, 32,96 M paramètres. C'est le modèle livré.

L'écart de FID entre les deux est d'un facteur 1,8, et il vient d'abord de la taille de
patch, pas du nombre de paramètres : à patch 2, un jeton couvre 16×16 pixels de l'image
décodée, soit un œil entier. Le modèle ne peut pas placer un détail plus fin que son
jeton.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from facedit.models.embeddings import (
    FourierAttributeEmbedder,
    GenderEmbedder,
    TimestepEmbedder,
    modulate,
)


# --------------------------------------------------------------------------------------
# Briques
# --------------------------------------------------------------------------------------


class Attention(nn.Module):
    """Auto-attention multi-tête. Aucun masque : tous les jetons se voient."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, T, head_dim
        query, key, value = qkv.unbind(0)

        # SDPA plutôt qu'un produit matriciel explicite : sur 8 Go de VRAM, le noyau
        # fusionné évite de matérialiser la matrice d'attention B×heads×T×T.
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(out)


class Mlp(nn.Module):
    """MLP à deux couches, activation GELU tanh (celle du DiT de référence)."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class DiTBlock(nn.Module):
    """Bloc Transformer conditionné par adaLN-Zero (F-M3).

    Six paramètres sont dérivés de `c` par une unique couche linéaire **initialisée à
    zéro** : shift/scale/gate pour l'attention, idem pour le MLP. À l'initialisation les
    portes valent 0, chaque bloc est donc l'identité et le réseau entier calcule une
    fonction constante. L'entraînement démarre sans warmup et sans divergence — c'est
    tout l'intérêt du « -Zero ».

    Les LayerNorm sont sans paramètres affines : l'échelle et le décalage viennent
    exclusivement de la condition, sinon les deux se concurrenceraient.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(
            c
        ).chunk(6, dim=-1)
        x = x + gate_attn.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_attn, scale_attn))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """Projection des tokens vers les patchs de bruit prédits, également zero-init."""

    def __init__(self, dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.proj = nn.Linear(dim, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.ada_ln(c).chunk(2, dim=-1)
        return self.proj(modulate(self.norm(x), shift, scale))


# --------------------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------------------


class DiT(nn.Module):
    """DiT conditionné par (âge, genre, ITA).

    Contrat §6.3 : `DiT.forward(x, t, age, gender, ita) -> eps_pred`
      x      (B, 4, 16, 16) float
      t      (B,)           pas de diffusion
      age    (B,)           années, unités physiques
      gender (B,)           long, 0 = homme, 1 = femme, 2 = inconnu
      ita    (B,)           degrés, unités physiques
    """

    ATTRIBUTES: Tuple[str, str, str] = ("age", "gender", "ita")

    def __init__(
        self,
        latent_size: int = 16,
        in_channels: int = 4,
        patch_size: int = 2,
        depth: int = 8,
        hidden_size: int = 256,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        fourier_num_freqs: int = 64,
        fourier_max_freq: float = 64.0,
        dropout: float = 0.0,
        age_norm_max: float = 70.0,
        ita_norm_offset: float = 60.0,
        ita_norm_scale: float = 120.0,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError(f"hidden_size={hidden_size} non divisible par num_heads={num_heads}")
        if latent_size % patch_size:
            raise ValueError(f"latent_size={latent_size} non divisible par patch_size={patch_size}")

        self.latent_size = latent_size
        self.in_channels = in_channels
        self.out_channels = in_channels  # cible `epsilon`, pas de variance apprise (§7.3)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.grid_size = latent_size // patch_size
        self.num_patches = self.grid_size**2

        # --- entrée (F-M2) ---
        patch_dim = patch_size * patch_size * in_channels
        self.patch_proj = nn.Linear(patch_dim, hidden_size, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # --- conditionnement (F-M4, F-M5, F-M6) ---
        self.t_embed = TimestepEmbedder(hidden_size)
        self.age_embed = FourierAttributeEmbedder(
            hidden_size, fourier_num_freqs, fourier_max_freq, offset=0.0, scale=age_norm_max
        )
        self.gender_embed = GenderEmbedder(hidden_size)
        self.ita_embed = FourierAttributeEmbedder(
            hidden_size,
            fourier_num_freqs,
            fourier_max_freq,
            offset=ita_norm_offset,
            scale=ita_norm_scale,
        )

        # --- corps ---
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.final = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

    # ---------------------------------------------------------------------- init
    def initialize_weights(self) -> None:
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Les embeddings ont déjà leur propre initialisation ; `basic` l'a écrasée pour
        # leurs couches linéaires, on la restaure.
        for module in (self.t_embed, self.age_embed, self.ita_embed):
            nn.init.normal_(module.mlp[0].weight, std=0.02)
            nn.init.normal_(module.mlp[2].weight, std=0.02)
        nn.init.normal_(self.gender_embed.table.weight, std=0.02)

        # adaLN-Zero : toutes les modulations partent de zéro.
        for block in self.blocks:
            nn.init.zeros_(block.ada_ln[-1].weight)
            nn.init.zeros_(block.ada_ln[-1].bias)
        nn.init.zeros_(self.final.ada_ln[-1].weight)
        nn.init.zeros_(self.final.ada_ln[-1].bias)
        nn.init.zeros_(self.final.proj.weight)
        nn.init.zeros_(self.final.proj.bias)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------- patchify
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, S, S) → (B, T, C·p²). Réécrit à la main plutôt que via une Conv2d
        de stride p, pour que la correspondance token ↔ patch soit explicite."""
        batch, channels, height, width = x.shape
        patch = self.patch_size
        x = x.reshape(batch, channels, height // patch, patch, width // patch, patch)
        x = x.permute(0, 2, 4, 1, 3, 5)  # B, gh, gw, C, p, p
        return x.reshape(batch, self.num_patches, channels * patch * patch)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        """(B, T, C·p²) → (B, C, S, S). Inverse exact de `patchify`."""
        batch = tokens.shape[0]
        patch, grid, channels = self.patch_size, self.grid_size, self.out_channels
        x = tokens.reshape(batch, grid, grid, channels, patch, patch)
        x = x.permute(0, 3, 1, 4, 2, 5)  # B, C, gh, p, gw, p
        return x.reshape(batch, channels, grid * patch, grid * patch)

    # ---------------------------------------------------------------- condition
    def embed_attributes(
        self,
        age: torch.Tensor,
        gender: torch.Tensor,
        ita: torch.Tensor,
        drop: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Partie du vecteur de condition indépendante du pas de diffusion.

        Séparée de `forward` parce que, pendant l'échantillonnage, les attributs sont
        constants sur les 25 pas : on l'évalue une fois et on la réutilise. C'est aussi
        le vecteur qu'interpole F-S4.
        """
        drop = drop or {}
        return (
            self.age_embed(age, drop.get("age"))
            + self.gender_embed(gender, drop.get("gender"))
            + self.ita_embed(ita, drop.get("ita"))
        )

    def null_attributes(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Vecteur de condition entièrement nul — la branche inconditionnelle du CFG."""
        ones = torch.ones(batch_size, dtype=torch.bool, device=device)
        zeros = torch.zeros(batch_size, device=device)
        return self.embed_attributes(
            zeros,
            torch.zeros(batch_size, dtype=torch.long, device=device),
            zeros,
            drop={"age": ones, "gender": ones, "ita": ones},
        )

    # ------------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        age: Optional[torch.Tensor] = None,
        gender: Optional[torch.Tensor] = None,
        ita: Optional[torch.Tensor] = None,
        drop: Optional[Dict[str, torch.Tensor]] = None,
        attr_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Prédit le bruit ε̂. Voir le contrat §6.3.

        `attr_emb` court-circuite les trois attributs par un vecteur de condition déjà
        calculé (interpolation F-S4, ou condition nulle pré-calculée du CFG).
        """
        if attr_emb is None:
            if age is None or gender is None or ita is None:
                raise ValueError("Fournir (age, gender, ita) ou bien attr_emb")
            attr_emb = self.embed_attributes(age, gender, ita, drop)

        c = self.t_embed(t) + attr_emb

        tokens = self.patch_proj(self.patchify(x)) + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens, c)
        return self.unpatchify(self.final(tokens, c))

    # ----------------------------------------------------------------------- CFG
    @torch.no_grad()
    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attr_emb: torch.Tensor,
        null_emb: torch.Tensor,
        guidance: float,
    ) -> torch.Tensor:
        """Classifier-free guidance à échelle unique (F-S2).

        Une seule passe avant sur un lot dupliqué : moins d'allers-retours noyau que
        deux appels séparés, pour un coût VRAM identique à batch effectif égal.
        """
        if guidance == 1.0:
            return self.forward(x, t, attr_emb=attr_emb)

        doubled_x = torch.cat([x, x], dim=0)
        doubled_t = torch.cat([t, t], dim=0)
        doubled_c = torch.cat([attr_emb, null_emb], dim=0)
        cond, uncond = self.forward(doubled_x, doubled_t, attr_emb=doubled_c).chunk(2, dim=0)
        return uncond + guidance * (cond - uncond)

    @torch.no_grad()
    def forward_with_per_attribute_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        age: torch.Tensor,
        gender: torch.Tensor,
        ita: torch.Tensor,
        w_age: float,
        w_gender: float,
        w_ita: float,
    ) -> torch.Tensor:
        """Guidage à trois échelles indépendantes (F-S6, prio C).

        Décomposition télescopique, dans l'ordre âge → genre → ITA :

            ε = ε_∅
              + w_age    (ε_A    − ε_∅)
              + w_gender (ε_AG   − ε_A)
              + w_ita    (ε_AGI  − ε_AG)

        Propriété qui justifie ce choix plutôt qu'une variante « leave-one-out » :
        lorsque les trois échelles sont égales à w, l'expression se réduit **exactement**
        à ε_∅ + w(ε_AGI − ε_∅), c'est-à-dire au CFG standard. Le mode par attribut est
        donc une généralisation stricte du mode par défaut, et le balayage F-E6 reste
        comparable.

        Contrepartie assumée : la décomposition n'est pas symétrique. `w_age` pilote le
        passage « rien → âge » tandis que `w_ita` pilote « âge+genre → âge+genre+ITA ».
        L'ordre est figé ici pour que les mesures restent comparables entre exécutions.

        Coût : 4 passes avant par pas au lieu de 2.
        """
        batch = x.shape[0]
        device = x.device
        yes = torch.ones(batch, dtype=torch.bool, device=device)
        no = torch.zeros(batch, dtype=torch.bool, device=device)

        conditions = [
            {"age": yes, "gender": yes, "ita": yes},  # ε_∅
            {"age": no, "gender": yes, "ita": yes},  # ε_A
            {"age": no, "gender": no, "ita": yes},  # ε_AG
            {"age": no, "gender": no, "ita": no},  # ε_AGI
        ]
        embeddings = torch.cat(
            [self.embed_attributes(age, gender, ita, drop=d) for d in conditions], dim=0
        )
        stacked_x = x.repeat(4, 1, 1, 1)
        stacked_t = t.repeat(4)
        eps_null, eps_a, eps_ag, eps_agi = self.forward(
            stacked_x, stacked_t, attr_emb=embeddings
        ).chunk(4, dim=0)

        return (
            eps_null
            + w_age * (eps_a - eps_null)
            + w_gender * (eps_ag - eps_a)
            + w_ita * (eps_agi - eps_ag)
        )


# --------------------------------------------------------------------------------------
# Fabrique
# --------------------------------------------------------------------------------------


def build_dit(model_cfg) -> DiT:
    """Instancie le DiT depuis un `ModelConfig`."""
    return DiT(
        latent_size=model_cfg.latent_size,
        in_channels=model_cfg.in_channels,
        patch_size=model_cfg.patch_size,
        depth=model_cfg.depth,
        hidden_size=model_cfg.hidden_size,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        fourier_num_freqs=model_cfg.fourier_num_freqs,
        fourier_max_freq=model_cfg.fourier_max_freq,
        dropout=model_cfg.dropout,
        age_norm_max=model_cfg.age_norm_max,
        ita_norm_offset=model_cfg.ita_norm_offset,
        ita_norm_scale=model_cfg.ita_norm_scale,
    )
