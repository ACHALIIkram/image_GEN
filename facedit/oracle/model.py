"""Oracle d'attributs — F-O1, F-O3.

Un **unique** ResNet-18 pré-entraîné ImageNet portant trois têtes : genre (2 classes),
âge (régression scalaire), ITA (régression scalaire).

Un seul tronc plutôt que trois réseaux : les trois tâches partagent les mêmes traits
bas niveau, et surtout un tronc partagé rend la mesure des trois attributs cohérente
sur une même image — trois réseaux indépendants pourraient être en désaccord sur ce
qu'ils regardent, ce qui rendrait la matrice de fuite (F-E4) ininterprétable.

L'oracle est la **barre d'erreur de tout le protocole** (F-O4). Une MAE d'âge de 8 ans
mesurée sur les images générées n'a de sens que rapportée à la MAE de l'oracle sur des
images réelles : si l'oracle se trompe déjà de 5 ans sur du réel, l'écart imputable au
générateur est ce qui dépasse ces 5 ans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

# Statistiques ImageNet — le tronc est pré-entraîné avec, s'en écarter dégraderait le
# transfert.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AttributeOracle(nn.Module):
    """ResNet-18 à trois têtes. Les têtes de régression prédisent en unités normalisées.

    Les constantes de dénormalisation (`age_mean`, `age_std`, …) sont enregistrées comme
    buffers : elles voyagent dans le `state_dict`, donc un checkpoint d'oracle est
    autoportant et `predict` ne peut pas être appelé avec les mauvaises constantes.
    """

    def __init__(
        self,
        pretrained: bool = True,
        image_size: int = 128,
        age_mean: float = 40.0,
        age_std: float = 15.0,
        ita_mean: float = 25.0,
        ita_std: float = 25.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)

        self.head_gender = nn.Linear(feature_dim, 2)
        self.head_age = nn.Linear(feature_dim, 1)
        self.head_ita = nn.Linear(feature_dim, 1)

        self.image_size = image_size
        for name, value in (
            ("age_mean", age_mean),
            ("age_std", age_std),
            ("ita_mean", ita_mean),
            ("ita_std", ita_std),
        ):
            self.register_buffer(name, torch.tensor(float(value)))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    # ------------------------------------------------------------------------ forward
    def forward(self, pixels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """`pixels` : (B, 3, H, W) déjà normalisé ImageNet."""
        features = self.dropout(self.backbone(pixels))
        return {
            "gender_logits": self.head_gender(features),
            "age_norm": self.head_age(features).squeeze(-1),
            "ita_norm": self.head_ita(features).squeeze(-1),
        }

    # ------------------------------------------------------------------ (dé)normalisation
    def denormalize(self, age_norm: torch.Tensor, ita_norm: torch.Tensor):
        return age_norm * self.age_std + self.age_mean, ita_norm * self.ita_std + self.ita_mean

    def normalize_targets(self, age: torch.Tensor, ita: torch.Tensor):
        return (age - self.age_mean) / self.age_std, (ita - self.ita_mean) / self.ita_std

    def preprocess(self, images_uint8: np.ndarray) -> torch.Tensor:
        """(B, H, W, 3) uint8 → (B, 3, S, S) normalisé ImageNet, sur le bon device."""
        device = self.pixel_mean.device
        array = np.asarray(images_uint8)
        if array.ndim == 3:
            array = array[None]
        if array.dtype != np.uint8:
            raise ValueError(f"`predict` attend du uint8 (contrat §6.3), reçu {array.dtype}")

        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(0, 3, 1, 2)))
        tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
        if tensor.shape[-1] != self.image_size:
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return (tensor - self.pixel_mean) / self.pixel_std

    # -------------------------------------------------------------------------- API F-O3
    @torch.no_grad()
    def predict(self, images_uint8: np.ndarray, batch_size: int = 128) -> Dict[str, np.ndarray]:
        """Contrat §6.3 : `Oracle.predict(images_uint8) -> {gender, gender_conf, age, ita}`.

        Toutes les sorties sont des tableaux NumPy de forme (B,). `gender_conf` est la
        probabilité softmax de la classe prédite — l'interface l'affiche (F-I3) parce
        qu'un genre prédit à 0.51 et un genre prédit à 0.99 ne disent pas la même chose
        de l'obéissance du modèle.
        """
        was_training = self.training
        self.eval()
        outputs = {k: [] for k in ("gender", "gender_conf", "age", "ita")}

        array = np.asarray(images_uint8)
        if array.ndim == 3:
            array = array[None]

        for start in range(0, array.shape[0], batch_size):
            pixels = self.preprocess(array[start : start + batch_size])
            result = self(pixels)
            probabilities = torch.softmax(result["gender_logits"].float(), dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            age, ita = self.denormalize(result["age_norm"].float(), result["ita_norm"].float())

            outputs["gender"].append(predicted.cpu().numpy())
            outputs["gender_conf"].append(confidence.cpu().numpy())
            outputs["age"].append(age.cpu().numpy())
            outputs["ita"].append(ita.cpu().numpy())

        if was_training:
            self.train()
        return {k: np.concatenate(v) for k, v in outputs.items()}


# --------------------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------------------


def load_oracle(
    checkpoint_path: str | Path, device: str = "cuda", strict: bool = True
) -> AttributeOracle:
    """Charge un oracle entraîné. Ne télécharge pas les poids ImageNet (écrasés de suite)."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Oracle introuvable : {checkpoint_path}. Lancer d'abord :\n"
            f"  python -m facedit.oracle.train_oracle --config <config>"
        )
    if not torch.cuda.is_available():
        device = "cpu"

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper = payload.get("hyper", {})
    model = AttributeOracle(
        pretrained=False,
        image_size=hyper.get("image_size", 128),
        age_mean=hyper.get("age_mean", 40.0),
        age_std=hyper.get("age_std", 15.0),
        ita_mean=hyper.get("ita_mean", 25.0),
        ita_std=hyper.get("ita_std", 25.0),
    )
    model.load_state_dict(payload["model"], strict=strict)
    model.to(device).eval().requires_grad_(False)
    model.metrics = payload.get("metrics", {})  # F-O4 : la barre d'erreur voyage avec
    return model
