"""Dataset de latents pré-encodés — consommé par `train.py` (contrat §6.3).

Le cache tient en RAM (NF-5 : < 1 Go), on le charge donc en dur plutôt qu'en memmap :
à 128 latents par pas et 60 000 pas, l'accès disque deviendrait le goulot d'étranglement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class LatentDataset(Dataset):
    """Latents VAE + attributs. Renvoie exactement ce que `DiT.forward` consomme.

    L'âge est re-tiré uniformément dans sa tranche à chaque accès quand `age_bounds.npy`
    est présent (F-D3) ; à défaut, le centre de tranche stocké dans `labels[:, 0]` est
    renvoyé tel quel. Le tirage passe par le RNG de Torch, donc il hérite du seeding
    par worker du `DataLoader` et reste reproductible sous `seed_everything`.
    """

    def __init__(
        self,
        latents_path: str | Path,
        labels_path: str | Path,
        age_bounds_path: Optional[str | Path] = None,
        mmap: bool = False,
        resample_age: bool = True,
    ):
        latents_path, labels_path = Path(latents_path), Path(labels_path)
        if not latents_path.exists():
            raise FileNotFoundError(
                f"{latents_path} absent. Lancer d'abord :\n"
                f"  python -m facedit.data.encode --config <config> --split train"
            )

        self.latents = np.load(latents_path, mmap_mode="r" if mmap else None)
        self.labels = np.load(labels_path)

        if self.latents.shape[0] != self.labels.shape[0]:
            raise ValueError(
                f"Cache incohérent : {self.latents.shape[0]} latents pour "
                f"{self.labels.shape[0]} labels"
            )
        if self.labels.shape[1] != 3:
            raise ValueError(
                f"labels attendu de forme (N, 3) [age, gender, ita], reçu {self.labels.shape}"
            )

        self.age_bounds = None
        if resample_age and age_bounds_path is not None and Path(age_bounds_path).exists():
            bounds = np.load(age_bounds_path)
            if bounds.shape[0] == self.labels.shape[0]:
                self.age_bounds = bounds

        self.latent_size = self.latents.shape[-1]
        self.in_channels = self.latents.shape[1]

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        latent = torch.from_numpy(np.asarray(self.latents[index], dtype=np.float32))
        label = self.labels[index]

        if self.age_bounds is not None:
            low, high = self.age_bounds[index]
            age = float(low) + (float(high) - float(low)) * torch.rand(()).item()
        else:
            age = float(label[0])

        return {
            "latent": latent,
            "age": torch.tensor(age, dtype=torch.float32),
            "gender": torch.tensor(int(label[1]), dtype=torch.long),
            "ita": torch.tensor(float(label[2]), dtype=torch.float32),
        }

    # ------------------------------------------------------------------ introspection
    def stats(self) -> Dict[str, float]:
        """Statistiques utilisées pour cadrer les sliders de l'interface et les bins d'éval."""
        ages, genders, itas = self.labels[:, 0], self.labels[:, 1], self.labels[:, 2]
        return {
            "n": int(len(self)),
            "latent_size": int(self.latent_size),
            "age_min": float(ages.min()),
            "age_max": float(ages.max()),
            "age_mean": float(ages.mean()),
            "female_ratio": float((genders == 1).mean()),
            "ita_mean": float(itas.mean()),
            "ita_std": float(itas.std()),
            "ita_p01": float(np.percentile(itas, 1)),
            "ita_p99": float(np.percentile(itas, 99)),
            # `.std()` sur un tableau float16 accumule la somme des carrés en float16 :
            # 2048x4x16x16 termes d'ordre 1 dépassent le max float16 (65504) et la
            # statistique remonte `inf`. Le cast en float32 est obligatoire, pas cosmétique.
            "latent_std": float(
                np.asarray(self.latents[: min(2048, len(self))], dtype=np.float32).std()
            ),
        }


class InfiniteSampler(Sampler[int]):
    """Permutations enchaînées sans fin : l'entraînement se compte en pas, pas en époques.

    Un `RandomSampler` classique impose de gérer la fin d'époque dans la boucle et de
    recréer un itérateur, ce qui relance les workers du `DataLoader` toutes les ~590 pas
    à batch 128 sur 75 k images.
    """

    def __init__(self, size: int, seed: int = 0):
        self.size = size
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        while True:
            yield from torch.randperm(self.size, generator=generator).tolist()

    def __len__(self) -> int:
        return 1 << 62  # borne factice : le sampler est infini par construction


def build_loader(cfg, split: str = "train") -> torch.utils.data.DataLoader:
    """`DataLoader` infini prêt pour la boucle d'entraînement."""
    dataset = LatentDataset(
        cfg.data.latents_path(split),
        cfg.data.labels_path(split),
        cfg.data.age_bounds_path(split),
    )
    if dataset.latent_size != cfg.model.latent_size:
        raise ValueError(
            f"Le cache contient des latents {dataset.latent_size}×{dataset.latent_size} "
            f"mais model.latent_size={cfg.model.latent_size}. Ré-encoder ou corriger la config."
        )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        sampler=InfiniteSampler(len(dataset), seed=cfg.seed),
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.data.num_workers > 0,
        prefetch_factor=4 if cfg.data.num_workers > 0 else None,
    )
