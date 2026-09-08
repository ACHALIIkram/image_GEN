"""Reproductibilité : seeds, empreintes de fichiers, empreinte d'environnement.

NF-6 impose qu'une expérience soit rejouable depuis un fichier de configuration et une
seed. `results.json` (F-E10) embarque en plus le hash du checkpoint évalué et
l'empreinte d'environnement, parce qu'une seed ne suffit pas à reproduire un résultat
si la version de PyTorch ou du VAE a changé entre-temps.
"""

from __future__ import annotations

import hashlib
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Fixe les RNG Python / NumPy / PyTorch (CPU et CUDA).

    `deterministic=True` force les noyaux cuDNN déterministes : reproductibilité
    bit-à-bit au prix d'un ralentissement notable. Réservé à l'évaluation, pas à
    l'entraînement.

    **Cette fonction ne touche délibérément pas à `cudnn.benchmark`.** L'autotuning de
    cuDNN essaie plusieurs algorithmes de convolution au premier appel de chaque forme,
    et certains candidats réservent un espace de travail considérable : mesuré sur ce
    projet, il fait passer le pic VRAM d'inférence de **0.49 Go à 5.09 Go**, ce qui viole
    NF-3 (< 4 Go) et met en danger une carte de 8 Go.

    La contrepartie serait nulle : cuDNN ne gouverne que les convolutions, or le DiT n'en
    contient aucune (uniquement des `Linear` et de l'attention). Seul le décodeur VAE est
    convolutif, et il tourne à formes fixes en inférence. L'autotuning ne peut donc rien
    accélérer de ce qui compte, tout en coûtant 4.6 Go de pic transitoire.

    Activer l'autotuning reste possible explicitement via `enable_cudnn_autotune()`, pour
    qui voudrait le mesurer — mais ce n'est pas un effet de bord du seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:  # utilisable sans torch (préparation de données pure NumPy)
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def enable_cudnn_autotune(enabled: bool = True) -> None:
    """Active explicitement l'autotuning cuDNN. Voir l'avertissement de `seed_everything`
    sur son coût en VRAM (+4.6 Go de pic transitoire sur le décodeur VAE)."""
    import torch

    torch.backends.cudnn.benchmark = enabled


def torch_generator(seed: int, device: str = "cpu"):
    """Générateur explicite : le bruit initial d'échantillonnage ne doit jamais
    dépendre de l'état global du RNG (F-S3, seed contrôlable et reproductible)."""
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hash d'un fichier (checkpoint évalué, cache de latents)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parents[2],
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def env_fingerprint() -> Dict[str, Any]:
    """Ce qu'il faut connaître pour rejouer un résultat, au-delà de la seed."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git": _git_revision(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["gpu"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except ImportError:
        info["torch"] = None
    try:
        import diffusers

        info["diffusers"] = diffusers.__version__
    except ImportError:
        info["diffusers"] = None
    return info
