"""Moyenne mobile exponentielle des poids — F-T3.

« La génération utilise **toujours** les poids EMA. » Cette règle est appliquée
mécaniquement : `sample.py` charge la clé `ema` du checkpoint et refuse de démarrer si
elle est absente, plutôt que de retomber silencieusement sur les poids bruts.
"""

from __future__ import annotations

import copy
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """Copie fantôme des paramètres, mise à jour par θ_ema ← d·θ_ema + (1−d)·θ.

    Deux détails qui comptent :

    - **Rampe de décroissance.** Un decay fixe de 0.9999 a une constante de temps de
      10 000 pas ; les 10 000 premiers pas d'EMA sont alors dominés par les poids
      d'initialisation, donc du bruit. On utilise `min(decay, (1+n)/(10+n))`, qui
      démarre à ~0.1 et converge vers 0.9999. L'EMA est utilisable bien plus tôt, ce qui
      rend les grilles d'échantillons de F-T6 lisibles dès les premiers milliers de pas.
    - **Les buffers sont copiés, pas moyennés.** Les fréquences de Fourier sont des
      constantes ; les moyenner n'aurait pas de sens.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, start_step: int = 1000):
        self.decay = decay
        self.start_step = start_step
        self.num_updates = 0
        self.shadow = copy.deepcopy(model).eval()
        self.shadow.requires_grad_(False)

    # --------------------------------------------------------------------- mise à jour
    @property
    def current_decay(self) -> float:
        ramp = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        return min(self.decay, ramp)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Une mise à jour par pas d'optimisation, après `optimizer.step()`."""
        if step < self.start_step:
            # Avant le démarrage, l'EMA suit le modèle à l'identique : au pas
            # `start_step` elle part donc des poids courants, pas de l'initialisation.
            self.copy_from(model)
            return

        decay = self.current_decay
        model_params = dict(model.named_parameters())
        for name, shadow_param in self.shadow.named_parameters():
            shadow_param.lerp_(model_params[name].detach(), 1.0 - decay)
        self.num_updates += 1

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        for shadow_param, param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.copy_(param.detach())
        for shadow_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            shadow_buffer.copy_(buffer)

    # ------------------------------------------------------------------- (dé)sérialisation
    def state_dict(self) -> Dict[str, object]:
        return {"shadow": self.shadow.state_dict(), "num_updates": self.num_updates}

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.shadow.load_state_dict(state["shadow"])
        self.num_updates = int(state.get("num_updates", 0))

    def to(self, *args, **kwargs) -> "EMA":
        self.shadow.to(*args, **kwargs)
        return self
