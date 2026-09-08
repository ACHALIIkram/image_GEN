"""Baseline 1 — StyleGAN2-FFHQ + directions SVM dans l'espace latent W. §8.4, prio M.

Rôle : borne de photoréalisme à faible coût et référence historique. StyleGAN2 produit
des visages nettement plus réalistes que notre DiT-XS de 10 M paramètres entraîné cinq
heures ; l'intérêt de la comparaison n'est pas de gagner sur le FID, c'est de montrer
que la **contrôlabilité vérifiée** ne suit pas le réalisme.

Méthode (édition d'espace latent, sans aucun entraînement génératif de notre part) :

1. tirer N vecteurs latents w et rendre les images correspondantes ;
2. **les étiqueter avec notre propre oracle** — c'est le point crucial : StyleGAN2 n'a
   pas d'attributs, ce sont les nôtres qui sont projetés dans son espace ;
3. ajuster un SVM linéaire par attribut ; la normale à l'hyperplan est la direction
   d'édition ;
4. pour une condition demandée, partir d'un w moyen et se déplacer le long des trois
   directions jusqu'à atteindre les valeurs voulues, mesurées par l'oracle.

Limite à énoncer dans le rapport : les trois directions ne sont pas orthogonales. La
fuite entre attributs est donc *attendue* pour cette baseline — et c'est précisément ce
que la matrice F-E4 doit chiffrer, sur elle comme sur notre modèle.

Poids requis (non redistribués, licence NVIDIA) :
    https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan2/versions/1/files/stylegan2-ffhq-512x512.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from baselines.base import BaselineGenerator


class StyleGAN2Baseline(BaselineGenerator):
    label = "stylegan2-svm"

    def __init__(
        self,
        cfg,
        pickle_path: str,
        oracle,
        num_fit_samples: int = 4000,
        truncation: float = 0.7,
        device: str = "cuda",
        cache_path: Optional[str] = None,
    ):
        super().__init__(cfg)
        self.device = device
        self.truncation = truncation
        self.oracle = oracle

        with open(pickle_path, "rb") as handle:
            self.G = pickle.load(handle)["G_ema"].to(device).eval()

        cache = Path(cache_path or f"artifacts/stylegan_directions_{cfg.data.image_size}.npz")
        if cache.exists():
            payload = np.load(cache)
            self.directions = {k: payload[k] for k in ("age", "gender", "ita")}
            self.w_mean = payload["w_mean"]
            self.attribute_stats = json.loads(str(payload["stats"]))
            print(f"[stylegan] directions relues depuis {cache}")
        else:
            self._fit_directions(num_fit_samples, cache)

    # ------------------------------------------------------------------------ ajustement
    @torch.no_grad()
    def _fit_directions(self, num_samples: int, cache: Path) -> None:
        from sklearn.svm import LinearSVC

        from facedit.data.fairface import resize_uint8, to_uint8
        from facedit.data.ita import compute_ita_batch

        latents, ages, genders, itas = [], [], [], []
        batch = 16

        for start in tqdm(range(0, num_samples, batch), desc="stylegan : échantillons"):
            count = min(batch, num_samples - start)
            z = torch.randn(count, self.G.z_dim, device=self.device)
            w = self.G.mapping(z, None, truncation_psi=self.truncation)
            images = self.G.synthesis(w, noise_mode="const")
            frames = np.stack([
                resize_uint8(img, self.image_size)
                for img in to_uint8(images.clamp(-1, 1))
            ])

            prediction = self.oracle.predict(frames)
            # L'ITA est mesuré analytiquement, comme partout ailleurs dans le protocole :
            # utiliser la tête ITA de l'oracle ici introduirait une boucle (on ajusterait
            # une direction sur la sortie d'un réseau, puis on l'évaluerait avec lui).
            analytic, valid = compute_ita_batch(frames, use_mediapipe=False)

            latents.append(w[:, 0].cpu().numpy())  # W (et non W+) : une direction unique
            ages.append(prediction["age"])
            genders.append(prediction["gender"])
            itas.append(np.where(valid, analytic, np.nan))

        latents = np.concatenate(latents)
        ages = np.concatenate(ages)
        genders = np.concatenate(genders)
        itas = np.concatenate(itas)

        self.w_mean = latents.mean(axis=0)
        self.directions = {}
        self.attribute_stats = {}

        for name, values in (("age", ages), ("gender", genders.astype(float)), ("ita", itas)):
            mask = np.isfinite(values)
            x, y = latents[mask], values[mask]
            if name == "gender":
                labels = (y > 0.5).astype(int)
            else:
                # Séparation par les tercles extrêmes : entraîner le SVM sur les valeurs
                # médianes n'apporte que du bruit, la frontière utile est aux extrêmes.
                low, high = np.percentile(y, [33, 67])
                keep = (y <= low) | (y >= high)
                x, labels = x[keep], (y[keep] >= high).astype(int)

            svm = LinearSVC(C=1.0, max_iter=5000, dual="auto").fit(x, labels)
            direction = svm.coef_[0]
            self.directions[name] = direction / np.linalg.norm(direction)
            self.attribute_stats[name] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "svm_train_accuracy": float(svm.score(x, labels)),
                "n": int(len(x)),
            }
            print(f"[stylegan] direction {name} · accuracy SVM {svm.score(x, labels):.3f}")

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache, w_mean=self.w_mean, stats=json.dumps(self.attribute_stats),
            **self.directions,
        )

    # -------------------------------------------------------------------------- rendu
    @torch.no_grad()
    def _render(self, age, gender, ita, seed: int) -> np.ndarray:
        from facedit.data.fairface import resize_uint8, to_uint8

        count = len(age)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        z = torch.randn(count, self.G.z_dim, generator=generator).to(self.device)
        w = self.G.mapping(z, None, truncation_psi=self.truncation)

        # Déplacement le long de chaque direction, proportionnel à l'écart entre la
        # valeur demandée et la moyenne de la population StyleGAN, en écarts-types.
        # L'échelle (`STEP`) est empirique : c'est la faiblesse assumée de l'approche par
        # directions, il n'y a pas d'unité naturelle entre l'espace W et les années.
        STEP = 2.5
        offset = np.zeros((count, w.shape[-1]), dtype=np.float32)
        for name, requested in (("age", age), ("gender", gender.astype(float)), ("ita", ita)):
            stats = self.attribute_stats[name]
            magnitude = (requested - stats["mean"]) / max(stats["std"], 1e-6)
            offset += STEP * magnitude[:, None] * self.directions[name][None, :]

        w = w + torch.from_numpy(offset).to(self.device)[:, None, :]
        images = self.G.synthesis(w, noise_mode="const")
        return np.stack([
            resize_uint8(img, self.image_size) for img in to_uint8(images.clamp(-1, 1))
        ])


def main() -> None:
    from facedit.eval.run_eval import run_eval
    from facedit.oracle.model import load_oracle
    from facedit.utils.config import load_config

    parser = argparse.ArgumentParser(description="Baseline StyleGAN2 + directions SVM (§8.4)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pkl", required=True, help="stylegan2-ffhq-*.pkl (NVIDIA)")
    parser.add_argument("--fit-samples", type=int, default=4000)
    parser.add_argument("--truncation", type=float, default=0.7)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--only", nargs="*",
        default=["fid", "attributes", "leakage", "diversity"],
        help="Sections du harnais. `w_sweep` n'a pas de sens ici : il n'y a pas de CFG.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    oracle = load_oracle(cfg.eval.oracle_ckpt, cfg.device)
    generator = StyleGAN2Baseline(
        cfg, args.pkl, oracle, args.fit_samples, args.truncation, cfg.device
    )
    run_eval(
        checkpoint_path=args.pkl, cfg=cfg, quick=args.quick, only=args.only,
        generator=generator, label="stylegan2-svm",
    )


if __name__ == "__main__":
    main()
