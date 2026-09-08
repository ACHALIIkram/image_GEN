"""Échantillonnage — F-S1 à F-S6, §7.4.

DDIM 25 pas, `eta=0`, classifier-free guidance à échelle paramétrable, seed contrôlable.
Le planning de bruit provient de `diffusers` (F-T2) ; seul le DiT est de nous.

Règle non négociable (F-T3) : la génération utilise **toujours** les poids EMA.
`load_generator` échoue explicitement si le checkpoint n'en contient pas.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from facedit.data.fairface import GENDER_UNKNOWN, to_uint8
from facedit.models.dit import DiT, build_dit
from facedit.utils.config import Config, SampleConfig
from facedit.utils.repro import torch_generator


# --------------------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------------------


@dataclass
class Generator:
    """Tout ce qu'il faut pour produire une image : modèle EMA, VAE gelé, planning.

    C'est aussi le **contrat que les baselines doivent honorer** (§8.4 : « toutes les
    baselines passent par le même harnais d'évaluation »). Le harnais n'appelle jamais
    que `generator.generate(...)`, `generator.device` et `generator.label` ; n'importe
    quel objet exposant cette surface — StyleGAN2, FLUX — est évaluable à l'identique.
    Voir `baselines/base.py`.
    """

    model: DiT
    cfg: Config
    device: str
    scheduler: object
    vae: Optional[object] = None
    step: int = 0
    checkpoint_path: Optional[str] = None
    label: str = "facedit-dit"

    @property
    def latent_shape(self) -> Tuple[int, int, int]:
        return (self.model.in_channels, self.model.latent_size, self.model.latent_size)

    def generate(
        self,
        age,
        gender,
        ita,
        batch_size: int = 1,
        sample_cfg: Optional[SampleConfig] = None,
        seed: Optional[int] = None,
        decode_images: bool = True,
    ) -> Dict[str, object]:
        """Surface appelée par `eval/`. Voir la fonction `generate` du module."""
        return generate(self, age, gender, ita, batch_size, sample_cfg, seed, decode_images)


def build_ddim_scheduler(cfg: Config):
    """`DDIMScheduler` aligné sur le planning d'entraînement.

    `clip_sample=False` est impératif : le défaut de diffusers borne l'échantillon à
    [-1, 1], ce qui est correct pour des pixels mais destructeur pour des latents VAE,
    dont l'écart-type après mise à l'échelle avoisine 1 et dont la queue dépasse
    largement 1 en valeur absolue.
    """
    from diffusers import DDIMScheduler

    return DDIMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )


def load_generator(
    checkpoint_path: str | Path,
    device: str = "cuda",
    with_vae: bool = True,
    config_override: Optional[Config] = None,
) -> Generator:
    """Charge un checkpoint et renvoie un `Generator` prêt à l'emploi."""
    from facedit.utils.config import Config as ConfigCls
    from facedit.utils.config import _from_dict  # reconstruction depuis le dict embarqué

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "ema" not in payload:
        raise ValueError(
            f"{checkpoint_path} ne contient pas de poids EMA. F-T3 impose que la "
            f"génération les utilise ; refus de retomber sur les poids bruts."
        )

    cfg = config_override or _from_dict(ConfigCls, payload["config"])
    if not torch.cuda.is_available():
        device = "cpu"

    model = build_dit(cfg.model)
    model.load_state_dict(payload["ema"]["shadow"])
    model.to(device).eval().requires_grad_(False)

    vae = None
    if with_vae:
        from facedit.data.encode import load_vae

        vae = load_vae(
            cfg.data.vae_id, device, torch.float16 if device == "cuda" else torch.float32
        )

    return Generator(
        model=model,
        cfg=cfg,
        device=device,
        scheduler=build_ddim_scheduler(cfg),
        vae=vae,
        step=int(payload.get("step", 0)),
        checkpoint_path=str(checkpoint_path),
    )


# --------------------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------------------


def as_tensor(value, size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Diffuse un scalaire ou une séquence vers un tenseur (size,)."""
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    elif np.isscalar(value):
        tensor = torch.full((size,), float(value), device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(np.asarray(value), device=device, dtype=dtype)
    if tensor.numel() == 1 and size > 1:
        tensor = tensor.expand(size).clone()
    if tensor.shape[0] != size:
        raise ValueError(f"Attendu {size} valeurs de condition, reçu {tensor.shape[0]}")
    return tensor


def encode_conditions_soft_gender(
    generator: Generator, age, gender_probability, ita, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Variante à genre continu, utilisée par le slider de l'interface (F-I1).

    Le PRD place le curseur de genre à 0.5 par défaut, donc sur une échelle continue,
    alors que F-M5 impose une table apprise à 3 entrées. Les deux sont conciliés en
    interpolant les **embeddings** : `p·emb(femme) + (1−p)·emb(homme)`.

    À p = 0 et p = 1 on retombe exactement sur les jetons appris, donc sur le
    comportement mesuré par l'évaluation ; les valeurs intermédiaires sont une
    exploration de l'espace de condition, pas une classe de genre. C'est le compromis
    que la question ouverte n°3 tranche : les labels sources restent binaires, seule
    l'exploration est continue.
    """
    device = generator.device
    model = generator.model
    age_t = as_tensor(age, batch_size, device, torch.float32)
    ita_t = as_tensor(ita, batch_size, device, torch.float32)
    probability = as_tensor(gender_probability, batch_size, device, torch.float32).clamp(0.0, 1.0)

    with torch.no_grad():
        male = model.gender_embed.table.weight[0]
        female = model.gender_embed.table.weight[1]
        gender_emb = (1.0 - probability)[:, None] * male + probability[:, None] * female
        attr = model.age_embed(age_t) + gender_emb + model.ita_embed(ita_t)
        null = model.null_attributes(batch_size, torch.device(device))
    return attr, null


def encode_conditions(
    generator: Generator, age, gender, ita, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(attr_emb, null_emb) pour un lot. Les attributs sont en unités physiques."""
    device = generator.device
    model = generator.model
    age_t = as_tensor(age, batch_size, device, torch.float32)
    gender_t = as_tensor(gender, batch_size, device, torch.long)
    ita_t = as_tensor(ita, batch_size, device, torch.float32)

    if int(gender_t.max()) > GENDER_UNKNOWN or int(gender_t.min()) < 0:
        raise ValueError("`gender` doit valoir 0 (homme), 1 (femme) ou 2 (inconnu)")

    with torch.no_grad():
        attr = model.embed_attributes(age_t, gender_t, ita_t)
        null = model.null_attributes(batch_size, torch.device(device))
    return attr, null


# --------------------------------------------------------------------------------------
# Boucle DDIM
# --------------------------------------------------------------------------------------


@torch.no_grad()
def sample_latents(
    generator: Generator,
    attr_emb: torch.Tensor,
    null_emb: torch.Tensor,
    num_steps: int = 25,
    guidance: float = 3.0,
    eta: float = 0.0,
    seed: Optional[int] = None,
    noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boucle DDIM (F-S1, F-S2, F-S3). Renvoie des latents (B, 4, S, S) mis à l'échelle.

    Le bruit initial est tiré depuis un générateur explicite plutôt que depuis le RNG
    global : c'est ce qui rend « seed verrouillée » (F-I4) exact, y compris si l'appelant
    a consommé du hasard entre deux générations.
    """
    device, model, scheduler = generator.device, generator.model, generator.scheduler
    batch = attr_emb.shape[0]

    if noise is None:
        gen = torch_generator(seed if seed is not None else 0, device="cpu")
        noise = torch.randn(
            (batch, *generator.latent_shape), generator=gen, dtype=torch.float32
        ).to(device)
    latents = noise.to(device) * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_steps, device=device)
    for timestep in scheduler.timesteps:
        t_batch = timestep.expand(batch).to(device)
        eps = model.forward_with_cfg(latents, t_batch, attr_emb, null_emb, guidance)
        latents = scheduler.step(eps, timestep, latents, eta=eta).prev_sample

    return latents


@torch.no_grad()
def sample_latents_per_attribute(
    generator: Generator,
    age,
    gender,
    ita,
    batch_size: int,
    num_steps: int = 25,
    w_age: float = 3.0,
    w_gender: float = 3.0,
    w_ita: float = 3.0,
    eta: float = 0.0,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Variante F-S6 : trois échelles de guidage indépendantes, 4 passes par pas."""
    device, model, scheduler = generator.device, generator.model, generator.scheduler
    age_t = as_tensor(age, batch_size, device, torch.float32)
    gender_t = as_tensor(gender, batch_size, device, torch.long)
    ita_t = as_tensor(ita, batch_size, device, torch.float32)

    gen = torch_generator(seed if seed is not None else 0, device="cpu")
    latents = torch.randn(
        (batch_size, *generator.latent_shape), generator=gen, dtype=torch.float32
    ).to(device)
    latents = latents * scheduler.init_noise_sigma

    scheduler.set_timesteps(num_steps, device=device)
    for timestep in scheduler.timesteps:
        t_batch = timestep.expand(batch_size).to(device)
        eps = model.forward_with_per_attribute_cfg(
            latents, t_batch, age_t, gender_t, ita_t, w_age, w_gender, w_ita
        )
        latents = scheduler.step(eps, timestep, latents, eta=eta).prev_sample
    return latents


@torch.no_grad()
def decode(generator: Generator, latents: torch.Tensor, chunk: int = 32) -> np.ndarray:
    """Latents → images (B, H, W, 3) uint8, par tranches pour tenir dans 4 Go (NF-3)."""
    if generator.vae is None:
        raise RuntimeError("Ce Generator a été chargé sans VAE (with_vae=False)")
    from facedit.data.encode import decode_latents

    outputs = [
        decode_latents(generator.vae, latents[i : i + chunk], generator.cfg.data.vae_scale)
        for i in range(0, latents.shape[0], chunk)
    ]
    return np.concatenate(outputs, axis=0)


def generate(
    generator: Generator,
    age,
    gender,
    ita,
    batch_size: int = 1,
    sample_cfg: Optional[SampleConfig] = None,
    seed: Optional[int] = None,
    decode_images: bool = True,
) -> Dict[str, object]:
    """Point d'entrée de haut niveau utilisé par `eval/` et par l'interface."""
    sample_cfg = sample_cfg or generator.cfg.sample
    per_attribute = any(
        w is not None
        for w in (sample_cfg.guidance_age, sample_cfg.guidance_gender, sample_cfg.guidance_ita)
    )

    if per_attribute:
        latents = sample_latents_per_attribute(
            generator,
            age,
            gender,
            ita,
            batch_size,
            num_steps=sample_cfg.num_steps,
            w_age=sample_cfg.guidance_age if sample_cfg.guidance_age is not None else sample_cfg.guidance,
            w_gender=sample_cfg.guidance_gender
            if sample_cfg.guidance_gender is not None
            else sample_cfg.guidance,
            w_ita=sample_cfg.guidance_ita if sample_cfg.guidance_ita is not None else sample_cfg.guidance,
            eta=sample_cfg.eta,
            seed=seed,
        )
    else:
        attr, null = encode_conditions(generator, age, gender, ita, batch_size)
        latents = sample_latents(
            generator,
            attr,
            null,
            num_steps=sample_cfg.num_steps,
            guidance=sample_cfg.guidance,
            eta=sample_cfg.eta,
            seed=seed,
        )

    result: Dict[str, object] = {"latents": latents}
    if decode_images:
        images = decode(generator, latents)
        if sample_cfg.restore:
            images = restore_faces(images, sample_cfg)
        result["images"] = images
    return result


# --------------------------------------------------------------------------------------
# Interpolation — F-S4
# --------------------------------------------------------------------------------------


@torch.no_grad()
def interpolate(
    generator: Generator,
    attrs_a: Tuple[float, int, float],
    attrs_b: Tuple[float, int, float],
    num_frames: int = 8,
    seed: int = 0,
    sample_cfg: Optional[SampleConfig] = None,
) -> Dict[str, object]:
    """Transition entre deux jeux d'attributs, à seed fixe (F-S4, F-I5).

    C'est le **vecteur de condition** qui est interpolé, pas les valeurs d'attributs.
    La différence n'est pas cosmétique : le genre est une table à trois entrées, donc
    interpoler `gender=0 → 1` n'a aucun sens numérique, alors qu'interpoler
    `emb(homme) → emb(femme)` parcourt un segment de l'espace de condition et produit la
    transition continue attendue.

    Le bruit initial est partagé par les `num_frames` images : c'est ce qui préserve
    approximativement l'identité le long de la trajectoire.
    """
    sample_cfg = sample_cfg or generator.cfg.sample
    device = generator.device

    attr_a, null = encode_conditions(generator, attrs_a[0], attrs_a[1], attrs_a[2], 1)
    attr_b, _ = encode_conditions(generator, attrs_b[0], attrs_b[1], attrs_b[2], 1)

    alphas = torch.linspace(0.0, 1.0, num_frames, device=device)[:, None]
    attr_emb = (1.0 - alphas) * attr_a + alphas * attr_b
    null_emb = null.expand(num_frames, -1).contiguous()

    gen = torch_generator(seed, device="cpu")
    shared_noise = torch.randn(
        (1, *generator.latent_shape), generator=gen, dtype=torch.float32
    ).repeat(num_frames, 1, 1, 1)

    latents = sample_latents(
        generator,
        attr_emb,
        null_emb,
        num_steps=sample_cfg.num_steps,
        guidance=sample_cfg.guidance,
        eta=sample_cfg.eta,
        noise=shared_noise,
    )
    images = decode(generator, latents)
    if sample_cfg.restore:
        images = restore_faces(images, sample_cfg)

    alphas_np = alphas.squeeze(1).cpu().numpy()
    return {
        "images": images,
        "latents": latents,
        "alphas": alphas_np,
        # Attributs nominaux le long du chemin. Pour l'âge et l'ITA, l'interpolation du
        # vecteur de condition est *approximativement* équivalente à celle des valeurs
        # (les features de Fourier ne sont pas linéaires) : ces nombres sont un repère
        # d'affichage, pas une prédiction. Seule la mesure de l'oracle fait foi.
        "nominal_age": attrs_a[0] + (attrs_b[0] - attrs_a[0]) * alphas_np,
        "nominal_ita": attrs_a[2] + (attrs_b[2] - attrs_a[2]) * alphas_np,
    }


# --------------------------------------------------------------------------------------
# Restauration 128 → 512 — F-S5
# --------------------------------------------------------------------------------------

_RESTORER_STATE: dict = {"tried": False, "net": None}


def _load_codeformer(device: str):
    if not _RESTORER_STATE["tried"]:
        _RESTORER_STATE["tried"] = True
        try:
            from basicsr.utils.download_util import load_file_from_url
            from codeformer.basicsr.archs.codeformer_arch import CodeFormer

            net = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            ).to(device)
            url = (
                "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
            )
            weights = torch.load(load_file_from_url(url, model_dir="weights"), map_location=device)
            net.load_state_dict(weights["params_ema"])
            net.eval()
            _RESTORER_STATE["net"] = net
        except Exception as exc:
            warnings.warn(
                f"CodeFormer indisponible ({exc.__class__.__name__}). Repli sur un "
                f"agrandissement Lanczos. La restauration est un post-traitement "
                f"cosmétique (F-S5, prio S) : son absence n'affecte aucune métrique, "
                f"qui sont toutes calculées en 128×128 avant restauration.",
                RuntimeWarning,
                stacklevel=2,
            )
            _RESTORER_STATE["net"] = None
    return _RESTORER_STATE["net"]


def restore_faces(
    images_uint8: np.ndarray, sample_cfg: SampleConfig, device: str = "cuda"
) -> np.ndarray:
    """128 → 512 par CodeFormer si disponible, sinon agrandissement Lanczos.

    Point de méthode : la restauration n'est **jamais** appliquée avant l'évaluation.
    Un restaurateur entraîné sur des visages réels rapproche mécaniquement les images
    générées de la distribution réelle et ferait baisser le FID sans que notre modèle
    y soit pour quoi que ce soit.
    """
    net = _load_codeformer(device) if torch.cuda.is_available() else None
    size = sample_cfg.restore_size

    if net is None:
        from PIL import Image

        return np.stack(
            [
                np.asarray(
                    Image.fromarray(img).resize((size, size), Image.LANCZOS), dtype=np.uint8
                )
                for img in images_uint8
            ]
        )

    from facedit.data.fairface import to_signed_float

    outputs = []
    with torch.no_grad():
        for image in images_uint8:
            from PIL import Image

            upscaled = np.asarray(
                Image.fromarray(image).resize((512, 512), Image.BICUBIC), dtype=np.uint8
            )
            tensor = torch.from_numpy(to_signed_float(upscaled))[None].to(device)
            restored = net(tensor, w=sample_cfg.codeformer_fidelity, adain=True)[0]
            outputs.append(to_uint8(restored.clamp(-1, 1))[0])
    return np.stack(outputs)


# --------------------------------------------------------------------------------------
# Grilles et CLI
# --------------------------------------------------------------------------------------


def make_grid(images_uint8: np.ndarray, ncol: int = 8, pad: int = 2) -> np.ndarray:
    """Assemble un lot d'images en une planche unique."""
    count, height, width = images_uint8.shape[:3]
    ncol = min(ncol, count)
    nrow = int(np.ceil(count / ncol))
    canvas = np.full(
        (nrow * (height + pad) + pad, ncol * (width + pad) + pad, 3), 255, dtype=np.uint8
    )
    for index, image in enumerate(images_uint8):
        row, col = divmod(index, ncol)
        y = pad + row * (height + pad)
        x = pad + col * (width + pad)
        canvas[y : y + height, x : x + width] = image
    return canvas


def attribute_sweep_grid(
    generator: Generator, seed: int = 0, sample_cfg: Optional[SampleConfig] = None
) -> np.ndarray:
    """Planche de diagnostic : une ligne par attribut, l'attribut balayé en colonnes.

    Utilisée par F-T6 pendant l'entraînement. Si les trois lignes se ressemblent, le
    conditionnement ne prend pas — c'est le symptôme du risque n°1 du §12, visible sans
    attendre l'évaluation complète.
    """
    sample_cfg = sample_cfg or generator.cfg.sample
    base_age, base_gender, base_ita = 35.0, 1, 30.0
    ncol = 8
    rows: List[np.ndarray] = []

    sweeps = [
        (np.linspace(20.0, 70.0, ncol), np.full(ncol, base_gender), np.full(ncol, base_ita)),
        (
            np.full(ncol, base_age),
            np.array([0] * (ncol // 2) + [1] * (ncol - ncol // 2)),
            np.full(ncol, base_ita),
        ),
        (np.full(ncol, base_age), np.full(ncol, base_gender), np.linspace(-40.0, 60.0, ncol)),
    ]
    for ages, genders, itas in sweeps:
        attr, null = encode_conditions(generator, ages, genders, itas, ncol)
        latents = sample_latents(
            generator,
            attr,
            null,
            num_steps=sample_cfg.num_steps,
            guidance=sample_cfg.guidance,
            eta=sample_cfg.eta,
            seed=seed,
        )
        rows.append(decode(generator, latents))

    return make_grid(np.concatenate(rows, axis=0), ncol=ncol)


def main() -> None:
    import argparse

    from PIL import Image

    parser = argparse.ArgumentParser(description="Échantillonnage FaceDiT (F-S1..F-S6)")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="samples")
    parser.add_argument("--age", type=float, default=35.0)
    parser.add_argument("--gender", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--ita", type=float, default=30.0)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", "-w", type=float, default=None)
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Planche de diagnostic 3 lignes")
    parser.add_argument(
        "--interpolate",
        nargs=3,
        type=float,
        metavar=("AGE_B", "GENDER_B", "ITA_B"),
        default=None,
        help="Second jeu d'attributs : produit une bande d'interpolation (F-S4)",
    )
    args = parser.parse_args()

    generator = load_generator(args.ckpt)
    sample_cfg = generator.cfg.sample
    if args.steps is not None:
        sample_cfg.num_steps = args.steps
    if args.guidance is not None:
        sample_cfg.guidance = args.guidance
    sample_cfg.restore = args.restore

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        grid = attribute_sweep_grid(generator, args.seed, sample_cfg)
        path = out_dir / f"sweep_step{generator.step}.png"
        Image.fromarray(grid).save(path)
        print(f"Planche de diagnostic → {path}")
        return

    if args.interpolate is not None:
        result = interpolate(
            generator,
            (args.age, args.gender, args.ita),
            (args.interpolate[0], int(args.interpolate[1]), args.interpolate[2]),
            num_frames=args.num,
            seed=args.seed,
            sample_cfg=sample_cfg,
        )
        path = out_dir / f"interp_seed{args.seed}.png"
        Image.fromarray(make_grid(result["images"], ncol=args.num)).save(path)
        print(f"Bande d'interpolation ({args.num} images) → {path}")
        return

    result = generate(
        generator,
        args.age,
        args.gender,
        args.ita,
        batch_size=args.num,
        sample_cfg=sample_cfg,
        seed=args.seed,
    )
    images = result["images"]
    path = out_dir / f"grid_age{args.age:.0f}_g{args.gender}_ita{args.ita:.0f}_s{args.seed}.png"
    Image.fromarray(make_grid(images)).save(path)
    print(f"{len(images)} images → {path}")

    # Mesure immédiate de l'ITA obtenu : la boucle fermée du §1.2, même en CLI.
    from facedit.data.ita import compute_ita_batch

    ita_measured, valid = compute_ita_batch(images, use_mediapipe=True)
    if valid.any():
        print(
            f"ITA demandé {args.ita:.1f}° · mesuré {np.nanmean(ita_measured):.1f}° "
            f"· ΔITA {abs(np.nanmean(ita_measured) - args.ita):.1f}°"
        )


if __name__ == "__main__":
    main()
