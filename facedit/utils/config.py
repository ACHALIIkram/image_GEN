"""Configuration typée, chargée depuis un unique fichier YAML.

NF-6 : toute expérience est rejouable depuis un seul fichier de configuration + une seed.
Le `Config` complet (y compris les valeurs par défaut non écrites dans le YAML) est
re-sérialisé dans chaque checkpoint et dans `results.json`, afin qu'une expérience
puisse être reconstruite sans le fichier d'origine.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, get_args, get_origin, get_type_hints

import yaml


# --------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Préparation des données et pré-encodage VAE (module `data`)."""

    # Source brute
    root: str = "data/fairface"
    """Dossier contenant `fairface_label_{train,val}.csv` et les images (padding=0.25)."""
    hf_dataset: Optional[str] = "HuggingFaceM4/FairFace"
    """Repli : identifiant Hugging Face si `root` est absent. None pour désactiver."""
    hf_config: str = "0.25"

    # Filtrage (F-D2)
    age_min: float = 18.0
    age_max: float = 70.0
    drop_partial_age_bins: bool = True
    """Si vrai, une tranche FairFace qui déborde de [age_min, age_max] est rejetée en
    entier plutôt que rognée. La tranche `10-19` contient majoritairement des mineurs
    dont l'âge exact est inconnu : la rogner en [18, 20) fabriquerait du bruit de label.
    Conséquence documentée : la plage effectivement couverte devient [20, 70)."""

    # Géométrie
    image_size: int = 128
    latent_size: int = 16
    hflip: bool = True
    """F-D6 : flip horizontal appliqué AVANT l'encodage VAE, en doublant le cache."""

    # VAE gelé
    vae_id: str = "stabilityai/sd-vae-ft-mse"
    vae_scale: float = 0.18215

    # Artefacts produits
    cache_dir: str = "artifacts"
    latents_file: str = "latents.npy"
    labels_file: str = "labels.npy"
    age_bounds_file: str = "age_bounds.npy"
    meta_file: str = "dataset_meta.json"

    # Encodage
    encode_batch_size: int = 64
    num_workers: int = 4

    # Alignement (F-D10)
    align_faces: bool = False
    """Aligne chaque visage par similitude sur les deux yeux avant l'ITA et le VAE.
    Mesuré : divise par deux l'écart-type de position (7.1 → 3.3 px), d'échelle
    (5.9 → 2.6 px) et de roulis (5.9 → 3.5°). Écarte ~7 % d'images où aucun visage
    exploitable n'est détecté, plutôt que de les laisser non alignées."""

    # ITA (F-D4)
    ita_use_mediapipe: bool = True
    ita_min_pixels: int = 200
    """En dessous de ce nombre de pixels de peau retenus, l'ITA est marqué invalide."""

    @property
    def cache(self) -> Path:
        return Path(self.cache_dir)

    def latents_path(self, split: str = "train") -> Path:
        return self.cache / f"{split}_{self.image_size}_{self.latents_file}"

    def labels_path(self, split: str = "train") -> Path:
        return self.cache / f"{split}_{self.image_size}_{self.labels_file}"

    def age_bounds_path(self, split: str = "train") -> Path:
        return self.cache / f"{split}_{self.image_size}_{self.age_bounds_file}"

    def meta_path(self, split: str = "train") -> Path:
        return self.cache / f"{split}_{self.image_size}_{self.meta_file}"


@dataclass
class ModelConfig:
    """DiT conditionnel. Écrit à la main, aucune classe modèle de bibliothèque (F-M1).

    Les valeurs par défaut ci-dessous sont celles du PROTOTYPE (9,95 M paramètres).
    Le modèle livré les surcharge dans `configs/final_v2.yaml` : patch_size 1,
    depth 12, hidden_size 384, num_heads 6 — soit 32,96 M paramètres.
    """

    latent_size: int = 16
    in_channels: int = 4
    patch_size: int = 2
    depth: int = 8
    hidden_size: int = 256
    num_heads: int = 4
    mlp_ratio: float = 4.0
    fourier_num_freqs: int = 64
    fourier_max_freq: float = 64.0
    dropout: float = 0.0

    # Bornes de normalisation des attributs continus (doivent rester figées entre
    # entraînement et inférence, sinon le conditionnement se décale silencieusement).
    age_norm_max: float = 70.0
    ita_norm_offset: float = 60.0
    ita_norm_scale: float = 120.0

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} n'est pas divisible par "
                f"num_heads={self.num_heads}"
            )
        if self.latent_size % self.patch_size != 0:
            raise ValueError(
                f"latent_size={self.latent_size} n'est pas divisible par "
                f"patch_size={self.patch_size}"
            )

    @property
    def num_patches(self) -> int:
        return (self.latent_size // self.patch_size) ** 2


@dataclass
class TrainConfig:
    """Boucle d'entraînement (§7.3)."""

    steps: int = 60_000
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    ema_decay: float = 0.9999
    ema_start: int = 1000

    cond_dropout: float = 0.1
    independent_cond_dropout: bool = False
    """F-M8 (prio C) : masque de dropout tiré indépendamment par attribut, ce qui rend
    possible le guidage séparé par attribut à l'inférence (F-S6)."""

    num_train_timesteps: int = 1000
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    precision: str = "bf16"  # "bf16" | "fp32"
    fused_adam: bool = True

    ckpt_every: int = 5000
    sample_every: int = 2000
    log_every: int = 100
    keep_last_ckpts: int = 3
    out_dir: str = "runs/final"


@dataclass
class SampleConfig:
    """Échantillonnage (§7.4)."""

    num_steps: int = 25
    eta: float = 0.0
    guidance: float = 3.0
    guidance_age: Optional[float] = None
    guidance_gender: Optional[float] = None
    guidance_ita: Optional[float] = None
    """F-S6 : si l'un des trois est renseigné, le guidage passe en mode par-attribut."""
    restore: bool = False
    restore_size: int = 512
    codeformer_fidelity: float = 0.7


@dataclass
class EvalConfig:
    """Protocole d'évaluation (§8)."""

    fid_num_samples: int = 10_000
    attr_num_samples: int = 2000
    leakage_num_seeds: int = 100
    diversity_num_pairs: int = 500
    subgroup_samples_per_cell: int = 300
    w_sweep: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    memorization_num_samples: int = 500

    age_bins: List[float] = field(default_factory=lambda: [20.0, 33.0, 46.0, 58.0, 70.0])
    ita_bins: List[float] = field(default_factory=lambda: [-40.0, 10.0, 28.0, 45.0, 70.0])

    real_stats_dir: str = "artifacts/fid_ref"
    oracle_ckpt: str = "runs/oracle/oracle_best.pt"
    batch_size: int = 64
    out_dir: str = "results"


@dataclass
class OracleConfig:
    """Oracle d'attributs à 3 têtes (§4.2)."""

    image_size: int = 128
    backbone: str = "resnet18"
    pretrained: bool = True
    epochs: int = 12
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    age_loss_weight: float = 1.0
    ita_loss_weight: float = 1.0
    gender_loss_weight: float = 1.0
    huber_delta: float = 5.0
    num_workers: int = 4
    out_dir: str = "runs/oracle"
    # Normalisation des cibles de régression (l'oracle prédit en unités normalisées).
    age_mean: float = 40.0
    age_std: float = 15.0
    ita_mean: float = 25.0
    ita_std: float = 25.0


@dataclass
class Config:
    name: str = "final"
    seed: int = 1234
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)

    # ---------------------------------------------------------------- (de)sérialisation
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True)

    def hash(self) -> str:
        """Empreinte stable de la configuration, reportée dans `results.json`."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml(), encoding="utf-8")

    def validate(self) -> None:
        """Cohérences inter-sections qu'aucune section ne peut vérifier seule."""
        if self.model.latent_size != self.data.latent_size:
            raise ValueError(
                f"model.latent_size={self.model.latent_size} != "
                f"data.latent_size={self.data.latent_size}"
            )
        expected = self.data.image_size // 8  # facteur de compression du VAE SD
        if self.data.latent_size != expected:
            raise ValueError(
                f"data.latent_size={self.data.latent_size} incohérent avec "
                f"image_size={self.data.image_size} (attendu {expected}, le VAE "
                f"stabilityai/sd-vae-ft-mse compresse d'un facteur 8)"
            )
        if not 0.0 <= self.train.cond_dropout < 1.0:
            raise ValueError(f"cond_dropout hors [0,1) : {self.train.cond_dropout}")
        if self.train.prediction_type != "epsilon":
            raise ValueError(
                "Seule la cible `epsilon` est implémentée (§7.3) ; "
                f"reçu {self.train.prediction_type!r}"
            )


# --------------------------------------------------------------------------------------
# Chargement
# --------------------------------------------------------------------------------------


def _coerce(value: Any, target_type: Any) -> Any:
    """Convertit une valeur YAML vers le type annoté du champ dataclass."""
    origin = get_origin(target_type)

    if origin is None and dataclasses.is_dataclass(target_type):
        return _from_dict(target_type, value or {})

    # Optional[X] / Union[X, None]
    if origin is not None and type(None) in get_args(target_type):
        if value is None:
            return None
        inner = [a for a in get_args(target_type) if a is not type(None)]
        return _coerce(value, inner[0]) if len(inner) == 1 else value

    if origin in (list, List):
        (item_type,) = get_args(target_type) or (Any,)
        return [_coerce(v, item_type) for v in value]

    if origin in (tuple, Tuple):
        args = get_args(target_type)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        return tuple(_coerce(v, t) for v, t in zip(value, args))

    if target_type is float and isinstance(value, int):
        return float(value)

    return value


def _from_dict(cls: Any, payload: Dict[str, Any]) -> Any:
    # `from __future__ import annotations` rend `field.type` textuel : on résout les
    # annotations en vrais objets typing avant toute conversion.
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ValueError(
            f"Clés inconnues dans la section {cls.__name__} : {sorted(unknown)}. "
            f"Clés valides : {sorted(known)}"
        )
    kwargs = {k: _coerce(v, hints[k]) for k, v in payload.items()}
    return cls(**kwargs)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_payload(path: Path, seen: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Charge un YAML en suivant `_base_` de façon RÉCURSIVE.

    La version précédente ne résolvait qu'un seul niveau et supprimait le `_base_` du
    parent, ce qui faisait silencieusement perdre tout ce que le grand-parent déclarait.
    `final_v4 -> final_v3 -> final_v2` retombait ainsi sur les valeurs par défaut de la
    dataclasse : le modèle entraîné était un DiT d8/w256/p2 de 9.95 M paramètres au lieu du
    d12/w384/p1 de 32.96 M attendu. Une nuit d'entraînement sur la mauvaise architecture,
    sans le moindre avertissement — la chaîne d'héritage doit être suivie jusqu'au bout ou
    échouer bruyamment, jamais se taire à mi-parcours.

    Le chemin déjà visité est mémorisé : un `_base_` circulaire lève une erreur explicite
    plutôt que de récurser jusqu'au débordement de pile.
    """
    path = path.resolve()
    seen = list(seen or [])
    if path in seen:
        cycle = " -> ".join(p.name for p in seen + [path])
        raise ValueError(f"`_base_` circulaire : {cycle}")
    seen.append(path)

    payload: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    return _deep_merge(_load_payload(path.parent / base_ref, seen), payload)


def load_config(path: str | Path, overrides: Optional[List[str]] = None) -> Config:
    """Charge un YAML, applique `_base_` puis les surcharges `section.clef=valeur`."""
    payload = _load_payload(Path(path))

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Surcharge mal formée (attendu clef=valeur) : {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = payload
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    cfg = _from_dict(Config, payload)
    cfg.validate()
    return cfg
