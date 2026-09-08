"""Chargement et filtrage de FairFace — F-D1, F-D2, F-D3.

Deux sources sont supportées, dans cet ordre :

1. un dossier local au format officiel (`fairface_label_train.csv` + images) ;
2. le miroir Hugging Face `HuggingFaceM4/FairFace`, config `0.25`.

H1 (§2.3) : les labels de FairFace sont des **perceptions d'annotateurs**, pas des
vérités biologiques. Rien dans ce module ne les traite autrement ; le champ `race` est
conservé pour l'audit de biais (F-E7) mais n'est jamais utilisé comme conditionnement
(E-1 : on conditionne sur l'ITA, grandeur physique, pas sur une catégorie ethnique).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Taxonomie FairFace
# --------------------------------------------------------------------------------------

# Bornes réelles de chaque tranche, en années : [low, high). FairFace annonce des
# tranches inclusives ("60-69") ; la borne haute exclusive vaut donc high+1.
AGE_BIN_BOUNDS: Dict[str, Tuple[float, float]] = {
    "0-2": (0.0, 3.0),
    "3-9": (3.0, 10.0),
    "10-19": (10.0, 20.0),
    "20-29": (20.0, 30.0),
    "30-39": (30.0, 40.0),
    "40-49": (40.0, 50.0),
    "50-59": (50.0, 60.0),
    "60-69": (60.0, 70.0),
    "more than 70": (70.0, 90.0),
}

# Ordre des tranches tel que sérialisé par le miroir Hugging Face (labels entiers).
HF_AGE_ORDER: Tuple[str, ...] = (
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
)

GENDER_TO_INDEX: Dict[str, int] = {"Male": 0, "Female": 1}
INDEX_TO_GENDER: Tuple[str, str] = ("Male", "Female")
HF_GENDER_ORDER: Tuple[str, ...] = ("Male", "Female")

RACE_ORDER: Tuple[str, ...] = (
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
)

GENDER_UNKNOWN = 2
"""Index du token « inconnu » consommé par `nn.Embedding(3, ...)` (F-M5) et utilisé
comme condition nulle du CFG (F-M7)."""


# --------------------------------------------------------------------------------------
# Filtrage d'âge (F-D2) et âge continu (F-D3)
# --------------------------------------------------------------------------------------


def resolve_age_bins(
    age_min: float, age_max: float, drop_partial: bool
) -> Dict[str, Tuple[float, float]]:
    """Tranches retenues, avec leurs bornes éventuellement rognées à [age_min, age_max].

    `drop_partial=True` (défaut) rejette toute tranche qui déborde du périmètre plutôt
    que de la rogner. Motif : rogner `10-19` en `[18, 20)` étiquetterait des enfants de
    11 ans comme des adultes de 18-20 ans, puisque l'âge exact intra-tranche est inconnu.
    Le coût est que la plage réellement couverte devient [20, 70) et non [18, 70) — un
    écart au périmètre §2.1 qui est assumé et documenté plutôt que masqué par du bruit
    de label.
    """
    kept: Dict[str, Tuple[float, float]] = {}
    for name, (low, high) in AGE_BIN_BOUNDS.items():
        if high <= age_min or low >= age_max:
            continue  # entièrement hors périmètre
        inside = low >= age_min and high <= age_max
        if inside:
            kept[name] = (low, high)
        elif not drop_partial:
            kept[name] = (max(low, age_min), min(high, age_max))
    if not kept:
        raise ValueError(
            f"Aucune tranche FairFace ne survit au filtre [{age_min}, {age_max}]"
        )
    return kept


def sample_continuous_age(
    bounds: np.ndarray, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Âge continu ~ Uniform(low, high) par échantillon (F-D3, §7.1).

    Tiré **à chaque chargement** et non figé une fois pour toutes : figer reviendrait à
    n'exposer le modèle qu'à autant de valeurs distinctes qu'il y a d'images, ce qui
    suffit en pratique, mais le tirage par pas coûte zéro et garantit la densité de la
    couverture nécessaire à l'interpolation (F-S4).
    """
    rng = rng or np.random.default_rng()
    low, high = bounds[:, 0], bounds[:, 1]
    return (low + (high - low) * rng.random(len(bounds))).astype(np.float32)


def age_bin_centers(bounds: np.ndarray) -> np.ndarray:
    """Centre de tranche : valeur déterministe stockée dans `labels.npy` (contrat §6.3)."""
    return ((bounds[:, 0] + bounds[:, 1]) / 2.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------


@dataclass
class FairFaceRecord:
    """Une ligne d'annotation, source-agnostique."""

    key: str
    age_bin: str
    gender: int
    race: int
    age_low: float
    age_high: float


class FairFaceSource:
    """Interface commune aux deux backends. `records` est déjà filtré."""

    records: List[FairFaceRecord]
    origin: str

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, index: int) -> np.ndarray:
        raise NotImplementedError


class _LocalFairFace(FairFaceSource):
    """Dossier au format officiel : `fairface_label_<split>.csv` + `<split>/*.jpg`."""

    def __init__(self, root: Path, split: str, kept_bins: Dict[str, Tuple[float, float]]):
        import pandas as pd

        csv_path = root / f"fairface_label_{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        frame = pd.read_csv(csv_path)
        missing = {"file", "age", "gender", "race"} - set(frame.columns)
        if missing:
            raise ValueError(f"Colonnes absentes de {csv_path} : {sorted(missing)}")

        self.root = root
        self.origin = f"local:{root}"
        self.records = []
        for row in frame.itertuples(index=False):
            if row.age not in kept_bins:
                continue
            gender = GENDER_TO_INDEX.get(str(row.gender))
            if gender is None:
                continue
            low, high = kept_bins[row.age]
            race = RACE_ORDER.index(row.race) if row.race in RACE_ORDER else -1
            self.records.append(
                FairFaceRecord(str(row.file), str(row.age), gender, race, low, high)
            )

    def load_image(self, index: int) -> np.ndarray:
        from PIL import Image

        path = self.root / self.records[index].key
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB"), dtype=np.uint8)


class _HFFairFace(FairFaceSource):
    """Miroir Hugging Face. Les images restent paresseuses (Arrow memory-mapped)."""

    def __init__(
        self,
        dataset_id: str,
        config: str,
        split: str,
        kept_bins: Dict[str, Tuple[float, float]],
    ):
        from datasets import load_dataset

        hf_split = {"train": "train", "val": "validation"}.get(split, split)
        self.dataset = load_dataset(dataset_id, config, split=hf_split)
        self.origin = f"hf:{dataset_id}/{config}#{hf_split}"

        ages = np.asarray(self.dataset["age"])
        genders = np.asarray(self.dataset["gender"])
        races = np.asarray(self.dataset["race"])

        self.records = []
        self.row_index: List[int] = []
        for row, (age_idx, gender_idx, race_idx) in enumerate(zip(ages, genders, races)):
            age_bin = HF_AGE_ORDER[int(age_idx)]
            if age_bin not in kept_bins:
                continue
            low, high = kept_bins[age_bin]
            self.records.append(
                FairFaceRecord(
                    f"hf#{row}", age_bin, int(gender_idx), int(race_idx), low, high
                )
            )
            self.row_index.append(row)

    def load_image(self, index: int) -> np.ndarray:
        item = self.dataset[self.row_index[index]]
        return np.asarray(item["image"].convert("RGB"), dtype=np.uint8)


def open_fairface(cfg, split: str) -> FairFaceSource:
    """Ouvre FairFace : dossier local en priorité, miroir HF en repli."""
    kept = resolve_age_bins(cfg.age_min, cfg.age_max, cfg.drop_partial_age_bins)
    root = Path(cfg.root)

    if (root / f"fairface_label_{split}.csv").exists():
        return _LocalFairFace(root, split, kept)

    if cfg.hf_dataset:
        warnings.warn(
            f"{root / f'fairface_label_{split}.csv'} absent : repli sur le miroir "
            f"Hugging Face {cfg.hf_dataset}. Le premier appel télécharge le jeu.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _HFFairFace(cfg.hf_dataset, cfg.hf_config, split, kept)

    raise FileNotFoundError(
        f"FairFace introuvable. Attendu {root}/fairface_label_{split}.csv, ou bien "
        f"data.hf_dataset renseigné dans la configuration."
    )


# --------------------------------------------------------------------------------------
# Dataset Torch (utilisé par l'encodage VAE et par l'oracle)
# --------------------------------------------------------------------------------------


def resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Redimensionne en (size, size, 3) uint8, filtre Lanczos (F-D1).

    Lanczos plutôt que bilinéaire : les crops FairFace font 224 px, la réduction vers
    128 px est un facteur 1.75 et un filtre trop doux effacerait le grain de peau que
    le VAE doit encoder — et donc que l'ITA doit pouvoir mesurer.
    """
    if image.shape[0] == size and image.shape[1] == size:
        return np.ascontiguousarray(image)
    from PIL import Image

    return np.asarray(
        Image.fromarray(image).resize((size, size), Image.LANCZOS), dtype=np.uint8
    )


def to_signed_float(images_uint8: np.ndarray) -> np.ndarray:
    """uint8 [0,255] HWC → float32 [-1,1] CHW (F-D1, normalisation)."""
    array = np.asarray(images_uint8, dtype=np.float32) / 127.5 - 1.0
    if array.ndim == 3:
        return np.ascontiguousarray(array.transpose(2, 0, 1))
    return np.ascontiguousarray(array.transpose(0, 3, 1, 2))


def to_uint8(images_signed) -> np.ndarray:
    """float [-1,1] CHW (torch ou numpy) → uint8 HWC. Inverse exact de `to_signed_float`.

    L'arrondi (`np.rint`) est indispensable : `astype(np.uint8)` **tronque**. Comme
    `(v + 1) · 127.5` retombe en général un epsilon sous l'entier visé, une simple
    troncature retirerait un niveau à une grande partie des pixels — un assombrissement
    systématique d'un demi-LSB sur toute image générée. L'effet est invisible à l'œil
    mais pas pour l'ITA : il abaisse L* et donc l'ITA mesuré, en biaisant ΔITA (F-E3)
    dans le sens qui nous arrangerait sur les teints clairs.
    """
    array = images_signed
    if hasattr(array, "detach"):
        array = array.detach().float().cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(np.rint((array + 1.0) * 127.5), 0.0, 255.0).astype(np.uint8)
    if array.ndim == 3:
        return np.ascontiguousarray(array.transpose(1, 2, 0))
    return np.ascontiguousarray(array.transpose(0, 2, 3, 1))


class FairFaceImages:
    """`torch.utils.data.Dataset` renvoyant des images uint8 déjà redimensionnées.

    Volontairement pauvre : pas de tenseur, pas de normalisation, pas d'augmentation.
    L'ITA se calcule sur du uint8, le VAE veut du float normalisé, l'oracle veut ses
    propres augmentations — chacun applique sa transformation en aval.
    """

    def __init__(
        self,
        source: FairFaceSource,
        image_size: int,
        flip: bool = False,
        align: bool = False,
    ):
        self.source = source
        self.image_size = image_size
        self.flip = flip
        # F-D10. Quand l'alignement est actif, `image` peut valoir None : aucun visage
        # exploitable n'a été trouvé. L'appelant DOIT traiter ce cas plutôt que de
        # supposer une image ; c'est pour ça qu'on renvoie None au lieu de retomber
        # silencieusement sur un simple redimensionnement, qui réintroduirait dans le jeu
        # la variance géométrique qu'on cherche précisément à supprimer.
        self.align = align

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> Dict[str, object]:
        original = self.source.load_image(index)
        if self.align:
            from facedit.data.align import align_face

            # Aligner depuis l'image d'origine : un seul rééchantillonnage au lieu de deux.
            image = align_face(original, self.image_size)
        else:
            image = resize_uint8(original, self.image_size)
        if image is None:
            record = self.source.records[index]
            return {
                "image": None, "index": index, "gender": record.gender,
                "race": record.race, "age_low": record.age_low, "age_high": record.age_high,
            }
        if self.flip:
            image = np.ascontiguousarray(image[:, ::-1, :])
        record = self.source.records[index]
        return {
            "image": image,
            "index": index,
            "gender": record.gender,
            "race": record.race,
            "age_low": record.age_low,
            "age_high": record.age_high,
        }


def collate_uint8(batch: Sequence[Dict[str, object]]) -> Dict[str, np.ndarray]:
    """Collate NumPy : évite une conversion Torch inutile côté workers.

    Les entrées dont `image` vaut None (alignement impossible, F-D10) sont retirées ici.
    Le lot peut donc être plus petit que `batch_size`, et vide si aucune image n'a pu être
    alignée — d'où le champ `n_dropped`, que l'appelant doit comptabiliser pour que le
    rapport de jeu de données dise combien d'images ont disparu et pourquoi.
    """
    dropped = sum(1 for b in batch if b["image"] is None)
    batch = [b for b in batch if b["image"] is not None]
    if not batch:
        return {"image": np.zeros((0, 1, 1, 3), np.uint8), "n_dropped": dropped,
                "index": np.zeros(0, np.int64), "gender": np.zeros(0, np.int64),
                "race": np.zeros(0, np.int64), "age_low": np.zeros(0, np.float32),
                "age_high": np.zeros(0, np.float32)}
    return {
        "n_dropped": dropped,
        "image": np.stack([b["image"] for b in batch]),
        "index": np.asarray([b["index"] for b in batch], dtype=np.int64),
        "gender": np.asarray([b["gender"] for b in batch], dtype=np.int64),
        "race": np.asarray([b["race"] for b in batch], dtype=np.int64),
        "age_low": np.asarray([b["age_low"] for b in batch], dtype=np.float32),
        "age_high": np.asarray([b["age_high"] for b in batch], dtype=np.float32),
    }
