"""Calcul de l'Individual Typology Angle (ITA) — F-D4, §7.1.

Pipeline, dans l'ordre imposé par le PRD :

1. Masque de peau : seuillage YCbCr (77 ≤ Cb ≤ 127, 133 ≤ Cr ≤ 173), intersecté avec
   une région joues/front. La région est obtenue par les repères MediaPipe si la
   bibliothèque est disponible, sinon par un gabarit géométrique (les crops FairFace
   sont alignés, la position des joues et du front y est stable).
2. Rejet des pixels dont la luminance est dans le décile inférieur ou supérieur
   (ombres portées et reflets spéculaires).
3. Conversion sRGB → CIELAB, illuminant D65, observateur 2°.
4. ITA = arctan((L* − 50) / b*) × 180/π, **valeur médiane** sur les pixels retenus.
   La médiane, et non la moyenne : la distribution des pixels de peau est asymétrique
   et le masque laisse toujours passer quelques pixels non cutanés.

Limite à documenter (§7.1) : l'ITA est sensible à l'éclairage de la photo originale.
`ita_dispersion_report` mesure cette dispersion — elle constitue le plancher d'erreur
de la métrique ΔITA (F-E3), et donc la barre d'erreur de tout le protocole.

Ce module est volontairement pur NumPy : il est réutilisé par `eval/` sur des images
générées, sans dépendance à Torch ni au GPU.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Constantes colorimétriques
# --------------------------------------------------------------------------------------

# sRGB linéaire → XYZ, illuminant D65 (IEC 61966-2-1).
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# Point blanc D65, observateur 2°.
_WHITE_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

_DELTA = 6.0 / 29.0

# Seuils de peau en YCbCr (§7.1). Bornes classiques de Chai & Ngan.
_CB_MIN, _CB_MAX = 77.0, 127.0
_CR_MIN, _CR_MAX = 133.0, 173.0

# b* strictement positif pour la peau ; borne basse pour éviter une division explosive
# sur les pixels quasi neutres qui auraient franchi le masque.
_B_STAR_FLOOR = 1e-3

MIN_B_STAR = 5.0
"""Seuil de validité sur b* médian, en unités CIELAB.

La peau réelle a une composante jaune franche : b* y vaut typiquement 12 à 25. Un b*
proche de zéro signifie que la région mesurée n'est pas de la peau colorée — en
pratique, une **photographie en noir et blanc**, dont FairFace contient une proportion
non négligeable.

Pourquoi cela ne peut pas être ignoré : ITA = arctan((L* − 50) / b*). Quand b* → 0,
l'arc-tangente sature vers ±90° selon le seul signe de (L* − 50). Une photo N&B claire
produit donc +90° et une photo N&B sombre −90°, deux valeurs qui n'ont aucun rapport
avec le teint du sujet. Mesuré sur FairFace, cela concerne ~6 % des images, et ces
valeurs aberrantes se logent exactement aux extrêmes de la plage d'ITA — là où le
contrôle du teint est le plus difficile et où l'évaluation le sonde.

Ces mesures sont donc déclarées invalides et écartées à la construction du cache, plutôt
que bornées : borner accumulerait une masse artificielle aux extrémités et apprendrait
au modèle que « ITA = 90° » est un teint fréquent.
"""


# --------------------------------------------------------------------------------------
# Échelle de Monk (F-I6, §7.1 étape 5)
# --------------------------------------------------------------------------------------

# Table de correspondance Monk ↔ ITA utilisée par tout le système (interface et rapport).
#
# Construction : les six catégories ITA de Chardon/Del Bino (very light > 55, light
# 41..55, intermediate 28..41, tan 10..28, brown −30..10, dark < −30) couvrent la plage
# utile ≈ [−55, +70]. L'échelle de Monk a dix échelons perceptuellement réguliers ; on
# place donc dix centres régulièrement espacés sur cette plage, puis on assigne à chaque
# échelon la frontière médiane avec ses voisins.
#
# Cette table est une convention de projet, pas un résultat publié : elle est reportée
# telle quelle dans le rapport et toute comparaison externe doit la citer.
MONK_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
MONK_ITA_CENTERS: Tuple[float, ...] = (
    66.0,  # A — le plus clair
    58.0,  # B
    49.0,  # C
    40.0,  # D
    31.0,  # E
    21.0,  # F
    10.0,  # G
    -6.0,  # H
    -25.0,  # I
    -45.0,  # J — le plus foncé
)

# Catégories ITA dermatologiques, conservées pour le rapport (§7.1).
ITA_CATEGORIES: Tuple[Tuple[str, float, float], ...] = (
    ("dark", -np.inf, -30.0),
    ("brown", -30.0, 10.0),
    ("tan", 10.0, 28.0),
    ("intermediate", 28.0, 41.0),
    ("light", 41.0, 55.0),
    ("very light", 55.0, np.inf),
)


def monk_to_ita(label: str) -> float:
    """Échelon Monk (`"A"`..`"J"`, ou `1`..`10`) → valeur ITA centrale, en degrés."""
    key = str(label).strip().upper()
    if key.isdigit():
        index = int(key) - 1
        if not 0 <= index < len(MONK_LABELS):
            raise ValueError(f"Échelon Monk numérique hors [1,10] : {label!r}")
        return MONK_ITA_CENTERS[index]
    if key not in MONK_LABELS:
        raise ValueError(f"Échelon Monk inconnu : {label!r} (attendu A..J ou 1..10)")
    return MONK_ITA_CENTERS[MONK_LABELS.index(key)]


def ita_to_monk(ita: float | np.ndarray) -> np.ndarray | str:
    """ITA (degrés) → échelon Monk le plus proche. Vectorisé."""
    centers = np.asarray(MONK_ITA_CENTERS)
    values = np.atleast_1d(np.asarray(ita, dtype=np.float64))
    index = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
    labels = np.array(MONK_LABELS)[index]
    return labels[0] if np.isscalar(ita) or np.ndim(ita) == 0 else labels


def ita_to_category(ita: float | np.ndarray) -> np.ndarray | str:
    """ITA → catégorie dermatologique (`dark` … `very light`)."""
    values = np.atleast_1d(np.asarray(ita, dtype=np.float64))
    out = np.empty(values.shape, dtype=object)
    for name, low, high in ITA_CATEGORIES:
        out[(values > low) & (values <= high)] = name
    return out[0] if np.ndim(ita) == 0 else out


# --------------------------------------------------------------------------------------
# Conversions colorimétriques
# --------------------------------------------------------------------------------------


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Dé-gamma sRGB. `rgb` dans [0, 1]."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def srgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) → CIELAB (..., 3), illuminant D65.

    Implémenté à la main plutôt que via `cv2.cvtColor` : OpenCV utilise l'illuminant D65
    mais quantifie L* sur 8 bits en mode uint8, ce qui introduit ~0.4° d'erreur sur
    l'ITA — du même ordre que l'effet qu'on cherche à mesurer.
    """
    rgb = np.asarray(rgb_uint8, dtype=np.float64) / 255.0
    linear = srgb_to_linear(rgb)

    xyz = linear @ _RGB_TO_XYZ.T
    xyz = xyz / _WHITE_D65

    # f(t) de la CIE, avec sa partie linéaire près de 0.
    cube = np.cbrt(xyz)
    linear_part = xyz / (3.0 * _DELTA**2) + 4.0 / 29.0
    f = np.where(xyz > _DELTA**3, cube, linear_part)

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1
    )
    return lab


def rgb_to_ycbcr(rgb_uint8: np.ndarray) -> np.ndarray:
    """sRGB uint8 → YCbCr plage complète (convention JPEG / ITU-R BT.601)."""
    rgb = np.asarray(rgb_uint8, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return np.stack([y, cb, cr], axis=-1)


def lab_to_ita(lab: np.ndarray) -> np.ndarray:
    """CIELAB (..., 3) → ITA en degrés, pixel par pixel."""
    l_star = lab[..., 0]
    b_star = np.maximum(lab[..., 2], _B_STAR_FLOOR)
    return np.arctan((l_star - 50.0) / b_star) * 180.0 / np.pi


# --------------------------------------------------------------------------------------
# Régions faciales
# --------------------------------------------------------------------------------------

# Repères MediaPipe FaceMesh délimitant des zones peu ombrées et sans pilosité.
_MP_LEFT_CHEEK = (50, 101, 118, 117, 123, 116, 111, 205, 36, 142)
_MP_RIGHT_CHEEK = (280, 330, 347, 346, 352, 345, 340, 425, 266, 371)
_MP_FOREHEAD = (67, 69, 104, 108, 151, 337, 299, 333, 297, 109, 10)

_MEDIAPIPE_STATE: dict = {"tried": False, "mesh": None}


def _get_face_mesh():
    """Instancie FaceMesh une seule fois. Renvoie None si MediaPipe est indisponible."""
    if not _MEDIAPIPE_STATE["tried"]:
        _MEDIAPIPE_STATE["tried"] = True
        try:
            import mediapipe as mp

            _MEDIAPIPE_STATE["mesh"] = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.3,
            )
        except Exception as exc:  # ImportError, mais aussi échecs d'init natifs
            warnings.warn(
                f"MediaPipe indisponible ({exc.__class__.__name__}: {exc}). "
                "Repli sur le gabarit géométrique pour la région joues/front. "
                "Cette substitution doit être mentionnée dans le rapport.",
                RuntimeWarning,
                stacklevel=2,
            )
            _MEDIAPIPE_STATE["mesh"] = None
    return _MEDIAPIPE_STATE["mesh"]


def _polygon_mask(shape: Tuple[int, int], points: np.ndarray) -> np.ndarray:
    """Rasterise l'enveloppe convexe d'un nuage de points. Sans OpenCV si nécessaire."""
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    if len(points) < 3:
        return mask
    try:
        import cv2

        hull = cv2.convexHull(points.astype(np.int32))
        filled = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(filled, hull, 1)
        return filled.astype(bool)
    except ImportError:
        # Repli : boîte englobante du nuage, plus grossière mais jamais vide.
        x0, y0 = np.clip(points.min(axis=0), [0, 0], [width - 1, height - 1])
        x1, y1 = np.clip(points.max(axis=0), [0, 0], [width - 1, height - 1])
        mask[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1] = True
        return mask


def _geometric_region(shape: Tuple[int, int]) -> np.ndarray:
    """Gabarit joues + front pour un crop de visage aligné (FairFace padding=0.25).

    Trois ellipses en coordonnées relatives : joue gauche, joue droite, front. Les
    centres sont volontairement conservateurs (à l'écart des yeux, des sourcils, des
    narines et de la ligne de cheveux) parce qu'un pixel non cutané biaise l'ITA bien
    plus qu'une région trop petite ne le bruite.
    """
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)

    mask = np.zeros((height, width), dtype=bool)
    ellipses = (
        # (cx, cy, rx, ry)
        (0.295, 0.605, 0.085, 0.075),  # joue gauche
        (0.705, 0.605, 0.085, 0.075),  # joue droite
        (0.500, 0.285, 0.115, 0.055),  # front
    )
    for cx, cy, rx, ry in ellipses:
        mask |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask


def _mediapipe_region(image_uint8: np.ndarray) -> Optional[np.ndarray]:
    """Région joues/front par repères MediaPipe. None si aucun visage détecté."""
    mesh = _get_face_mesh()
    if mesh is None:
        return None

    height, width = image_uint8.shape[:2]
    # MediaPipe se dégrade fortement en dessous de ~64 px ; on lui donne une version
    # agrandie et on ramène les repères (normalisés) à la résolution d'origine.
    probe = image_uint8
    if min(height, width) < 192:
        try:
            import cv2

            probe = cv2.resize(
                image_uint8, (256, 256), interpolation=cv2.INTER_CUBIC
            )
        except ImportError:
            pass

    try:
        result = mesh.process(np.ascontiguousarray(probe))
    except Exception:
        return None
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    mask = np.zeros((height, width), dtype=bool)
    for group in (_MP_LEFT_CHEEK, _MP_RIGHT_CHEEK, _MP_FOREHEAD):
        pts = np.array(
            [
                [landmarks[i].x * (width - 1), landmarks[i].y * (height - 1)]
                for i in group
                if i < len(landmarks)
            ]
        )
        if len(pts) >= 3:
            mask |= _polygon_mask((height, width), pts)
    return mask if mask.any() else None


# --------------------------------------------------------------------------------------
# Mesure
# --------------------------------------------------------------------------------------


@dataclass
class ITAResult:
    """Résultat d'une mesure d'ITA, avec de quoi juger sa fiabilité."""

    ita: float
    """ITA médian en degrés. `nan` si la mesure a échoué."""
    valid: bool
    n_pixels: int
    """Nombre de pixels de peau retenus après masque et rejet des déciles."""
    ita_iqr: float
    """Écart interquartile de l'ITA sur les pixels retenus : dispersion intra-image."""
    l_star: float
    b_star: float
    region_source: str
    """`mediapipe` | `geometric` | `ycbcr-only` — à agréger pour le rapport."""

    def as_dict(self) -> dict:
        return {
            "ita": float(self.ita),
            "valid": bool(self.valid),
            "n_pixels": int(self.n_pixels),
            "ita_iqr": float(self.ita_iqr),
            "l_star": float(self.l_star),
            "b_star": float(self.b_star),
            "region_source": self.region_source,
        }


_FAILED = ITAResult(np.nan, False, 0, np.nan, np.nan, np.nan, "none")


MEASUREMENT_SIZE = 128
"""Résolution de référence à laquelle tout ITA est mesuré.

L'ITA est une statistique de couleur : il caractérise le visage, pas la résolution à
laquelle on l'observe. Le mesurer directement sur une image 32×32 (palier 0) ne donnerait
qu'une soixantaine de pixels de peau — sous le seuil de fiabilité, et surtout une
définition de la métrique qui changerait d'un palier à l'autre, rendant les ΔITA des
paliers 0, 1 et 2 incomparables.

Les images plus petites sont donc agrandies à cette résolution avant mesure. L'agrandi-
ssement n'ajoute aucune information — il stabilise l'estimateur en donnant au masque
assez de pixels, et il fige la définition de la métrique. Les images déjà à 128 ou
au-delà ne sont jamais touchées.
"""


def _ensure_measurement_size(image: np.ndarray) -> np.ndarray:
    if min(image.shape[:2]) >= MEASUREMENT_SIZE:
        return image
    from PIL import Image

    # Bilinéaire et non Lanczos : Lanczos produit un dépassement (ringing) aux
    # transitions, qui fabriquerait des pixels hors gamut aux bords du masque de peau.
    return np.asarray(
        Image.fromarray(image).resize(
            (MEASUREMENT_SIZE, MEASUREMENT_SIZE), Image.BILINEAR
        ),
        dtype=np.uint8,
    )


def compute_ita(
    image_uint8: np.ndarray,
    use_mediapipe: bool = True,
    min_pixels: int = 200,
    decile_reject: bool = True,
    min_b_star: float = MIN_B_STAR,
) -> ITAResult:
    """ITA d'une image RGB uint8 (H, W, 3). Voir le docstring du module pour le protocole."""
    image = np.asarray(image_uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Attendu une image RGB (H, W, 3), reçu {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Attendu du uint8, reçu {image.dtype}")

    image = _ensure_measurement_size(image)
    height, width = image.shape[:2]

    # --- étape 1a : seuillage YCbCr -----------------------------------------------
    ycbcr = rgb_to_ycbcr(image)
    skin = (
        (ycbcr[..., 1] >= _CB_MIN)
        & (ycbcr[..., 1] <= _CB_MAX)
        & (ycbcr[..., 2] >= _CR_MIN)
        & (ycbcr[..., 2] <= _CR_MAX)
    )

    # --- étape 1b : intersection avec la région joues/front -------------------------
    region = _mediapipe_region(image) if use_mediapipe else None
    source = "mediapipe"
    if region is None:
        region = _geometric_region((height, width))
        source = "geometric"

    mask = skin & region
    if mask.sum() < min_pixels:
        # La région est fiable mais le seuillage YCbCr peut échouer sur les teints très
        # foncés ou très clairs (les bornes de Chai & Ngan sont calibrées sur des teints
        # médians). Dans ce cas la région géométrique prime : elle est, par construction,
        # de la peau. C'est le compromis qui évite un biais systématique d'exclusion des
        # extrêmes de la distribution de teint — exactement les cellules que F-E7 audite.
        mask = region
        source = f"{source}+unthresholded"
    if mask.sum() < min_pixels:
        return _FAILED

    # --- étape 2 : rejet des déciles de luminance -----------------------------------
    lab = srgb_to_lab(image)
    l_values = lab[..., 0][mask]
    if decile_reject and l_values.size >= 20:
        low, high = np.percentile(l_values, [10.0, 90.0])
        keep = (l_values >= low) & (l_values <= high)
        if keep.sum() < min_pixels // 2:
            keep = np.ones_like(l_values, dtype=bool)
    else:
        keep = np.ones_like(l_values, dtype=bool)

    # --- étapes 3 & 4 : CIELAB → ITA médian -----------------------------------------
    lab_kept = lab[mask][keep]
    if lab_kept.shape[0] == 0:
        return _FAILED

    ita_pixels = lab_to_ita(lab_kept)
    q25, q75 = np.percentile(ita_pixels, [25.0, 75.0])
    b_star = float(np.median(lab_kept[:, 2]))

    # Garde-fou achromatique : voir `MIN_B_STAR`. La mesure est renvoyée pour le
    # diagnostic, mais marquée invalide pour qu'elle n'entre ni dans le cache
    # d'entraînement ni dans le calcul de ΔITA.
    achromatic = b_star < min_b_star

    return ITAResult(
        ita=float(np.median(ita_pixels)),
        valid=not achromatic,
        n_pixels=int(lab_kept.shape[0]),
        ita_iqr=float(q75 - q25),
        l_star=float(np.median(lab_kept[:, 0])),
        b_star=b_star,
        region_source=f"{source}+achromatic" if achromatic else source,
    )


def compute_ita_batch(
    images_uint8: np.ndarray,
    use_mediapipe: bool = True,
    min_pixels: int = 200,
    progress: bool = False,
    min_b_star: float = MIN_B_STAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """ITA d'un lot (B, H, W, 3) uint8.

    Renvoie `(ita, valid)` de forme (B,). Les entrées invalides portent `nan`, à
    l'appelant de décider s'il les écarte (préparation du dataset) ou les compte comme
    des échecs (évaluation).
    """
    images = np.asarray(images_uint8)
    if images.ndim != 4:
        raise ValueError(f"Attendu (B, H, W, 3), reçu {images.shape}")

    iterator: Sequence[int] = range(images.shape[0])
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="ITA", unit="img")
        except ImportError:
            pass

    ita = np.full(images.shape[0], np.nan, dtype=np.float32)
    valid = np.zeros(images.shape[0], dtype=bool)
    for i in iterator:
        result = compute_ita(
            images[i],
            use_mediapipe=use_mediapipe,
            min_pixels=min_pixels,
            min_b_star=min_b_star,
        )
        # Une mesure invalide ne doit pas circuler comme si elle était bonne : on
        # renvoie `nan`, charge à l'appelant de la compter comme échec (évaluation) ou
        # de l'écarter (construction du cache).
        ita[i] = result.ita if result.valid else np.nan
        valid[i] = result.valid
    return ita, valid


# --------------------------------------------------------------------------------------
# Dispersion — le plancher d'erreur de ΔITA (§7.1, risque « ITA trop bruité »)
# --------------------------------------------------------------------------------------


def ita_dispersion_report(
    images_uint8: np.ndarray,
    use_mediapipe: bool = True,
    brightness_deltas: Sequence[float] = (0.85, 0.925, 1.0, 1.075, 1.15),
) -> dict:
    """Quantifie la sensibilité de l'ITA à l'éclairage.

    On ne dispose pas de plusieurs photos du même individu dans FairFace ; on simule
    donc la variation d'éclairage par un gain multiplicatif appliqué en linéaire (et non
    sur les valeurs sRGB, ce qui reviendrait à modifier le gamma). L'écart-type de l'ITA
    d'une même image sous différents gains est une borne **inférieure** de la dispersion
    réelle intra-individu : il ignore les changements de température de couleur et de
    direction d'éclairage.

    Le chiffre à reporter dans le rapport est `mean_within_image_std` : en dessous de
    cette valeur, un ΔITA mesuré n'est pas distinguable du bruit d'éclairage.
    """
    images = np.asarray(images_uint8)
    per_image_std: List[float] = []
    per_image_iqr: List[float] = []
    base_values: List[float] = []

    for i in range(images.shape[0]):
        measurements = []
        for gain in brightness_deltas:
            linear = srgb_to_linear(images[i].astype(np.float64) / 255.0) * gain
            # Ré-encodage sRGB du signal linéaire éclairci/assombri.
            resrgb = np.where(
                linear <= 0.0031308,
                linear * 12.92,
                1.055 * np.clip(linear, 0, None) ** (1 / 2.4) - 0.055,
            )
            variant = np.clip(resrgb * 255.0, 0, 255).astype(np.uint8)
            result = compute_ita(variant, use_mediapipe=use_mediapipe)
            if result.valid:
                measurements.append(result.ita)
                if gain == 1.0:
                    base_values.append(result.ita)
                    per_image_iqr.append(result.ita_iqr)
        if len(measurements) >= 2:
            per_image_std.append(float(np.std(measurements)))

    return {
        "n_images": int(images.shape[0]),
        "n_measured": len(per_image_std),
        "brightness_gains": list(brightness_deltas),
        "mean_within_image_std": float(np.mean(per_image_std)) if per_image_std else float("nan"),
        "p95_within_image_std": (
            float(np.percentile(per_image_std, 95)) if per_image_std else float("nan")
        ),
        "mean_within_image_iqr": (
            float(np.mean(per_image_iqr)) if per_image_iqr else float("nan")
        ),
        "between_image_std": float(np.std(base_values)) if base_values else float("nan"),
        "note": (
            "mean_within_image_std est le plancher d'erreur de ΔITA (F-E3). Le rapport "
            "signal/bruit utile est between_image_std / mean_within_image_std ; en "
            "dessous de ~3, conditionner l'ITA en continu n'a pas de sens et il faut "
            "basculer sur des bins discrets (question ouverte n°1)."
        ),
    }
