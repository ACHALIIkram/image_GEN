"""Tests du calcul d'ITA — la brique de mesure du projet (F-D4, F-E3).

L'ITA est le point d'ancrage indépendant du protocole : c'est la seule métrique
d'attribut qui ne passe par aucun réseau entraîné. Si elle est fausse, la contribution
différenciante du §1.2 s'effondre sans que rien d'autre ne le signale. D'où des tests
sur des valeurs colorimétriques de référence, et pas seulement sur la cohérence interne.
"""

from __future__ import annotations

import numpy as np
import pytest

from facedit.data.ita import (
    MONK_ITA_CENTERS,
    MONK_LABELS,
    compute_ita,
    ita_to_category,
    ita_to_monk,
    lab_to_ita,
    monk_to_ita,
    rgb_to_ycbcr,
    srgb_to_lab,
    srgb_to_linear,
)


# --------------------------------------------------------------------------------------
# Colorimétrie — valeurs de référence
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rgb,expected_lab",
    [
        ((255, 255, 255), (100.0, 0.0, 0.0)),
        ((0, 0, 0), (0.0, 0.0, 0.0)),
        ((128, 128, 128), (53.585, 0.0, 0.0)),
        ((255, 0, 0), (53.241, 80.092, 67.203)),
        ((0, 255, 0), (87.735, -86.183, 83.179)),
        ((0, 0, 255), (32.297, 79.188, -107.860)),
    ],
)
def test_conversion_srgb_vers_cielab_sur_valeurs_de_reference(rgb, expected_lab):
    """Valeurs CIELAB D65 / observateur 2° publiées, tolérance 0.05."""
    pixel = np.array([[list(rgb)]], dtype=np.uint8)
    lab = srgb_to_lab(pixel)[0, 0]
    assert lab == pytest.approx(np.array(expected_lab), abs=0.05)


def test_la_delinearisation_srgb_respecte_ses_deux_regimes():
    """Le seuil à 0.04045 sépare la partie linéaire de la partie en puissance 2.4."""
    assert srgb_to_linear(np.array(0.04)) == pytest.approx(0.04 / 12.92, rel=1e-9)
    assert srgb_to_linear(np.array(0.5)) == pytest.approx(((0.5 + 0.055) / 1.055) ** 2.4, rel=1e-9)
    assert srgb_to_linear(np.array(1.0)) == pytest.approx(1.0, rel=1e-9)


def test_conversion_ycbcr_place_le_gris_au_centre_des_chromas():
    ycbcr = rgb_to_ycbcr(np.array([[[128, 128, 128]]], dtype=np.uint8))[0, 0]
    assert ycbcr[0] == pytest.approx(128.0, abs=0.5)
    assert ycbcr[1] == pytest.approx(128.0, abs=0.5)
    assert ycbcr[2] == pytest.approx(128.0, abs=0.5)


def test_la_formule_ita_suit_sa_definition():
    """ITA = arctan((L* − 50) / b*) × 180/π."""
    lab = np.array([[70.0, 10.0, 20.0]])
    assert lab_to_ita(lab)[0] == pytest.approx(np.degrees(np.arctan((70.0 - 50.0) / 20.0)), abs=1e-6)


# --------------------------------------------------------------------------------------
# Mesure sur images
# --------------------------------------------------------------------------------------


def _skin_patch(rgb, size: int = 128) -> np.ndarray:
    """Image unie dans le gamut de peau, avec un léger grain pour éviter un cas dégénéré."""
    image = np.zeros((size, size, 3), dtype=np.float64)
    image[:] = rgb
    rng = np.random.default_rng(0)
    image += rng.normal(0.0, 2.0, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


# Du plus clair au plus foncé, tous dans le gamut de peau.
SKIN_TONES = [
    (246, 224, 208),
    (232, 198, 172),
    (208, 166, 134),
    (172, 128, 96),
    (128, 92, 68),
    (86, 60, 44),
]


def test_l_ita_decroit_quand_la_peau_fonce():
    """Propriété fondatrice : l'ITA doit ordonner les teints, sinon il ne mesure rien."""
    values = [compute_ita(_skin_patch(tone), use_mediapipe=False).ita for tone in SKIN_TONES]
    assert all(np.isfinite(values)), values
    assert values == sorted(values, reverse=True), values


def test_l_ita_est_invariant_par_flip_horizontal():
    """Justifie la réutilisation de l'ITA de l'original pour son miroir dans `encode.py`
    (F-D6) : un flip ne change aucune valeur de pixel, seulement leur position."""
    image = _skin_patch((208, 166, 134))
    image[:, :64] = np.clip(image[:, :64].astype(int) + 6, 0, 255).astype(np.uint8)

    direct = compute_ita(image, use_mediapipe=False)
    mirrored = compute_ita(np.ascontiguousarray(image[:, ::-1, :]), use_mediapipe=False)
    assert direct.ita == pytest.approx(mirrored.ita, abs=0.6)


def _shaded_patch(rgb, size: int = 128, shading: float = 0.30) -> np.ndarray:
    """Peau avec un dégradé d'éclairage latéral — le cas réaliste, pas une teinte unie."""
    _, xx = np.mgrid[0:size, 0:size]
    gradient = 1.0 - shading * (xx / size)
    image = np.asarray(rgb, dtype=np.float64)[None, None, :] * gradient[:, :, None]
    rng = np.random.default_rng(0)
    return np.clip(image + rng.normal(0.0, 3.0, image.shape), 0, 255).astype(np.uint8)


def test_le_rejet_des_deciles_ecarte_les_extremes_de_luminance():
    """Étape 2 du protocole §7.1 : le rejet retire un décile de chaque côté.

    Ce que le test vérifie est la **mécanique** : environ 80 % des pixels survivent, et
    la dispersion du pool retenu diminue. Ce qu'il ne vérifie pas — délibérément — est un
    gain de justesse sur la valeur centrale : la médiane de l'étape 4 est déjà robuste
    aux valeurs aberrantes, et sur nos images synthétiques le rejet ne déplace pas
    l'estimation de plus de 0.2°. Les deux étapes se recouvrent donc partiellement ; le
    rejet reste utile parce qu'il resserre le pool sur lequel `ita_iqr` — l'indicateur de
    fiabilité remonté au rapport — est calculé.
    """
    image = _shaded_patch((208, 166, 134))
    kept = compute_ita(image, use_mediapipe=False, decile_reject=True)
    everything = compute_ita(image, use_mediapipe=False, decile_reject=False)

    assert kept.n_pixels / everything.n_pixels == pytest.approx(0.8, abs=0.03)
    assert kept.ita_iqr < everything.ita_iqr


def test_la_mediane_resiste_a_un_reflet_speculaire():
    """Étape 4 : c'est la médiane, et non la moyenne, qui protège la mesure.

    Un reflet couvrant 10 % de la région de peau ne doit pas déplacer l'ITA de plus d'un
    degré — très en dessous du plancher de bruit d'éclairage mesuré par
    `ita_dispersion_report`.
    """
    from facedit.data.ita import _geometric_region

    base = _shaded_patch((208, 166, 134))
    region = _geometric_region(base.shape[:2])
    coordinates = np.argwhere(region)
    rng = np.random.default_rng(1)
    selected = coordinates[rng.choice(len(coordinates), len(coordinates) // 10, replace=False)]

    polluted = base.copy()
    polluted[selected[:, 0], selected[:, 1]] = 250  # reflet spéculaire

    clean = compute_ita(base, use_mediapipe=False).ita
    measured = compute_ita(polluted, use_mediapipe=False).ita
    assert abs(measured - clean) < 1.0


def test_une_image_sans_peau_est_declaree_invalide():
    """Un vert saturé ne franchit pas le seuillage YCbCr *ni* le repli non seuillé ne
    doit produire une mesure présentée comme fiable."""
    green = np.zeros((128, 128, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    result = compute_ita(green, use_mediapipe=False)
    # Le repli géométrique mesure quand même quelque chose, mais l'origine de la région
    # doit signaler que le seuillage de peau a échoué.
    assert "unthresholded" in result.region_source or not result.valid


def test_le_resultat_porte_de_quoi_juger_sa_fiabilite():
    result = compute_ita(_skin_patch((208, 166, 134)), use_mediapipe=False)
    assert result.valid and result.n_pixels > 200
    assert np.isfinite(result.ita_iqr) and result.ita_iqr >= 0
    assert result.region_source.startswith("geometric")


def test_les_entrees_mal_typees_sont_rejetees():
    with pytest.raises(ValueError):
        compute_ita(np.zeros((128, 128), dtype=np.uint8))
    with pytest.raises(ValueError):
        compute_ita(np.zeros((128, 128, 3), dtype=np.float32))


# --------------------------------------------------------------------------------------
# Échelle de Monk — F-I6
# --------------------------------------------------------------------------------------


def test_la_table_monk_est_strictement_decroissante():
    """A est le plus clair, J le plus foncé : la monotonie garantit qu'un slider de teint
    se comporte de façon prévisible."""
    assert list(MONK_ITA_CENTERS) == sorted(MONK_ITA_CENTERS, reverse=True)
    assert len(MONK_LABELS) == len(MONK_ITA_CENTERS) == 10


@pytest.mark.parametrize("label", MONK_LABELS)
def test_aller_retour_monk_ita(label):
    assert ita_to_monk(monk_to_ita(label)) == label


def test_monk_accepte_la_notation_numerique():
    assert monk_to_ita("1") == monk_to_ita("A")
    assert monk_to_ita("10") == monk_to_ita("J")


@pytest.mark.parametrize("bad", ["K", "0", "11", "zzz"])
def test_monk_rejette_les_echelons_invalides(bad):
    with pytest.raises(ValueError):
        monk_to_ita(bad)


@pytest.mark.parametrize(
    "ita,expected",
    [(70.0, "very light"), (48.0, "light"), (35.0, "intermediate"),
     (20.0, "tan"), (0.0, "brown"), (-40.0, "dark")],
)
def test_categories_dermatologiques(ita, expected):
    assert ita_to_category(ita) == expected


def test_ita_to_monk_est_vectorise():
    labels = ita_to_monk(np.array([66.0, 31.0, -45.0]))
    assert list(labels) == ["A", "E", "J"]


# --------------------------------------------------------------------------------------
# Résolution de mesure — comparabilité entre les trois paliers (§6.2)
# --------------------------------------------------------------------------------------


def test_l_ita_est_mesurable_a_la_resolution_du_palier_0():
    """Palier 0 : 32×32. Sans mise à l'échelle de mesure, la région de peau ne compterait
    qu'une soixantaine de pixels, tomberait sous `min_pixels`, et l'encodage du palier
    smoke échouerait sur la totalité des images."""
    from PIL import Image

    big = _skin_patch((208, 166, 134), size=128)
    small = np.asarray(Image.fromarray(big).resize((32, 32), Image.BILINEAR), dtype=np.uint8)

    result = compute_ita(small, use_mediapipe=False)
    assert result.valid
    assert result.n_pixels > 200


@pytest.mark.parametrize("size", [32, 64, 128, 224])
def test_l_ita_est_stable_a_travers_les_paliers(size):
    """La métrique doit avoir la même définition aux trois paliers, sinon les ΔITA
    mesurés en 32², 64² et 128² ne seraient pas comparables entre eux."""
    from PIL import Image

    reference_image = _skin_patch((208, 166, 134), size=128)
    reference = compute_ita(reference_image, use_mediapipe=False).ita

    scaled = np.asarray(
        Image.fromarray(reference_image).resize((size, size), Image.BILINEAR), dtype=np.uint8
    )
    assert compute_ita(scaled, use_mediapipe=False).ita == pytest.approx(reference, abs=2.0)


def test_une_image_deja_assez_grande_n_est_pas_retouchee():
    from facedit.data.ita import _ensure_measurement_size

    image = _skin_patch((208, 166, 134), size=224)
    assert _ensure_measurement_size(image) is image


# --------------------------------------------------------------------------------------
# Garde-fou achromatique — ~6 % de FairFace
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("level", [70, 120, 180, 220])
def test_une_photo_en_noir_et_blanc_est_declaree_invalide(level):
    """ITA = arctan((L* − 50)/b*) sature à ±90° quand b* → 0. Une image N&B produirait
    donc +90° ou −90° selon sa seule luminosité, sans aucun rapport avec le teint."""
    rng = np.random.default_rng(0)
    grey = np.clip(
        np.full((128, 128, 3), float(level)) + rng.normal(0, 3, (128, 128, 1)), 0, 255
    ).astype(np.uint8)

    result = compute_ita(grey, use_mediapipe=False)
    assert not result.valid
    assert "achromatic" in result.region_source
    assert abs(result.b_star) < 5.0


def test_une_peau_coloree_reste_valide():
    """Contrôle négatif : le garde-fou ne doit pas écarter de la peau normale."""
    for tone in SKIN_TONES:
        result = compute_ita(_skin_patch(tone), use_mediapipe=False)
        assert result.valid, f"teint {tone} rejeté à tort (b* = {result.b_star:.1f})"
        assert result.b_star >= 5.0


def test_le_lot_renvoie_nan_pour_les_mesures_invalides():
    """`compute_ita_batch` ne doit jamais laisser filtrer une valeur saturée comme si
    elle était exploitable — `encode.py` et `eval/` s'appuient sur le masque `valid`."""
    from facedit.data.ita import compute_ita_batch

    grey = np.full((128, 128, 3), 200, dtype=np.uint8)
    skin = _skin_patch((208, 166, 134))
    values, valid = compute_ita_batch(np.stack([skin, grey, skin]), use_mediapipe=False)

    assert list(valid) == [True, False, True]
    assert np.isnan(values[1])
    assert np.isfinite(values[0]) and np.isfinite(values[2])


def test_le_seuil_achromatique_est_ajustable():
    grey = np.full((128, 128, 3), 200, dtype=np.uint8)
    assert compute_ita(grey, use_mediapipe=False, min_b_star=0.0).valid
