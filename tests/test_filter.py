"""Tests du filtre visage unique + netteté (F-D9) et de son branchement sur l'encodage."""

from __future__ import annotations

import numpy as np
import pytest

from facedit.data.filter_faces import laplacian_variance


def _checkerboard(size: int = 128, period: int = 4) -> np.ndarray:
    """Damier : contenu haute fréquence, variance laplacienne élevée par construction."""
    y, x = np.mgrid[0:size, 0:size]
    pattern = (((x // period) + (y // period)) % 2 * 255).astype(np.uint8)
    return np.repeat(pattern[:, :, None], 3, axis=2)


def _box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Moyenne glissante séparable — flou sans dépendance externe."""
    out = image.astype(np.float32)
    kernel = 2 * radius + 1
    for axis in (0, 1):
        padded = np.pad(out, [(radius, radius) if a == axis else (0, 0) for a in range(3)],
                        mode="edge")
        cumulative = np.cumsum(padded, axis=axis)
        lead = np.take(cumulative, range(kernel - 1, padded.shape[axis]), axis=axis)
        lag = np.take(cumulative, range(0, padded.shape[axis] - kernel + 1), axis=axis)
        first = np.take(lead, [0], axis=axis)
        out = np.concatenate([first, lead[tuple(slice(1, None) if a == axis else slice(None)
                                                for a in range(3))]
                              - lag[tuple(slice(0, -1) if a == axis else slice(None)
                                          for a in range(3))]], axis=axis) / kernel
    return np.clip(out, 0, 255).astype(np.uint8)


def test_laplacian_variance_is_monotone_in_blur():
    """La netteté mesurée doit décroître quand on floute — c'est tout ce qu'on lui demande.

    On ne teste pas une valeur absolue : le seuil du filtre est un quantile de la
    distribution observée, donc seule l'ordonnancement compte.
    """
    sharp = _checkerboard()
    values = [laplacian_variance(sharp)]
    for radius in (1, 2, 4):
        values.append(laplacian_variance(_box_blur(sharp, radius)))

    assert values == sorted(values, reverse=True), (
        f"la netteté doit décroître avec le flou, obtenu {values}"
    )
    assert values[-1] < values[0] / 10.0


def test_laplacian_variance_zero_on_flat_image():
    """Une image uniforme n'a aucune haute fréquence : variance exactement nulle."""
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert laplacian_variance(flat) == pytest.approx(0.0, abs=1e-9)


def test_keep_mask_maps_to_both_halves_of_the_cache():
    """Le cache range `[base, miroir]` : le même masque doit servir aux deux moitiés.

    Ce test fige la convention de rangement dont dépend `--keep-mask`. S'il casse, c'est
    que l'ordre des miroirs a changé et que le filtre sélectionnerait les mauvaises lignes.
    """
    n_base = 7
    keep = np.array([True, False, True, True, False, True, True])
    latents = np.arange(2 * n_base)

    doubled = np.concatenate([keep, keep])
    selected = latents[doubled]

    base_kept = selected[selected < n_base]
    flip_kept = selected[selected >= n_base] - n_base
    assert np.array_equal(base_kept, flip_kept), (
        "un original conservé doit entraîner son miroir, et lui seul"
    )
    assert len(selected) == 2 * int(keep.sum())


def test_encode_split_accepts_keep_mask_signature():
    """`--keep-mask` doit rester câblé jusqu'à `encode_split` (régression d'intégration)."""
    import inspect

    from facedit.data.encode import encode_split

    parameters = inspect.signature(encode_split).parameters
    assert "keep_mask" in parameters
    assert parameters["keep_mask"].default is None


def test_count_faces_degrades_gracefully_without_mediapipe(monkeypatch):
    """Sans MediaPipe, le détecteur renvoie None et le filtre retombe sur la seule netteté.

    Le critère d'unicité est alors inapplicable : il vaut mieux le désactiver franchement
    que de laisser passer un comptage silencieusement faux.
    """
    from facedit.data import filter_faces

    monkeypatch.setitem(filter_faces._DETECTOR_STATE, "detector", None)
    assert filter_faces.count_faces(np.zeros((32, 32, 3), np.uint8), 0.5) is None
