"""Tests du rejet à la génération (F-S6) et de l'alignement (F-D10)."""

from __future__ import annotations

import numpy as np
import pytest

from facedit.data.align import (
    canonical_eyes, eye_points, similarity_matrix,
)
from facedit.quality import acceptance_mask, generate_accepted


# --------------------------------------------------------------------------------------
# Alignement — la similitude
# --------------------------------------------------------------------------------------


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (matrix[:, :2] @ points.T).T + matrix[:, 2]


def test_similitude_envoie_exactement_les_yeux_sur_leurs_cibles():
    """C'est la propriété qui définit l'alignement : elle doit être exacte, pas approchée."""
    source = np.array([[30.0, 55.0], [88.0, 61.0]])
    target = canonical_eyes(128)
    matrix = similarity_matrix(source, target)
    assert np.allclose(_apply(matrix, source), target, atol=1e-9)


def test_similitude_ne_produit_jamais_de_miroir():
    """Un déterminant négatif retournerait le visage. Vérifié sur des poses variées.

    L'ordre des points MediaPipe a été mesuré stable (kp[1].x > kp[0].x dans 100 % de 370
    détections), mais la garantie doit tenir dans la matrice elle-même, pas seulement dans
    une statistique sur un corpus.
    """
    target = canonical_eyes(128)
    for angle in (-40.0, -12.0, 0.0, 17.0, 55.0):
        radians = np.radians(angle)
        offset = np.array([np.cos(radians), np.sin(radians)]) * 40.0
        source = np.array([[50.0, 50.0], [50.0, 50.0] + offset])
        matrix = similarity_matrix(source, target)
        assert np.linalg.det(matrix[:, :2]) > 0, f"miroir introduit à {angle}°"


def test_similitude_preserve_les_angles():
    """Une similitude conserve les rapports de longueur : c'est ce qui la distingue d'une
    affine générale, qui déformerait le visage."""
    source = np.array([[10.0, 20.0], [60.0, 30.0]])
    matrix = similarity_matrix(source, canonical_eyes(128))
    linear = matrix[:, :2]
    produit = linear @ linear.T
    # A·Aᵀ doit être un multiple de l'identité pour une similitude.
    assert produit[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert produit[0, 0] == pytest.approx(produit[1, 1], rel=1e-9)


def test_similitude_refuse_deux_points_confondus():
    with pytest.raises(ValueError, match="confondus"):
        similarity_matrix(np.array([[5.0, 5.0], [5.0, 5.0]]), canonical_eyes(128))


def test_cibles_canoniques_symetriques_et_ordonnees():
    target = canonical_eyes(128)
    assert target[0, 0] < target[1, 0], "l'œil gauche doit être à gauche"
    assert target[0, 1] == target[1, 1], "les deux yeux doivent être à la même hauteur"
    assert abs((target[0, 0] + target[1, 0]) / 2 - 64.0) < 1e-9, "visage non centré"


def test_eye_points_rejette_un_ecart_oculaire_degenere(monkeypatch):
    """Sous `MIN_EYE_GAP_PX`, la matrice amplifierait le bruit de détection."""
    import facedit.data.align as align

    class _Point:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class _Detection:
        def __init__(self, gap):
            pts = [_Point(0.5, 0.4), _Point(0.5 + gap, 0.4), _Point(0.5, 0.5)]
            self.location_data = type("L", (), {"relative_keypoints": pts})()

    class _Result:
        def __init__(self, gap):
            self.detections = [_Detection(gap)]

    class _Detector:
        def __init__(self, gap):
            self.gap = gap

        def process(self, _):
            return _Result(self.gap)

    image = np.zeros((128, 128, 3), np.uint8)
    monkeypatch.setitem(align._DETECTOR_STATE, "detector", _Detector(0.01))
    assert eye_points(image) is None, "écart de 1.3 px : doit être refusé"

    monkeypatch.setitem(align._DETECTOR_STATE, "detector", _Detector(0.30))
    assert eye_points(image) is not None, "écart de 38 px : doit être accepté"


# --------------------------------------------------------------------------------------
# Rejet à la génération
# --------------------------------------------------------------------------------------


def _stub_quality(monkeypatch, verdicts):
    """Force `face_quality` à renvoyer une suite de verdicts imposée."""
    import facedit.quality as quality

    sequence = iter(verdicts)
    cache = {}

    def fake(image, min_confidence=0.5):
        key = int(image[0, 0, 0])
        if key not in cache:
            cache[key] = next(sequence)
        return cache[key]

    monkeypatch.setattr(quality, "face_quality", fake)


def test_acceptance_mask_motive_chaque_rejet(monkeypatch):
    _stub_quality(monkeypatch, [
        {"detected": True, "n_faces": 1, "yaw": 0.05},
        {"detected": False, "n_faces": 0, "yaw": None},
        {"detected": True, "n_faces": 2, "yaw": 0.02},
        {"detected": True, "n_faces": 1, "yaw": 0.80},
    ])
    images = np.stack([np.full((8, 8, 3), i, np.uint8) for i in range(4)])
    keep, reasons = acceptance_mask(images, max_yaw=0.25)

    assert keep.tolist() == [True, False, False, False]
    assert reasons == {"non_detecte": 1, "multi_visages": 1, "trop_tourne": 1, "accepte": 1}


def test_generate_accepted_rend_toujours_le_nombre_demande(monkeypatch):
    """Même quand tout est rejeté : on complète par repli plutôt que de rendre moins.

    Rendre moins d'images que demandé obligerait chaque appelant à gérer un cas partiel.
    On préfère livrer le moins mauvais candidat et le SIGNALER dans `stats`.
    """
    _stub_quality(monkeypatch, [{"detected": False, "n_faces": 0, "yaw": None}] * 200)

    class _Generator:
        def __init__(self):
            self.counter = 0

        def generate(self, *a, batch_size=1, **k):
            out = np.stack([
                np.full((8, 8, 3), (self.counter + i) % 250, np.uint8)
                for i in range(batch_size)
            ])
            self.counter += batch_size
            return {"images": out}

    result = generate_accepted(_Generator(), 30.0, 1, 20.0, count=3, max_rounds=2)
    assert len(result["images"]) == 3
    assert result["stats"]["acceptes"] == 0
    assert result["stats"]["complete_par_repli"] == 3
    assert result["stats"]["rejets"]["non_detecte"] > 0


def test_generate_accepted_s_arrete_des_que_le_quota_est_atteint(monkeypatch):
    """Ne pas gaspiller de GPU : une seule vague doit suffire quand tout passe."""
    _stub_quality(monkeypatch, [{"detected": True, "n_faces": 1, "yaw": 0.01}] * 200)

    class _Generator:
        def __init__(self):
            self.calls = 0
            self.counter = 0

        def generate(self, *a, batch_size=1, **k):
            self.calls += 1
            out = np.stack([
                np.full((8, 8, 3), (self.counter + i) % 250, np.uint8)
                for i in range(batch_size)
            ])
            self.counter += batch_size
            return {"images": out}

    generator = _Generator()
    result = generate_accepted(generator, 30.0, 1, 20.0, count=2, max_rounds=4)
    assert generator.calls == 1
    assert result["stats"]["taux_acceptation"] == 1.0
    assert result["stats"]["complete_par_repli"] == 0
