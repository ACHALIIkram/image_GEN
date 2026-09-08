"""Alignement géométrique des visages par similitude — F-D10.

Pourquoi. FairFace n'est pas aligné : ce sont des recadrages larges autour d'une boîte de
détection, pas des visages calés sur des repères. Mesuré sur 1 410 vignettes 128×128 :

    position horizontale de l'œil   moy 41.2 px   écart-type 7.9 px
    position verticale de l'œil     moy 39.8 px   écart-type 5.3 px
    écart oculaire (échelle)        moy 46.6 px   écart-type 6.2 px, de 10 à 58
    roulis (tête penchée)                         écart-type 6.2°, p95 13.2°

Le latent fait 16×16, donc **une case de latent vaut 8 pixels**. L'œil se déplace sur plus
d'une case entière d'une image à l'autre, et l'échelle du visage varie de 13 %. Le modèle
dépense donc une part de ses 33 M de paramètres à apprendre OÙ poser un visage et à QUELLE
taille, avant d'apprendre à quoi il ressemble.

C'est exactement ce que FFHQ et CelebA-HQ éliminent par construction, et c'est la première
raison de leur qualité apparente — avant leur résolution. On récupère ici le même bénéfice
sans changer de corpus, donc sans perdre les étiquettes humaines d'âge et de genre de
FairFace ni son équilibre démographique (7 groupes entre 11.6 % et 21.0 %), qui est le
sujet même du projet.

Comment. Une similitude — rotation, échelle uniforme, translation — calée sur les deux
yeux. Deux points suffisent à la déterminer.

Effet réellement mesuré, en re-détectant les yeux sur 533 images alignées :

    position œil x    7.1 px  ->  3.3 px
    position œil y    4.8 px  ->  3.7 px
    écart oculaire    5.9 px  ->  2.6 px
    roulis            5.9°    ->  3.5°

La variance est donc DIVISÉE PAR DEUX, pas annulée. La transformation est pourtant exacte
au regard des points détectés : ce qui subsiste est la répétabilité de MediaPipe lui-même,
qui ne redonne pas exactement les mêmes coordonnées sur l'image rééchantillonnée. Le gain
est plafonné par la précision du détecteur, pas par la géométrie. Annoncer « zéro par
construction » serait exact en théorie et faux en pratique.

On ne corrige PAS le lacet (rotation hors-plan) : une similitude 2D ne peut pas redresser
un profil. C'est le filtre de pose (`filter_faces.max_yaw`) qui s'en charge, et les deux
sont complémentaires.

L'ordre des points-clés MediaPipe a été vérifié : `keypoints[1].x > keypoints[0].x` dans
100 % de 370 détections, donc la correspondance œil→cible est directe et ne peut pas
provoquer d'effet miroir.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Position canonique des yeux, en fraction de la taille de sortie.
#
# EYE_Y à 0.34 place le regard légèrement au-dessus du centre, ce qui laisse la place du
# menton sans gaspiller de pixels en front — cadrage usuel des jeux alignés.
#
# L'écart oculaire cible vaut 0.36 × 128 = 46.1 px, choisi pour coïncider avec la moyenne
# déjà observée sur FairFace (46.6 px). Le rééchantillonnage moyen est donc proche de
# l'identité : on redresse sans agrandir ni réduire l'ensemble du corpus, ce qui éviterait
# soit du flou d'interpolation, soit une perte de champ.
EYE_Y = 0.34
EYE_LEFT_X = 0.32
EYE_RIGHT_X = 0.68

MIN_EYE_GAP_PX = 8.0
"""En deçà, les deux yeux sont trop proches pour définir une similitude stable : la
matrice amplifierait le bruit de détection. Concerne les profils marqués, que le filtre de
pose écarte déjà — c'est une seconde barrière, pas la principale."""


def canonical_eyes(out_size: int) -> np.ndarray:
    """Coordonnées cibles des deux yeux, en pixels, pour une sortie carrée `out_size`."""
    return np.array(
        [
            [EYE_LEFT_X * out_size, EYE_Y * out_size],
            [EYE_RIGHT_X * out_size, EYE_Y * out_size],
        ],
        dtype=np.float64,
    )


def similarity_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Matrice affine 2×3 de la similitude envoyant `source[i]` sur `target[i]`.

    Deux points déterminent exactement une similitude directe (sans miroir) : l'échelle est
    le rapport des distances, l'angle la différence des orientations. On l'écrit à la main
    plutôt que d'appeler un estimateur au moindre carré — avec deux points la solution est
    exacte, un solveur itératif n'apporterait qu'une dépendance et des cas limites.
    """
    p0, p1 = source.astype(np.float64)
    q0, q1 = target.astype(np.float64)

    source_vector = p1 - p0
    target_vector = q1 - q0
    source_norm = float(np.linalg.norm(source_vector))
    if source_norm < 1e-6:
        raise ValueError("les deux points source sont confondus : similitude indéfinie")

    scale = float(np.linalg.norm(target_vector)) / source_norm
    angle = float(
        np.arctan2(target_vector[1], target_vector[0])
        - np.arctan2(source_vector[1], source_vector[0])
    )
    cos, sin = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    translation = q0 - rotation @ p0
    return np.hstack([rotation, translation.reshape(2, 1)])


_DETECTOR_STATE: Dict[str, object] = {}


def _get_detector(min_confidence: float = 0.5):
    if "detector" not in _DETECTOR_STATE:
        try:
            import mediapipe as mp

            # model_selection=0, cohérent avec `filter_faces` : mesuré à 92.7 % de
            # détection sur FairFace contre 38.5 % pour le modèle « full range ».
            _DETECTOR_STATE["detector"] = mp.solutions.face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=min_confidence
            )
        except Exception:  # pragma: no cover - dépend de l'environnement
            _DETECTOR_STATE["detector"] = None
    return _DETECTOR_STATE["detector"]


def eye_points(image_uint8: np.ndarray, min_confidence: float = 0.5) -> Optional[np.ndarray]:
    """Coordonnées pixel des deux yeux, ou None si aucun visage exploitable."""
    detector = _get_detector(min_confidence)
    if detector is None:
        return None
    result = detector.process(np.ascontiguousarray(image_uint8))
    if not result.detections:
        return None
    height, width = image_uint8.shape[:2]
    keypoints = result.detections[0].location_data.relative_keypoints
    points = np.array(
        [[keypoints[0].x * width, keypoints[0].y * height],
         [keypoints[1].x * width, keypoints[1].y * height]],
        dtype=np.float64,
    )
    if float(np.linalg.norm(points[1] - points[0])) < MIN_EYE_GAP_PX:
        return None
    return points


def align_face(
    image_uint8: np.ndarray, out_size: int = 128, min_confidence: float = 0.5
) -> Optional[np.ndarray]:
    """Redresse et recadre un visage vers la géométrie canonique. None si impossible.

    L'alignement est appliqué à l'image **d'origine**, pas à une version déjà redimensionnée
    à 128 : le recadrage et le redimensionnement se font en un seul rééchantillonnage, ce
    qui évite d'empiler deux interpolations et le flou qui va avec.

    Le remplissage des bords est un miroir (`BORDER_REFLECT_101`). Un remplissage noir
    apprendrait au modèle à produire des bandes noires — un mode parasite bien visible sur
    les visages proches du bord.
    """
    import cv2

    points = eye_points(image_uint8, min_confidence)
    if points is None:
        return None

    matrix = similarity_matrix(points, canonical_eyes(out_size))
    return cv2.warpAffine(
        image_uint8,
        matrix.astype(np.float32),
        (out_size, out_size),
        flags=cv2.INTER_AREA if matrix[0, 0] < 1.0 else cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def alignment_residual(image_uint8: np.ndarray, out_size: int = 128) -> Optional[Tuple[float, float]]:
    """(erreur de position des yeux en px, écart d'angle en degrés) après alignement.

    Sert à vérifier que la transformation fait bien ce qu'elle prétend, en re-détectant les
    yeux sur l'image alignée. Le résidu n'est pas nul : la détection elle-même est bruitée,
    et c'est justement ce bruit résiduel qu'on mesure ici plutôt que de le supposer absent.
    """
    aligned = align_face(image_uint8, out_size)
    if aligned is None:
        return None
    points = eye_points(aligned)
    if points is None:
        return None
    target = canonical_eyes(out_size)
    position_error = float(np.linalg.norm(points - target, axis=1).mean())
    vector = points[1] - points[0]
    angle_error = float(abs(np.degrees(np.arctan2(vector[1], vector[0]))))
    return position_error, angle_error
