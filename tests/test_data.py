"""Tests de la couche données — F-D1, F-D2, F-D3, et le contrat §6.3."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from facedit.data.dataset import InfiniteSampler, LatentDataset
from facedit.data.fairface import (
    AGE_BIN_BOUNDS,
    GENDER_UNKNOWN,
    age_bin_centers,
    resize_uint8,
    resolve_age_bins,
    sample_continuous_age,
    to_signed_float,
    to_uint8,
)


# --------------------------------------------------------------------------------------
# Filtrage d'âge — F-D2
# --------------------------------------------------------------------------------------


def test_les_tranches_de_mineurs_sont_ecartees():
    kept = resolve_age_bins(18.0, 70.0, drop_partial=True)
    assert "0-2" not in kept and "3-9" not in kept
    assert "more than 70" not in kept


def test_la_tranche_10_19_est_rejetee_en_entier_par_defaut():
    """Décision documentée : rogner `10-19` en [18, 20) étiquetterait des enfants comme
    de jeunes adultes, puisque l'âge exact intra-tranche est inconnu. La plage
    effectivement couverte devient [20, 70), ce que le README énonce."""
    kept = resolve_age_bins(18.0, 70.0, drop_partial=True)
    assert "10-19" not in kept
    assert min(low for low, _ in kept.values()) == 20.0
    assert max(high for _, high in kept.values()) == 70.0


def test_le_rognage_reste_disponible_explicitement():
    kept = resolve_age_bins(18.0, 70.0, drop_partial=False)
    assert kept["10-19"] == (18.0, 20.0)


def test_un_filtre_impossible_leve_une_erreur():
    with pytest.raises(ValueError, match="Aucune tranche"):
        resolve_age_bins(200.0, 300.0, drop_partial=True)


def test_les_bornes_de_tranches_sont_contigues():
    """Une discontinuité laisserait un intervalle d'âge que rien ne peut produire."""
    ordered = sorted(AGE_BIN_BOUNDS.values())
    for (_, high), (low, _) in zip(ordered, ordered[1:]):
        assert high == low


# --------------------------------------------------------------------------------------
# Âge continu — F-D3
# --------------------------------------------------------------------------------------


def test_l_age_continu_reste_dans_sa_tranche():
    bounds = np.array([[20.0, 30.0], [60.0, 70.0]] * 50)
    ages = sample_continuous_age(bounds, np.random.default_rng(0))
    assert np.all(ages >= bounds[:, 0]) and np.all(ages < bounds[:, 1])


def test_l_age_continu_couvre_reellement_la_tranche():
    """Un tirage au centre de tranche n'exposerait le modèle qu'à 5 valeurs distinctes
    et rendrait l'interpolation (F-S4) impossible entre elles."""
    bounds = np.tile(np.array([[30.0, 40.0]]), (2000, 1))
    ages = sample_continuous_age(bounds, np.random.default_rng(0))
    assert len(np.unique(ages)) > 1500
    assert ages.min() < 31.0 and ages.max() > 39.0


def test_le_centre_de_tranche_est_deterministe():
    """C'est la valeur figée dans `labels[:, 0]` par le contrat §6.3."""
    bounds = np.array([[20.0, 30.0], [60.0, 70.0]])
    assert np.allclose(age_bin_centers(bounds), [25.0, 65.0])


# --------------------------------------------------------------------------------------
# Normalisation — F-D1
# --------------------------------------------------------------------------------------


def test_normalisation_vers_moins_un_un():
    image = np.array([[[0, 128, 255]]], dtype=np.uint8)
    signed = to_signed_float(image)
    assert signed.shape == (3, 1, 1)  # HWC → CHW
    assert signed[0, 0, 0] == pytest.approx(-1.0)
    assert signed[2, 0, 0] == pytest.approx(1.0)


def test_aller_retour_uint8_float_signe():
    rng = np.random.default_rng(0)
    original = rng.integers(0, 256, (4, 16, 16, 3), dtype=np.uint8)
    assert np.array_equal(to_uint8(to_signed_float(original)), original)


def test_l_aller_retour_fonctionne_aussi_sur_un_tenseur_torch():
    """`to_uint8` reçoit des sorties de VAE, donc des tenseurs Torch sur GPU."""
    original = np.random.default_rng(1).integers(0, 256, (2, 8, 8, 3), dtype=np.uint8)
    tensor = torch.from_numpy(to_signed_float(original))
    assert np.array_equal(to_uint8(tensor), original)


def test_le_redimensionnement_produit_la_bonne_forme():
    image = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    assert resize_uint8(image, 128).shape == (128, 128, 3)
    # Une image déjà à la bonne taille doit ressortir inchangée, sans passer par PIL.
    already = np.zeros((64, 64, 3), dtype=np.uint8)
    assert np.array_equal(resize_uint8(already, 64), already)


def test_le_jeton_de_genre_inconnu_vaut_deux():
    """Contrat §6.3 : « gender == 2 signifie inconnu (dropout CFG) »."""
    assert GENDER_UNKNOWN == 2


# --------------------------------------------------------------------------------------
# LatentDataset — contrat §6.3
# --------------------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path):
    """Cache synthétique respectant le contrat : latents float16, labels float32 (N, 3)."""
    rng = np.random.default_rng(0)
    count = 64
    latents = rng.normal(0, 1, (count, 4, 4, 4)).astype(np.float16)
    labels = np.stack(
        [
            rng.uniform(20, 70, count),
            rng.integers(0, 2, count),
            rng.uniform(-40, 60, count),
        ],
        axis=1,
    ).astype(np.float32)
    bounds = np.stack([labels[:, 0] - 5.0, labels[:, 0] + 5.0], axis=1).astype(np.float32)

    np.save(tmp_path / "latents.npy", latents)
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "bounds.npy", bounds)
    return tmp_path


def test_le_dataset_renvoie_exactement_ce_que_le_dit_consomme(cache):
    dataset = LatentDataset(cache / "latents.npy", cache / "labels.npy")
    item = dataset[0]
    assert item["latent"].shape == (4, 4, 4)
    assert item["latent"].dtype == torch.float32  # les latents sont stockés en fp16
    assert item["gender"].dtype == torch.long  # index d'embedding
    assert item["age"].dtype == torch.float32
    assert item["ita"].dtype == torch.float32


def test_l_age_est_retire_a_chaque_acces_quand_les_bornes_existent(cache):
    """F-D3 : le tirage est par accès, pas figé à la construction du cache."""
    torch.manual_seed(0)
    dataset = LatentDataset(cache / "latents.npy", cache / "labels.npy", cache / "bounds.npy")
    values = {float(dataset[0]["age"]) for _ in range(20)}
    assert len(values) > 10


def test_sans_bornes_l_age_est_le_centre_de_tranche(cache):
    dataset = LatentDataset(cache / "latents.npy", cache / "labels.npy")
    labels = np.load(cache / "labels.npy")
    assert float(dataset[3]["age"]) == pytest.approx(float(labels[3, 0]), abs=1e-5)


def test_un_cache_incoherent_est_rejete(tmp_path):
    np.save(tmp_path / "latents.npy", np.zeros((10, 4, 4, 4), dtype=np.float16))
    np.save(tmp_path / "labels.npy", np.zeros((7, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="incohérent"):
        LatentDataset(tmp_path / "latents.npy", tmp_path / "labels.npy")


def test_un_cache_absent_donne_un_message_actionnable(tmp_path):
    with pytest.raises(FileNotFoundError, match="facedit.data.encode"):
        LatentDataset(tmp_path / "absent.npy", tmp_path / "labels.npy")


def test_les_statistiques_du_dataset_sont_exploitables(cache):
    stats = LatentDataset(cache / "latents.npy", cache / "labels.npy").stats()
    assert stats["n"] == 64
    assert 0.0 <= stats["female_ratio"] <= 1.0
    assert stats["age_min"] <= stats["age_mean"] <= stats["age_max"]


# --------------------------------------------------------------------------------------
# Échantillonneur
# --------------------------------------------------------------------------------------


def test_l_echantillonneur_infini_couvre_chaque_epoque_sans_repetition():
    sampler = InfiniteSampler(10, seed=0)
    iterator = iter(sampler)
    first = [next(iterator) for _ in range(10)]
    second = [next(iterator) for _ in range(10)]
    assert sorted(first) == list(range(10))
    assert sorted(second) == list(range(10))


def test_l_echantillonneur_est_reproductible_a_seed_egale():
    a = [i for i, _ in zip(iter(InfiniteSampler(20, seed=3)), range(30))]
    b = [i for i, _ in zip(iter(InfiniteSampler(20, seed=3)), range(30))]
    c = [i for i, _ in zip(iter(InfiniteSampler(20, seed=4)), range(30))]
    assert a == b and a != c
