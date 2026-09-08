"""Encodage : contrat de sortie et reprise après interruption — F-D5, F-D6, §12.

Le §12 classe « perte suite à un redémarrage Windows » en probabilité **élevée**, et y
répond par des checkpoints d'entraînement. L'encodage méritait la même protection : il
dure une quarantaine de minutes sur le jeu complet, et le perdre à 61 % — ce qui est
arrivé pour de vrai pendant le développement — impose de tout refaire.

Ces tests utilisent une source et un VAE factices : ils vérifient la mécanique du cache
et de la reprise, pas la qualité de l'encodage, et tournent sans GPU ni téléchargement.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import facedit.data.encode as encode_module
from facedit.data.encode import encode_split
from facedit.data.fairface import FairFaceRecord
from facedit.utils.config import Config

NUM_IMAGES = 96
BATCH = 8

# Teints couvrant la plage utile, tous chromatiques (b* franc) pour passer le garde-fou.
TONES = [(246, 224, 208), (232, 198, 172), (208, 166, 134), (172, 128, 96), (128, 92, 68)]


class FakeSource:
    """Source FairFace minimale : images unies de teint connu, attributs déterministes."""

    origin = "fake://encode-test"

    def __init__(self, count: int = NUM_IMAGES):
        self.records = [
            FairFaceRecord(
                key=f"fake/{i}.jpg",
                age_bin="30-39",
                gender=i % 2,
                race=i % 7,
                age_low=20.0 + (i % 5) * 10.0,
                age_high=30.0 + (i % 5) * 10.0,
            )
            for i in range(count)
        ]

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, index: int) -> np.ndarray:
        image = np.zeros((128, 128, 3), dtype=np.float64)
        image[:] = TONES[index % len(TONES)]
        # Grain déterministe : sans variation, l'IQR est nul et le rejet des déciles
        # travaille sur une distribution dégénérée.
        rng = np.random.default_rng(index)
        return np.clip(image + rng.normal(0, 2.0, image.shape), 0, 255).astype(np.uint8)


class FakeVAE:
    dtype = torch.float32


def fake_encode_images(vae, images_uint8, scale, device="cuda"):
    """Latent déterministe dérivé du contenu de l'image.

    Déterministe et content-dependent : c'est ce qui rend le test de reprise probant.
    Un latent aléatoire rendrait toute comparaison entre les deux passes impossible ;
    un latent constant la rendrait triviale.
    """
    array = np.asarray(images_uint8, dtype=np.float32)
    count = array.shape[0]
    per_channel = array.reshape(count, -1, 3).mean(axis=1) / 255.0
    latents = np.zeros((count, 4, 16, 16), dtype=np.float32)
    for row in range(count):
        latents[row, :3] = per_channel[row][:, None, None]
        latents[row, 3] = array[row].std() / 255.0
    return torch.from_numpy(latents)


@pytest.fixture
def cfg(tmp_path) -> Config:
    configuration = Config()
    configuration.device = "cpu"
    configuration.data.cache_dir = str(tmp_path)
    configuration.data.image_size = 128
    configuration.data.latent_size = 16
    configuration.data.encode_batch_size = BATCH
    configuration.data.num_workers = 0
    configuration.data.hflip = True
    configuration.data.ita_use_mediapipe = False
    configuration.validate()
    return configuration


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    monkeypatch.setattr(encode_module, "open_fairface", lambda data, split: FakeSource())
    monkeypatch.setattr(encode_module, "load_vae", lambda *a, **k: FakeVAE())
    monkeypatch.setattr(encode_module, "encode_images", fake_encode_images)


def _read(cfg: Config, split: str = "train"):
    return (
        np.load(cfg.data.latents_path(split)),
        np.load(cfg.data.labels_path(split)),
        np.load(cfg.data.age_bounds_path(split)),
    )


# --------------------------------------------------------------------------------------
# Contrat de sortie — §6.3
# --------------------------------------------------------------------------------------


def test_l_encodage_respecte_le_contrat_de_sortie(cfg):
    meta = encode_split(cfg, "train", overwrite=True)
    latents, labels, bounds = _read(cfg)

    assert latents.dtype == np.float16
    assert latents.shape[1:] == (4, 16, 16)
    assert labels.dtype == np.float32 and labels.shape[1] == 3
    assert len(latents) == len(labels) == len(bounds)
    # F-D6 : le flip double le cache.
    assert meta["n_total"] == 2 * meta["n_base"]
    assert len(latents) == meta["n_total"]


def test_le_miroir_partage_les_attributs_de_l_original(cfg):
    """Un flip horizontal ne change ni l'âge, ni le genre, ni le teint."""
    meta = encode_split(cfg, "train", overwrite=True)
    _, labels, _ = _read(cfg)
    n = meta["n_base"]
    assert np.array_equal(labels[:n], labels[n:])


def test_les_artefacts_temporaires_sont_nettoyes(cfg, tmp_path):
    encode_split(cfg, "train", overwrite=True)
    assert not list(tmp_path.glob(".tmp_*"))


def test_un_cache_existant_est_protege(cfg):
    encode_split(cfg, "train", overwrite=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        encode_split(cfg, "train")


# --------------------------------------------------------------------------------------
# Reprise — §12
# --------------------------------------------------------------------------------------


class _Interrupt(RuntimeError):
    """Simule une coupure brutale (redémarrage, OOM, fin de session)."""


def test_la_reprise_reconstitue_exactement_le_resultat_complet(cfg, monkeypatch):
    """Le cœur du test : interrompre puis reprendre doit produire, octet pour octet, le
    même cache qu'une exécution jamais interrompue."""
    reference_meta = encode_split(cfg, "train", overwrite=True)
    reference = tuple(array.copy() for array in _read(cfg))
    for path in (cfg.data.latents_path("train"), cfg.data.labels_path("train"),
                 cfg.data.age_bounds_path("train")):
        path.unlink()

    # --- passe 1 : coupure au deuxième point de reprise ---
    real_save = encode_module._save_progress
    calls = {"n": 0}

    def save_then_die(*args, **kwargs):
        real_save(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] >= 2:
            raise _Interrupt("coupure simulée")

    monkeypatch.setattr(encode_module, "_save_progress", save_then_die)
    with pytest.raises(_Interrupt):
        encode_split(cfg, "train", overwrite=True, resume=True, checkpoint_every=2)

    progress_file = encode_module._progress_path(cfg.data.cache, "train")
    assert progress_file.exists(), "aucun point de reprise écrit"
    # `with` obligatoire : un NpzFile laissé ouvert verrouille le fichier, et la reprise
    # échouerait en tentant de le remplacer (WinError 5).
    with np.load(progress_file, allow_pickle=True) as state:
        consumed = int(state["consumed"])
    assert 0 < consumed < NUM_IMAGES, f"coupure hors du domaine utile ({consumed})"

    # --- passe 2 : reprise jusqu'au bout ---
    monkeypatch.setattr(encode_module, "_save_progress", real_save)
    resumed_meta = encode_split(cfg, "train", resume=True, checkpoint_every=2)
    resumed = _read(cfg)

    assert resumed_meta["resumed"] is True
    assert resumed_meta["n_base"] == reference_meta["n_base"]
    for name, expected, obtained in zip(("latents", "labels", "bounds"), reference, resumed):
        assert np.array_equal(expected, obtained), f"divergence sur {name} après reprise"


def test_la_reprise_sans_point_de_reprise_repart_de_zero(cfg):
    meta = encode_split(cfg, "train", overwrite=True, resume=True)
    assert meta["resumed"] is False
    assert meta["n_base"] > 0


def test_un_point_de_reprise_d_un_autre_perimetre_est_ignore(cfg, monkeypatch):
    """Un point de reprise construit sur un autre nombre d'images ne doit pas être
    recyclé : mélanger deux encodages produirait un cache silencieusement corrompu."""
    real_save = encode_module._save_progress
    calls = {"n": 0}

    def save_then_die(*args, **kwargs):
        real_save(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] >= 1:
            raise _Interrupt("coupure simulée")

    monkeypatch.setattr(encode_module, "_save_progress", save_then_die)
    with pytest.raises(_Interrupt):
        encode_split(cfg, "train", overwrite=True, resume=True, checkpoint_every=1)

    # Même cache, mais périmètre réduit : le point de reprise porte 96 lignes, la
    # nouvelle exécution en attend 32.
    monkeypatch.setattr(encode_module, "_save_progress", real_save)
    meta = encode_split(cfg, "train", limit=32, resume=True, overwrite=True)
    assert meta["resumed"] is False
    assert meta["n_base"] <= 32


def test_le_point_de_reprise_survit_a_une_ecriture_interrompue(cfg):
    """L'écriture passe par un fichier de transit renommé : un `.part` résiduel ne doit
    jamais être pris pour un point de reprise valide."""
    cache = cfg.data.cache
    cache.mkdir(parents=True, exist_ok=True)
    encode_module._save_progress(
        cache, "train", 10, 16, np.zeros((NUM_IMAGES, 3), np.float32),
        np.zeros((NUM_IMAGES, 2), np.float32), 3, {"geometric": 16}, [1.0, 2.0],
    )
    target = encode_module._progress_path(cache, "train")
    assert target.exists()
    assert not target.with_name(target.name + ".part").exists()

    with np.load(target, allow_pickle=True) as state:
        assert int(state["cursor"]) == 10 and int(state["consumed"]) == 16
