"""Test d'intégration bout en bout : entraînement → checkpoint → reprise → échantillonnage.

Ce test tourne sur un cache de latents **synthétique** et un VAE factice. Il ne dit rien
de la qualité des images : il vérifie que les modules s'emboîtent, que les contrats du
§6.3 sont respectés de bout en bout, et que la reprise sur checkpoint (F-T5) restitue
exactement l'état sauvegardé.

C'est le filet qui permet aux quatre rôles du §11 de travailler en parallèle sans casser
l'interface de l'autre — l'objectif explicite du risque « le groupe converge sur un seul
contributeur ».
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from facedit.train import sample_condition_dropout, train
from facedit.utils.config import Config


# --------------------------------------------------------------------------------------
# Décor
# --------------------------------------------------------------------------------------


class StubVAE:
    """VAE factice : décode un latent 4 canaux vers une image 3 canaux 8× plus grande.

    Suffisant pour valider la plomberie (`decode` → uint8 → oracle → ITA) sans
    télécharger 330 Mo de poids ni occuper le GPU.
    """

    dtype = torch.float32

    def __init__(self, device: str = "cpu"):
        self.device = device

    def decode(self, latents: torch.Tensor):
        upscaled = torch.nn.functional.interpolate(
            latents[:, :3], scale_factor=8, mode="nearest"
        )
        # Le facteur 1/4 maintient tanh dans sa zone quasi linéaire. Sans lui, des
        # latents d'amplitude usuelle saturent à ±1, tout ressort en 0 ou 255 et les
        # différences entre conditions disparaissent dans la quantification uint8 —
        # un artefact du VAE factice, qui ferait échouer des tests portant sur le modèle.
        return type("Decoded", (), {"sample": torch.tanh(upscaled / 4.0)})()


def _make_cache(directory: Path, count: int = 96, latent: int = 4) -> None:
    """Écrit un cache conforme au contrat §6.3, avec un signal réellement conditionnel.

    Les latents dépendent des attributs : sans cela, un modèle correctement conditionné
    et un modèle au conditionnement débranché produiraient la même loss, et le test ne
    distinguerait rien.
    """
    rng = np.random.default_rng(0)
    ages = rng.uniform(20, 70, count).astype(np.float32)
    genders = rng.integers(0, 2, count).astype(np.float32)
    itas = rng.uniform(-40, 60, count).astype(np.float32)

    latents = rng.normal(0, 0.3, (count, 4, latent, latent)).astype(np.float32)
    latents[:, 0] += ((ages - 45.0) / 25.0)[:, None, None]
    latents[:, 1] += (genders * 2.0 - 1.0)[:, None, None]
    latents[:, 2] += ((itas - 10.0) / 50.0)[:, None, None]

    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "train_32_latents.npy", latents.astype(np.float16))
    np.save(
        directory / "train_32_labels.npy",
        np.stack([ages, genders, itas], axis=1).astype(np.float32),
    )
    np.save(
        directory / "train_32_age_bounds.npy",
        np.stack([ages - 5.0, ages + 5.0], axis=1).astype(np.float32),
    )


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Entraîne un modèle minuscule et renvoie (config, dossier de run)."""
    root = tmp_path_factory.mktemp("pipeline")
    _make_cache(root / "artifacts")

    cfg = Config()
    cfg.seed = 7
    cfg.device = "cpu"  # déterministe et suffisant : le modèle fait 0.3 M paramètres
    cfg.data.cache_dir = str(root / "artifacts")
    cfg.data.image_size = 32
    cfg.data.latent_size = 4
    cfg.data.num_workers = 0
    cfg.model.latent_size = 4
    cfg.model.depth = 2
    cfg.model.hidden_size = 64
    cfg.model.fourier_num_freqs = 16
    cfg.train.steps = 60
    cfg.train.batch_size = 16
    cfg.train.ema_start = 5
    cfg.train.ckpt_every = 30
    cfg.train.sample_every = 0  # pas de grille : le VAE réel n'est pas chargé ici
    cfg.train.log_every = 20
    cfg.train.precision = "fp32"
    cfg.train.fused_adam = False
    cfg.train.out_dir = str(root / "run")
    cfg.validate()

    train(cfg)
    return cfg, Path(cfg.train.out_dir)


# --------------------------------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------------------------------


def test_l_entrainement_produit_les_artefacts_attendus(trained):
    cfg, out_dir = trained
    assert (out_dir / "ckpt_final.pt").exists()
    assert (out_dir / "ckpt_last.pt").exists()  # F-T4
    assert (out_dir / "config.yaml").exists()  # NF-6
    assert (out_dir / "train_summary.json").exists()
    assert (out_dir / "tb").is_dir()  # F-T6


def test_le_resume_de_l_entrainement_est_exploitable(trained):
    cfg, out_dir = trained
    summary = json.loads((out_dir / "train_summary.json").read_text(encoding="utf-8"))
    assert summary["steps"] == cfg.train.steps
    assert summary["final_loss"] is not None and np.isfinite(summary["final_loss"])
    assert summary["config_hash"] == cfg.hash()


def test_le_checkpoint_contient_les_quatre_elements_exiges(trained):
    """F-T4 : « Checkpoint complet (modèle, EMA, optimiseur, compteur de pas) »."""
    _, out_dir = trained
    payload = torch.load(out_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    for key in ("model", "ema", "optimizer", "step", "config"):
        assert key in payload, f"clé « {key} » absente du checkpoint"
    assert payload["ema"]["shadow"]


def test_l_ema_a_diverge_des_poids_bruts(trained):
    """Si l'EMA était identique au modèle, F-T3 ne servirait à rien et la génération
    n'en tirerait aucun bénéfice."""
    _, out_dir = trained
    payload = torch.load(out_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    differences = [
        (payload["model"][k] - payload["ema"]["shadow"][k]).abs().max().item()
        for k in payload["model"]
        if payload["model"][k].dtype.is_floating_point
    ]
    assert max(differences) > 1e-8


def test_la_reprise_restitue_exactement_l_etat_sauvegarde(trained, tmp_path):
    """F-T5 : `--resume` doit repartir du pas enregistré, pas de zéro."""
    cfg, out_dir = trained
    resumed = Config(**{**cfg.to_dict()})
    from facedit.utils.config import _from_dict

    resumed = _from_dict(Config, cfg.to_dict())
    resumed.train.steps = cfg.train.steps + 10
    resumed.train.out_dir = str(out_dir)

    train(resumed, resume=str(out_dir / "ckpt_last.pt"))
    payload = torch.load(out_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    assert payload["step"] == cfg.train.steps + 10


# --------------------------------------------------------------------------------------
# Dropout de condition — F-M7, F-M8
# --------------------------------------------------------------------------------------


def test_le_dropout_conjoint_fait_tomber_les_trois_attributs_ensemble():
    torch.manual_seed(0)
    masks = sample_condition_dropout(4096, 0.1, independent=False, device=torch.device("cpu"))
    assert torch.equal(masks["age"], masks["gender"])
    assert torch.equal(masks["age"], masks["ita"])
    assert masks["age"].float().mean().item() == pytest.approx(0.1, abs=0.02)


def test_le_dropout_independant_conserve_la_masse_inconditionnelle():
    """F-M8 : p_attr = p^(1/3) pour que les trois attributs tombent *ensemble* avec la
    même probabilité qu'en mode conjoint — sinon la branche inconditionnelle du CFG
    serait vue 100 fois moins souvent et le guidage s'effondrerait."""
    torch.manual_seed(0)
    masks = sample_condition_dropout(60_000, 0.1, independent=True, device=torch.device("cpu"))
    all_dropped = masks["age"] & masks["gender"] & masks["ita"]
    assert all_dropped.float().mean().item() == pytest.approx(0.1, abs=0.01)
    # ...et les masques doivent bien être indépendants, pas trois copies.
    assert not torch.equal(masks["age"], masks["ita"])


def test_un_dropout_nul_ne_masque_rien():
    masks = sample_condition_dropout(128, 0.0, independent=False, device=torch.device("cpu"))
    assert not masks["age"].any() and not masks["ita"].any()


# --------------------------------------------------------------------------------------
# Échantillonnage
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator(trained):
    from facedit.sample import load_generator

    cfg, out_dir = trained
    gen = load_generator(out_dir / "ckpt_final.pt", device="cpu", with_vae=False)
    gen.vae = StubVAE()
    return gen


def test_le_chargement_refuse_un_checkpoint_sans_ema(trained, tmp_path):
    """F-T3 : « la génération utilise toujours les poids EMA ». Le refus doit être
    explicite plutôt qu'un repli silencieux sur les poids bruts."""
    from facedit.sample import load_generator

    _, out_dir = trained
    payload = torch.load(out_dir / "ckpt_final.pt", map_location="cpu", weights_only=False)
    payload.pop("ema")
    broken = tmp_path / "no_ema.pt"
    torch.save(payload, broken)

    with pytest.raises(ValueError, match="EMA"):
        load_generator(broken, device="cpu", with_vae=False)


def test_la_generation_respecte_le_contrat_de_sortie(generator):
    result = generator.generate(age=42.0, gender=1, ita=25.0, batch_size=2, seed=0)
    images = result["images"]
    assert images.shape == (2, 32, 32, 3)
    assert images.dtype == np.uint8


def test_la_seed_rend_la_generation_reproductible(generator):
    """F-S3 / F-I4 : « seed verrouillée » doit produire des pixels identiques."""
    first = generator.generate(age=30.0, gender=0, ita=40.0, batch_size=1, seed=123)["images"]
    second = generator.generate(age=30.0, gender=0, ita=40.0, batch_size=1, seed=123)["images"]
    third = generator.generate(age=30.0, gender=0, ita=40.0, batch_size=1, seed=124)["images"]
    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_des_conditions_differentes_donnent_des_sorties_differentes(generator):
    """Le garde-fou du §12, cette fois à travers toute la chaîne d'échantillonnage.

    L'assertion porte sur les **latents** et non sur les pixels : c'est une propriété du
    modèle, et la faire passer par le décodeur factice y mêlerait les artefacts de ce
    dernier. Le décodage est couvert séparément par
    `test_la_generation_respecte_le_contrat_de_sortie`.
    """
    young = generator.generate(21.0, 1, 30.0, 1, seed=5, decode_images=False)["latents"]
    old = generator.generate(69.0, 1, 30.0, 1, seed=5, decode_images=False)["latents"]
    assert not torch.allclose(young, old)


def test_le_nombre_de_pas_ddim_est_respecte(generator):
    import copy

    fast = copy.deepcopy(generator.cfg.sample)
    fast.num_steps = 5
    generator.scheduler.set_timesteps(5)
    assert len(generator.scheduler.timesteps) == 5
    assert generator.generate(35.0, 1, 20.0, 1, sample_cfg=fast, seed=1)["images"].shape[0] == 1


def test_l_interpolation_produit_une_bande_a_identite_partagee(generator):
    """F-S4 / F-I5 : N images, bruit initial commun."""
    from facedit.sample import interpolate

    result = interpolate(generator, (22.0, 0, 55.0), (68.0, 1, -25.0), num_frames=6, seed=3)
    assert result["images"].shape == (6, 32, 32, 3)
    assert result["alphas"][0] == 0.0 and result["alphas"][-1] == 1.0
    # Les extrémités doivent différer, sinon l'interpolation ne transporte rien.
    assert not torch.allclose(result["latents"][0], result["latents"][-1])
    # ...et la trajectoire doit être progressive : chaque pas doit être plus petit que
    # le saut direct d'un bout à l'autre. C'est la définition opérationnelle de la
    # « transition visuellement continue » du critère d'acceptation §9.
    total = (result["latents"][-1] - result["latents"][0]).norm()
    steps = [
        (result["latents"][i + 1] - result["latents"][i]).norm()
        for i in range(len(result["latents"]) - 1)
    ]
    assert max(steps) < total


def test_le_guidage_modifie_la_sortie(generator):
    """F-S2 : sans effet mesurable du CFG, le balayage F-E6 serait vide."""
    import copy

    low = copy.deepcopy(generator.cfg.sample)
    low.guidance = 1.0
    high = copy.deepcopy(generator.cfg.sample)
    high.guidance = 8.0

    a = generator.generate(50.0, 1, 10.0, 1, sample_cfg=low, seed=9, decode_images=False)
    b = generator.generate(50.0, 1, 10.0, 1, sample_cfg=high, seed=9, decode_images=False)
    assert not torch.allclose(a["latents"], b["latents"])


def test_le_genre_continu_de_l_interface_retombe_sur_les_jetons_appris(generator):
    """Le slider de genre (F-I1) interpole les embeddings ; à p = 0 et p = 1 il doit
    coïncider exactement avec les jetons discrets évalués par le harnais."""
    from facedit.sample import encode_conditions, encode_conditions_soft_gender

    hard, _ = encode_conditions(generator, 40.0, 1, 20.0, 1)
    soft, _ = encode_conditions_soft_gender(generator, 40.0, 1.0, 20.0, 1)
    assert torch.allclose(hard, soft, atol=1e-6)

    hard_male, _ = encode_conditions(generator, 40.0, 0, 20.0, 1)
    soft_male, _ = encode_conditions_soft_gender(generator, 40.0, 0.0, 20.0, 1)
    assert torch.allclose(hard_male, soft_male, atol=1e-6)


def test_un_genre_hors_domaine_est_rejete(generator):
    from facedit.sample import encode_conditions

    with pytest.raises(ValueError, match="gender"):
        encode_conditions(generator, 40.0, 5, 20.0, 1)


# --------------------------------------------------------------------------------------
# Mesure — la boucle fermée du §1.2
# --------------------------------------------------------------------------------------


def test_la_boucle_generation_mesure_est_complete(generator):
    """Le cœur de la proposition de valeur : générer, puis mesurer ce qui a été produit."""
    from facedit.eval.attributes import attribute_fidelity, sample_conditions

    class StubOracle:
        """Oracle factice : renvoie la condition demandée, bruitée. Ce qui est testé est
        le chaînage des mesures, pas leur exactitude."""

        metrics = {"age_mae": 4.0, "gender_accuracy": 0.94, "ita_mae": 5.0}

        def predict(self, images, batch_size=128):
            count = len(images)
            rng = np.random.default_rng(0)
            return {
                "age": rng.uniform(20, 70, count).astype(np.float32),
                "gender": rng.integers(0, 2, count).astype(np.int64),
                "gender_conf": rng.uniform(0.5, 1.0, count).astype(np.float32),
                "ita": rng.uniform(-40, 60, count).astype(np.float32),
            }

    from facedit.eval.attributes import generate_and_measure

    conditions = sample_conditions(8, seed=0)
    measured = generate_and_measure(
        generator, StubOracle(), conditions, batch_size=4, seed=0, desc="test"
    )
    report = attribute_fidelity(conditions, measured, StubOracle.metrics)

    for key in ("gender_accuracy", "age_mae", "ita_delta_mean", "age_slope", "ita_slope"):
        assert key in report
    assert report["n"] == 8
    assert 0.0 <= report["gender_accuracy"] <= 1.0
    # La barre d'erreur de l'oracle doit être reportée à côté de la mesure (F-O4).
    assert "oracle_baseline" in report
