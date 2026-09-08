"""Tests de configuration — NF-6 (une expérience rejouable depuis un fichier + une seed).

Une configuration qui accepte silencieusement une clé mal orthographiée est pire
qu'aucune configuration : l'expérience tourne, produit un résultat, et personne ne sait
que le paramètre visé n'a jamais été appliqué. D'où le rejet strict des clés inconnues.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from facedit.utils.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["smoke", "proto", "final"])
def test_les_configurations_livrees_se_chargent_et_sont_coherentes(name):
    cfg = load_config(CONFIGS / f"{name}.yaml")
    cfg.validate()
    assert cfg.name == name
    assert cfg.model.latent_size == cfg.data.image_size // 8


@pytest.mark.parametrize(
    "name,image_size,latent_size,tokens",
    [("smoke", 32, 4, 4), ("proto", 64, 8, 16), ("final", 128, 16, 64)],
)
def test_les_trois_paliers_montent_bien_en_resolution(name, image_size, latent_size, tokens):
    """§6.2 : palier 0 en 32², palier 1 en 64², palier 2 en 128²."""
    cfg = load_config(CONFIGS / f"{name}.yaml")
    assert cfg.data.image_size == image_size
    assert cfg.model.latent_size == latent_size
    assert cfg.model.num_patches == tokens


def test_les_valeurs_du_prd_sont_bien_les_defauts():
    """§7.3 et §7.4 : garde-fou contre une dérive silencieuse des hyperparamètres."""
    cfg = load_config(CONFIGS / "final.yaml")
    assert cfg.train.steps == 60_000
    assert cfg.train.batch_size == 128
    assert cfg.train.lr == 1e-4
    assert cfg.train.ema_decay == 0.9999
    assert cfg.train.cond_dropout == 0.1
    assert cfg.train.grad_clip == 1.0
    assert cfg.train.beta_schedule == "squaredcos_cap_v2"
    assert cfg.train.ckpt_every == 5000  # F-T4
    assert cfg.sample.num_steps == 25  # F-S1
    assert cfg.sample.eta == 0.0
    assert cfg.data.vae_scale == 0.18215
    assert cfg.model.patch_size == 2  # F-M2
    assert cfg.model.hidden_size == 256
    assert cfg.model.depth == 8
    assert cfg.model.fourier_num_freqs == 64  # F-M4, F-M6


def test_les_cibles_d_evaluation_suivent_le_paragraphe_8_3():
    cfg = load_config(CONFIGS / "final.yaml")
    # 4 tranches d'âge × 2 genres × 4 bins de teint = 32 cellules
    assert (len(cfg.eval.age_bins) - 1) * 2 * (len(cfg.eval.ita_bins) - 1) == 32
    assert cfg.eval.subgroup_samples_per_cell == 300
    assert cfg.eval.fid_num_samples == 10_000
    assert cfg.eval.leakage_num_seeds == 100


def test_surcharge_en_ligne_de_commande():
    cfg = load_config(
        CONFIGS / "final.yaml",
        ["train.batch_size=32", "sample.guidance=7.5", "seed=99"],
    )
    assert cfg.train.batch_size == 32
    assert cfg.sample.guidance == 7.5
    assert cfg.seed == 99


def test_une_clef_inconnue_est_rejetee(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {"batch_sizeee": 8}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Clés inconnues"):
        load_config(path)


def test_une_surcharge_mal_formee_est_rejetee():
    with pytest.raises(ValueError, match="mal formée"):
        load_config(CONFIGS / "final.yaml", ["train.batch_size 32"])


def test_une_taille_de_latent_incoherente_est_rejetee(tmp_path):
    """Le facteur de compression 8 du VAE n'est pas négociable : une config qui
    l'ignorerait produirait un cache de latents muet mais inutilisable."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump({"data": {"image_size": 128, "latent_size": 8},
                        "model": {"latent_size": 8}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incohérent"):
        load_config(path)


def test_seule_la_cible_epsilon_est_acceptee(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"train": {"prediction_type": "v_prediction"}}), encoding="utf-8")
    with pytest.raises(ValueError, match="epsilon"):
        load_config(path)


def test_le_hash_de_configuration_est_stable_et_discriminant():
    """F-E10 : `results.json` porte le hash de la configuration. Il doit changer si et
    seulement si la configuration change."""
    a = load_config(CONFIGS / "final.yaml")
    b = load_config(CONFIGS / "final.yaml")
    c = load_config(CONFIGS / "final.yaml", ["train.lr=2e-4"])
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_aller_retour_yaml(tmp_path):
    """Une config sauvegardée doit se recharger à l'identique — c'est la condition de
    NF-6, puisque `train.py` écrit `config.yaml` dans le dossier de run."""
    original = load_config(CONFIGS / "proto.yaml", ["seed=7"])
    path = tmp_path / "roundtrip.yaml"
    original.save(path)
    assert load_config(path).hash() == original.hash()


def test_les_valeurs_par_defaut_forment_une_config_valide():
    Config().validate()


def test_base_est_resolu_recursivement(tmp_path):
    """`_base_` doit être suivi sur toute la chaîne, pas seulement d'un niveau.

    Régression réelle : `final_v4 -> final_v3 -> final_v2` ne remontait pas jusqu'à v2, et
    le modèle retombait sur les défauts de la dataclasse (DiT 9.95 M au lieu de 32.96 M).
    Une nuit d'entraînement a été faite sur la mauvaise architecture, sans avertissement.
    """
    from facedit.utils.config import load_config

    (tmp_path / "a.yaml").write_text(
        "name: a\nmodel:\n  depth: 12\n  hidden_size: 384\n  patch_size: 1\n",
        encoding="utf-8",
    )
    # b n'redéclare PAS le modèle : sans récursion, il serait perdu.
    (tmp_path / "b.yaml").write_text(
        "_base_: a.yaml\nname: b\ntrain:\n  steps: 999\n", encoding="utf-8"
    )
    (tmp_path / "c.yaml").write_text(
        "_base_: b.yaml\nname: c\ntrain:\n  batch_size: 7\n", encoding="utf-8"
    )

    cfg = load_config(tmp_path / "c.yaml")
    assert cfg.model.depth == 12, "l'héritage de grand-parent a été perdu"
    assert cfg.model.hidden_size == 384
    assert cfg.model.patch_size == 1
    assert cfg.train.steps == 999, "le parent intermédiaire doit rester appliqué"
    assert cfg.train.batch_size == 7
    assert cfg.name == "c"


def test_base_circulaire_leve_une_erreur_explicite(tmp_path):
    """Mieux vaut une erreur nommée qu'un débordement de pile illisible."""
    import pytest

    from facedit.utils.config import load_config

    (tmp_path / "x.yaml").write_text("_base_: y.yaml\nname: x\n", encoding="utf-8")
    (tmp_path / "y.yaml").write_text("_base_: x.yaml\nname: y\n", encoding="utf-8")

    with pytest.raises(ValueError, match="circulaire"):
        load_config(tmp_path / "x.yaml")
