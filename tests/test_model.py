"""Tests du DiT — dont le garde-fou exigé par le §12.

Risque n°1 du tableau des risques : « L'entraînement ne converge pas (bug de
conditionnement) », probabilité moyenne, impact **critique**, mitigation : « test
unitaire vérifiant que deux conditions différentes produisent des sorties différentes ».
C'est `test_conditions_differentes_donnent_sorties_differentes`.
"""

from __future__ import annotations

import pytest
import torch

from facedit.models.dit import DiT
from facedit.models.ema import EMA


@pytest.fixture(scope="module")
def model() -> DiT:
    torch.manual_seed(0)
    return DiT(latent_size=16, depth=2, hidden_size=64, num_heads=4).eval()


def _inputs(batch: int = 4, latent: int = 16):
    torch.manual_seed(123)
    return (
        torch.randn(batch, 4, latent, latent),
        torch.randint(0, 1000, (batch,)),
        torch.tensor([25.0, 40.0, 55.0, 68.0][:batch]),
        torch.tensor([0, 1, 0, 1][:batch]),
        torch.tensor([-30.0, 5.0, 30.0, 60.0][:batch]),
    )


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


def test_forme_de_sortie_conforme_au_contrat(model):
    """Contrat §6.3 : forward(x, t, age, gender, ita) -> eps_pred de la forme de x."""
    x, t, age, gender, ita = _inputs()
    assert model(x, t, age, gender, ita).shape == x.shape


def test_patchify_est_inversible(model):
    x = torch.randn(3, 4, 16, 16)
    assert torch.equal(model.unpatchify(model.patchify(x)), x)


def test_adaln_zero_rend_le_reseau_identiquement_nul_a_l_initialisation():
    """F-M3 : les portes sont nulles à l'initialisation, donc chaque bloc est l'identité
    et la couche finale (zero-init) renvoie exactement zéro. C'est ce qui permet de se
    passer de warmup — si cette propriété casse, l'entraînement peut diverger au démarrage."""
    fresh = DiT(latent_size=8, depth=4, hidden_size=64, num_heads=4).eval()
    x, t, age, gender, ita = _inputs(latent=8)
    assert torch.count_nonzero(fresh(x, t, age, gender, ita)) == 0


def test_nombre_de_tokens_suit_la_taille_de_patch():
    assert DiT(latent_size=16, patch_size=2).num_patches == 64
    assert DiT(latent_size=8, patch_size=2).num_patches == 16
    assert DiT(latent_size=16, patch_size=4).num_patches == 16


@pytest.mark.parametrize(
    "kwargs", [{"hidden_size": 100, "num_heads": 3}, {"latent_size": 15, "patch_size": 2}]
)
def test_configurations_incoherentes_sont_rejetees(kwargs):
    with pytest.raises(ValueError):
        DiT(**kwargs)


# --------------------------------------------------------------------------------------
# Conditionnement — le garde-fou du §12
# --------------------------------------------------------------------------------------


def _trained_stub() -> DiT:
    """Un DiT dont les portes adaLN ne sont plus nulles.

    À l'initialisation, un DiT renvoie zéro **quelle que soit** la condition : tester la
    sensibilité au conditionnement sur un modèle fraîchement initialisé passerait
    toujours pour de mauvaises raisons. On perturbe donc les couches de modulation, ce
    qui simule un modèle ayant commencé à apprendre.
    """
    torch.manual_seed(7)
    model = DiT(latent_size=8, depth=3, hidden_size=64, num_heads=4).eval()
    with torch.no_grad():
        for block in model.blocks:
            block.ada_ln[-1].weight.normal_(0, 0.05)
            block.ada_ln[-1].bias.normal_(0, 0.05)
        model.final.proj.weight.normal_(0, 0.05)
        model.final.ada_ln[-1].weight.normal_(0, 0.05)
    return model


@pytest.mark.parametrize(
    "attribute,low,high",
    [("age", 20.0, 68.0), ("gender", 0, 1), ("ita", -40.0, 60.0)],
)
def test_conditions_differentes_donnent_sorties_differentes(attribute, low, high):
    """§12 — garde-fou contre le bug de conditionnement silencieux.

    Vérifié attribut par attribut : un conditionnement branché pour l'âge mais mort pour
    l'ITA passerait un test global qui ne ferait varier que l'âge.
    """
    model = _trained_stub()
    x = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 500, dtype=torch.long)

    base = {"age": torch.tensor([35.0, 35.0]),
            "gender": torch.tensor([1, 1]),
            "ita": torch.tensor([30.0, 30.0])}

    def run(value):
        kwargs = {k: v.clone() for k, v in base.items()}
        dtype = torch.long if attribute == "gender" else torch.float32
        kwargs[attribute] = torch.full((2,), value, dtype=dtype)
        return model(x, t, **kwargs)

    difference = (run(high) - run(low)).abs().max().item()
    assert difference > 1e-4, (
        f"L'attribut '{attribute}' n'influence pas la sortie du modèle : le "
        f"conditionnement est débranché."
    )


def test_le_jeton_nul_differe_de_toute_valeur_reelle():
    """Le CFG repose sur une condition nulle *distincte* de toute condition valide.
    Si le jeton nul coïncidait avec une valeur d'attribut, le guidage serait sans effet."""
    model = _trained_stub()
    age = torch.tensor([35.0, 35.0])
    gender = torch.tensor([1, 1])
    ita = torch.tensor([30.0, 30.0])
    drop = torch.ones(2, dtype=torch.bool)

    conditioned = model.embed_attributes(age, gender, ita)
    unconditioned = model.embed_attributes(
        age, gender, ita, drop={"age": drop, "gender": drop, "ita": drop}
    )
    assert (conditioned - unconditioned).abs().max() > 1e-4
    assert torch.allclose(unconditioned, model.null_attributes(2, torch.device("cpu")))


def test_dropout_partiel_n_affecte_que_l_attribut_vise():
    """F-M8 : le masque par attribut doit être réellement indépendant."""
    model = _trained_stub()
    age = torch.tensor([30.0, 30.0])
    gender = torch.tensor([0, 0])
    ita = torch.tensor([20.0, 20.0])
    yes = torch.ones(2, dtype=torch.bool)
    no = torch.zeros(2, dtype=torch.bool)

    full = model.embed_attributes(age, gender, ita)
    without_age = model.embed_attributes(age, gender, ita, drop={"age": yes, "gender": no, "ita": no})

    # Retirer l'âge doit changer l'embedding, et changer *uniquement* de la contribution
    # de l'âge : la différence doit égaler emb_age(30) − null_age.
    expected = model.age_embed(age) - model.age_embed.null_token[None, :].expand(2, -1)
    assert torch.allclose(full - without_age, expected, atol=1e-5)


# --------------------------------------------------------------------------------------
# Guidage
# --------------------------------------------------------------------------------------


def test_cfg_a_echelle_1_equivaut_a_l_absence_de_guidage():
    model = _trained_stub()
    x = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 300, dtype=torch.long)
    age, gender, ita = torch.tensor([40.0, 40.0]), torch.tensor([1, 1]), torch.tensor([25.0, 25.0])

    attr = model.embed_attributes(age, gender, ita)
    null = model.null_attributes(2, torch.device("cpu"))
    guided = model.forward_with_cfg(x, t, attr, null, guidance=1.0)
    plain = model(x, t, attr_emb=attr)
    assert torch.allclose(guided, plain, atol=1e-5)


def test_guidage_par_attribut_se_reduit_au_cfg_standard_a_echelles_egales():
    """Propriété revendiquée dans le docstring de `forward_with_per_attribute_cfg`.

    C'est elle qui justifie la décomposition télescopique plutôt qu'une variante
    « leave-one-out » : le mode F-S6 doit être une généralisation stricte du mode par
    défaut, sinon le balayage F-E6 ne serait pas comparable entre les deux.
    """
    model = _trained_stub()
    x = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 700, dtype=torch.long)
    age, gender, ita = torch.tensor([50.0, 22.0]), torch.tensor([0, 1]), torch.tensor([10.0, 45.0])
    w = 3.5

    per_attribute = model.forward_with_per_attribute_cfg(x, t, age, gender, ita, w, w, w)
    attr = model.embed_attributes(age, gender, ita)
    null = model.null_attributes(2, torch.device("cpu"))
    standard = model.forward_with_cfg(x, t, attr, null, guidance=w)

    assert torch.allclose(per_attribute, standard, atol=1e-4)


# --------------------------------------------------------------------------------------
# Embeddings continus
# --------------------------------------------------------------------------------------


def test_l_embedding_d_age_est_continu(model):
    """F-M4 : deux âges proches doivent produire des embeddings proches, sans quoi
    l'interpolation (F-S4) sauterait au lieu de transiter."""
    close = (model.age_embed(torch.tensor([40.0])) - model.age_embed(torch.tensor([40.5]))).norm()
    far = (model.age_embed(torch.tensor([20.0])) - model.age_embed(torch.tensor([65.0]))).norm()
    assert close < far


def test_les_frequences_de_fourier_sont_deterministes():
    """Deux instanciations doivent partager exactement les mêmes fréquences : sinon un
    checkpoint ne serait pas rechargeable à l'identique."""
    a = DiT(latent_size=8, depth=1, hidden_size=64)
    b = DiT(latent_size=8, depth=1, hidden_size=64)
    assert torch.equal(a.age_embed.freqs, b.age_embed.freqs)
    assert torch.equal(a.ita_embed.freqs, b.ita_embed.freqs)


# --------------------------------------------------------------------------------------
# EMA
# --------------------------------------------------------------------------------------


def test_l_ema_suit_le_modele_avant_son_demarrage():
    torch.manual_seed(0)
    model = DiT(latent_size=8, depth=1, hidden_size=64)
    ema = EMA(model, decay=0.999, start_step=10)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model, step=5)  # avant start_step : copie stricte

    for shadow, live in zip(ema.shadow.parameters(), model.parameters()):
        assert torch.allclose(shadow, live)


def test_l_ema_retarde_le_modele_apres_son_demarrage():
    torch.manual_seed(0)
    model = DiT(latent_size=8, depth=1, hidden_size=64)
    ema = EMA(model, decay=0.9, start_step=0)
    ema.update(model, step=0)

    reference = [p.clone() for p in ema.shadow.parameters()]
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    for step in range(1, 5):
        ema.update(model, step=step)

    moved = any(
        not torch.allclose(shadow, start)
        for shadow, start in zip(ema.shadow.parameters(), reference)
    )
    lagging = any(
        not torch.allclose(shadow, live)
        for shadow, live in zip(ema.shadow.parameters(), model.parameters())
    )
    assert moved and lagging


def test_la_rampe_de_decroissance_part_bas_et_converge():
    model = DiT(latent_size=8, depth=1, hidden_size=64)
    ema = EMA(model, decay=0.9999, start_step=0)
    assert ema.current_decay < 0.2
    ema.num_updates = 1_000_000
    assert abs(ema.current_decay - 0.9999) < 1e-6
