"""Interface de démonstration — F-I1 à F-I7.

    python app.py

Les poids sont cherchés dans poids/ puis dans runs/ ; `--ckpt` force un autre fichier.

La proposition de valeur du §1.2 est la **boucle fermée génération → mesure**. Cette
interface l'incarne : sous chaque image s'affiche l'écart entre l'attribut demandé et
l'attribut effectivement obtenu, mesuré à la volée.

Le teint est mesuré analytiquement (calcul ITA, sans réseau) et non par l'oracle : c'est
le point d'ancrage indépendant du protocole. L'âge et le genre passent par l'oracle.

--------------------------------------------------------------------------------------
DÉVIATION ASSUMÉE PAR RAPPORT À E-4 / F-I8 — à consigner dans le rapport.

Le PRD exige que « toute image générée porte une mention visible », ce que la version
précédente réalisait en incrustant « IMAGE GÉNÉRÉE PAR IA » dans les pixels. Sur demande
explicite, l'incrustation est retirée : elle recouvrait le bas du visage et rendait
l'inspection visuelle difficile, ce qui est précisément l'usage de cette interface.

La mention subsiste sous trois formes dans la page : un avertissement permanent en tête,
une ligne sous chaque image produite, et le libellé du cadre lui-même. La fonction
`stamp_ai_notice` est CONSERVÉE et reste appliquée à toute image exportée hors de la page
(`on_download`) : le compromis porte sur l'affichage, pas sur la diffusion. Une image
enregistrée depuis le navigateur par clic droit échappe en revanche à la mention — c'est
le coût réel de ce choix, et il doit être énoncé plutôt que passé sous silence.
--------------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr
import numpy as np

from facedit.data.ita import (
    MONK_ITA_CENTERS,
    MONK_LABELS,
    compute_ita,
    ita_to_monk,
    monk_to_ita,
)
from facedit.sample import load_generator
from facedit.utils.repro import torch_generator  # noqa: F401 - utilisé par sample_latents

AI_NOTICE = "IMAGE GÉNÉRÉE PAR IA"

# Nombre de pas de débruitage, fixé et retiré de l'interface. Mesuré sur ce modèle :
# passer de 25 à 250 pas ne change pas la structure des visages, seulement le temps de
# calcul. Exposer un réglage sans effet mesurable encombre l'interface et invite à
# attribuer au sampler des défauts qui viennent du modèle.
FIXED_STEPS = 25

# Couleurs officielles de la Monk Skin Tone Scale (Google, 2022), du plus clair au plus
# foncé. Elles servent UNIQUEMENT de repère visuel : la grandeur réellement pilotée est
# l'ITA, une mesure physique calculée sur les pixels. Le nuancier approxime, il ne définit
# pas.
MONK_SWATCHES = (
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420",
)

STATE: Dict[str, object] = {"generator": None, "oracle": None}


# --------------------------------------------------------------------------------------
# Mention « généré par IA »
# --------------------------------------------------------------------------------------


def stamp_ai_notice(image_uint8: np.ndarray) -> np.ndarray:
    """Incruste la mention dans les pixels — conservée pour les images exportées.

    Un libellé placé à côté de l'image dans la page disparaît au premier enregistrement.
    Pour toute image qui QUITTE la page, la seule mention qui tienne voyage avec les
    pixels.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(image_uint8).convert("RGB")
    width, height = image.size
    band = max(16, height // 10)
    canvas = Image.new("RGB", (width, height + band), (0, 0, 0))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, height + band // 4), AI_NOTICE, fill=(255, 214, 0))
    return np.asarray(canvas, dtype=np.uint8)


# --------------------------------------------------------------------------------------
# Mesure — la boucle fermée
# --------------------------------------------------------------------------------------


def _gender_label(probability: float) -> str:
    if probability <= 0.05:
        return "Homme"
    if probability >= 0.95:
        return "Femme"
    return f"mélange ({probability:.2f} vers femme)"


def measure(image_uint8: np.ndarray, requested: Dict[str, float]) -> str:
    """Compare demandé et mesuré, et rend le Markdown affiché sous l'image."""
    oracle = STATE["oracle"]
    monk_requested = ita_to_monk(requested["ita"])

    lines = [
        f"**Vous avez demandé** — {requested['age']:.0f} ans · "
        f"{_gender_label(requested['gender'])} · teint {monk_requested}"
    ]

    analytic = compute_ita(image_uint8, use_mediapipe=True)
    parts: List[str] = []
    age_measured = None

    if oracle is not None:
        prediction = oracle.predict(image_uint8[None])
        age_measured = float(prediction["age"][0])
        gender_measured = int(prediction["gender"][0])
        confidence = float(prediction["gender_conf"][0])
        parts.append(f"{age_measured:.0f} ans")
        parts.append(f"{'Femme' if gender_measured else 'Homme'} ({confidence:.2f})")
    else:
        parts.extend(["âge non mesurable", "genre non mesurable"])

    if analytic.valid:
        parts.append(f"teint {ita_to_monk(analytic.ita)}")
    else:
        parts.append("teint non mesurable")

    lines.append(f"**L'image obtenue contient** — {' · '.join(parts)}")

    deltas: List[str] = []
    if age_measured is not None:
        deltas.append(f"âge **{abs(age_measured - requested['age']):.0f} ans**")
    if analytic.valid:
        deltas.append(f"teint **{abs(analytic.ita - requested['ita']):.0f}°** d'ITA")
    if deltas:
        lines.append("Écart : " + " · ".join(deltas))

    baseline = getattr(oracle, "metrics", None) if oracle is not None else None
    if baseline:
        lines.append(
            f"<sub>L'instrument de mesure se trompe lui-même de "
            f"± {baseline.get('age_mae', float('nan')):.1f} ans sur des visages réels. "
            f"Un écart plus petit que cela ne veut rien dire.</sub>"
        )
    return "\n\n".join(lines)


# --------------------------------------------------------------------------------------
# Génération
# --------------------------------------------------------------------------------------


def on_generate(age: float, gender: float, monk_choice: str, strength: float,
                seed: int, lock_seed: bool, reject: bool, enlarge: bool):
    """Génère une image et mesure immédiatement ce qu'elle contient."""
    from dataclasses import replace

    from facedit.sample import decode, encode_conditions_soft_gender, sample_latents

    generator = STATE["generator"]
    if generator is None:
        return None, "**Aucun modèle chargé.**", seed, ""

    ita = monk_to_ita(monk_choice)
    effective_seed = int(seed) if lock_seed else int(np.random.randint(0, 2**31 - 1))

    started = time.time()
    note = ""
    if reject:
        from facedit.quality import generate_accepted

        sample_cfg = replace(generator.cfg.sample, guidance=float(strength),
                             num_steps=FIXED_STEPS)
        outcome = generate_accepted(generator, age, int(round(gender)), ita, count=1,
                                    sample_cfg=sample_cfg, seed=effective_seed)
        images = outcome["images"]
        stats = outcome["stats"]
        note = f" · {stats['tires']} essais pour obtenir cette image"
        if stats["complete_par_repli"]:
            note += " (aucun n'a passé le contrôle, voici le moins mauvais)"
    else:
        attr, null = encode_conditions_soft_gender(generator, age, gender, ita, 1)
        latents = sample_latents(generator, attr, null, num_steps=FIXED_STEPS,
                                 guidance=float(strength),
                                 eta=generator.cfg.sample.eta, seed=effective_seed)
        images = decode(generator, latents)
    elapsed = time.time() - started

    # La mesure porte TOUJOURS sur l'image native 128, jamais sur la version agrandie :
    # un rééchantillonnage modifie les pixels de peau et déplacerait l'ITA. Ce qu'on
    # affiche peut être agrandi ; ce qu'on mesure ne l'est jamais.
    report = measure(images[0], {"age": age, "gender": gender, "ita": ita})

    shown = images[0]
    enlarge_note = ""
    if enlarge:
        from dataclasses import replace as _replace

        from facedit.sample import _RESTORER_STATE, restore_faces

        shown = restore_faces(
            images[:1], _replace(generator.cfg.sample, restore_size=512), generator.device
        )[0]
        method = "CodeFormer" if _RESTORER_STATE.get("net") is not None else "Lanczos"
        enlarge_note = (
            f" · agrandi en 512 par {method}"
            + ("" if method == "CodeFormer"
               else " — simple rééchantillonnage, aucun détail ajouté")
        )
    footer = (
        f"<sub>⚠️ Visage synthétique — ne représente aucune personne réelle. "
        f"Produit en {elapsed:.1f} s · graine {effective_seed}{note}{enlarge_note}</sub>"
    )
    return shown, report, effective_seed, footer


def on_download(image_uint8):
    """Prépare une copie portant la mention incrustée, pour l'export hors de la page."""
    if image_uint8 is None:
        return None
    from PIL import Image

    path = Path("visage_genere_par_ia.png")
    Image.fromarray(stamp_ai_notice(np.asarray(image_uint8, dtype=np.uint8))).save(path)
    return str(path)


def on_interpolate(age_a: float, gender_a: float, monk_a: str,
                   age_b: float, gender_b: float, monk_b: str,
                   strength: float, seed: int, frames: int):
    """Transition continue entre deux réglages, à graine partagée (F-S4, F-I5).

    C'est le vecteur de condition qui est interpolé, pas les trois valeurs affichées :
    le genre est une table à trois entrées, où passer de 0 à 1 par pas de 0,25 n'aurait
    aucun sens numérique. Les étiquettes portées par les vignettes sont donc les valeurs
    *demandées* le long du chemin, pas des valeurs mesurées.

    Le bruit initial est le même pour toutes les vignettes : c'est ce qui conserve
    approximativement le même visage d'un bout à l'autre de la bande. Sans cela, on
    obtiendrait des personnes différentes plutôt qu'une transition.
    """
    from dataclasses import replace

    from facedit.sample import interpolate

    generator = STATE["generator"]
    if generator is None:
        return [], "**Aucun modèle chargé.**"

    started = time.time()
    sample_cfg = replace(generator.cfg.sample, guidance=float(strength),
                         num_steps=FIXED_STEPS)
    result = interpolate(
        generator,
        (float(age_a), int(round(gender_a)), monk_to_ita(monk_a)),
        (float(age_b), int(round(gender_b)), monk_to_ita(monk_b)),
        num_frames=int(frames), seed=int(seed), sample_cfg=sample_cfg,
    )
    elapsed = time.time() - started

    gallery = [
        (image, f"{age:.0f} ans · teint {ita_to_monk(float(ita))}")
        for image, age, ita in zip(result["images"], result["nominal_age"],
                                   result["nominal_ita"])
    ]
    caption = (
        f"<sub>⚠️ Visages synthétiques — aucun ne représente une personne réelle. "
        f"{len(gallery)} images en {elapsed:.1f} s · graine {int(seed)}. "
        f"Les étiquettes sont les réglages demandés le long du chemin ; seule la mesure "
        f"affichée sous l'image principale constate ce qui a réellement été produit."
        f"</sub>"
    )
    return gallery, caption


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------


def _monk_gallery() -> str:
    """Nuancier visuel des dix échelons, avec leur ITA de référence."""
    cells = []
    for label, colour, ita in zip(MONK_LABELS, MONK_SWATCHES, MONK_ITA_CENTERS):
        text = "#111" if ita > 25 else "#fff"
        cells.append(
            f'<div style="flex:1;min-width:52px;background:{colour};color:{text};'
            f'padding:10px 4px;text-align:center;border-radius:6px;font-size:13px">'
            f'<b>{label}</b><br><span style="font-size:11px">{ita:+.0f}°</span></div>'
        )
    return (
        '<div style="display:flex;gap:5px;flex-wrap:wrap;margin:6px 0 2px 0">'
        + "".join(cells)
        + "</div>"
    )


MONK_CHOICES = [
    f"{label} — {name}"
    for label, name in zip(
        MONK_LABELS,
        ("très clair", "clair", "clair doré", "beige", "beige hâlé", "doré",
         "brun clair", "brun", "brun foncé", "très foncé"),
    )
]


def _monk_from_choice(choice: str) -> str:
    return choice.split(" ")[0]


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="FaceDiT — génération de visages") as demo:
        gr.Markdown(
            "# FaceDiT\n"
            "Choisissez un âge, un genre et un teint : le modèle produit un visage, "
            "puis **mesure ce qu'il a réellement produit** et affiche l'écart.\n\n"
            "⚠️ Tous les visages présentés ici sont synthétiques. Aucun ne représente "
            "une personne réelle."
        )

        with gr.Row():
            with gr.Column(scale=1):
                age = gr.Slider(20, 70, value=35, step=1, label="Âge")
                gender = gr.Slider(
                    0.0, 1.0, value=0.5, step=0.01, label="Genre",
                    info="0 = homme, 1 = femme. Les valeurs intermédiaires mélangent "
                         "les deux plutôt que de désigner une catégorie.",
                )
                monk = gr.Dropdown(
                    MONK_CHOICES, value=MONK_CHOICES[4], label="Teint de peau",
                )
                gr.HTML(_monk_gallery())
                gr.Markdown(
                    "**L'échelle de Monk** classe les teints de peau en dix échelons, "
                    "de A (le plus clair) à J (le plus foncé). Elle a été conçue pour "
                    "couvrir la diversité humaine mieux que les échelles médicales "
                    "anciennes, trop centrées sur les peaux claires.\n\n"
                    "Le modèle, lui, ne travaille pas avec ces dix cases mais avec "
                    "l'**ITA**, un angle calculé directement à partir de la couleur des "
                    "pixels de la peau (les degrés affichés sur le nuancier). C'est une "
                    "grandeur physique continue et mesurable : c'est ce qui permet de "
                    "vérifier après coup que le teint demandé a bien été obtenu, au lieu "
                    "de devoir croire le modèle sur parole.\n\n"
                    "<sub>Les couleurs ci-dessus sont un repère visuel approximatif. Un "
                    "vrai teint dépend aussi de l'éclairage, ce que l'ITA prend en "
                    "compte et pas un simple carré de couleur.</sub>",
                )

                with gr.Accordion("Réglages avancés", open=False):
                    strength = gr.Slider(
                        1.0, 5.0, value=2.0, step=0.5, label="Force du contrôle",
                        info="Bas : visages plus naturels et plus variés, mais qui "
                             "suivent moins bien vos réglages. Haut : obéissance stricte, "
                             "au prix d'images plus artificielles. 2 est la valeur qui "
                             "donne le meilleur compromis d'après nos mesures.",
                    )
                    with gr.Row():
                        seed = gr.Number(value=0, precision=0, label="Graine")
                        lock_seed = gr.Checkbox(value=False, label="Garder la même graine")
                    gr.Markdown(
                        "<sub>La <b>graine</b> est le point de départ aléatoire du "
                        "modèle : elle décide de <i>quelle personne</i> apparaît, à "
                        "attributs identiques. En la laissant libre, chaque clic donne "
                        "un visage différent. En la <b>gardant</b>, vous refabriquez "
                        "exactement la même image — utile pour ne changer qu'un seul "
                        "réglage à la fois et voir son effet isolé.</sub>"
                    )
                    enlarge = gr.Checkbox(
                        value=False, label="Agrandir le résultat en 512 × 512",
                        info="Le modèle produit nativement du 128 × 128. L'agrandissement "
                             "rend l'image plus confortable à regarder mais n'ajoute "
                             "aucun détail : il ne peut pas inventer ce que le modèle "
                             "n'a pas généré.",
                    )
                    reject = gr.Checkbox(
                        value=True, label="Écarter automatiquement les images ratées",
                        info="Le modèle rate environ une image sur douze. Cette option "
                             "les détecte et retire une autre graine à la place. "
                             "Décochez pour voir la sortie brute.",
                    )

                button = gr.Button("Générer un visage", variant="primary", size="lg")

            with gr.Column(scale=1):
                # `interactive=False` : sans cela Gradio traite le composant comme une
                # ENTRÉE et affiche téléversement, webcam et collage — trois boutons qui
                # ne sont branchés sur rien puisque ce cadre ne sert qu'à afficher.
                image = gr.Image(label="Visage synthétique", type="numpy", height=430,
                                 interactive=False, show_download_button=False)
                footer = gr.Markdown()
                report = gr.Markdown()
                download = gr.Button("Télécharger avec la mention « généré par IA »",
                                     size="sm")
                download_file = gr.File(label="Fichier prêt", visible=True)

        with gr.Accordion("Passer progressivement d'un visage à un autre", open=False):
            gr.Markdown(
                "Le modèle peut relier deux réglages par une suite d'images : la même "
                "personne y vieillit, change de teint ou glisse d'un genre à l'autre "
                "sans saut brutal. Le **départ** est celui réglé ci-dessus ; indiquez "
                "ici l'**arrivée**."
            )
            gr.Markdown(
                "<sub>C'est ce qu'une liste de catégories ne permet pas : entre « 30 ans » "
                "et « 40 ans » il n'y a rien, alors qu'ici tous les âges intermédiaires "
                "existent.</sub>"
            )
            with gr.Row():
                age_b = gr.Slider(20, 70, value=65, step=1, label="Âge d'arrivée")
                gender_b = gr.Slider(0.0, 1.0, value=0.5, step=0.01,
                                     label="Genre d'arrivée")
                monk_b = gr.Dropdown(MONK_CHOICES, value=MONK_CHOICES[7],
                                     label="Teint d'arrivée")
            frames = gr.Slider(3, 9, value=5, step=1, label="Nombre d'images")
            interpolate_button = gr.Button("Créer la transition")
            strip = gr.Gallery(label="Transition", columns=9, height=210,
                               object_fit="contain", show_download_button=False)
            strip_caption = gr.Markdown()

        button.click(
            lambda a, g, m, s, sd, lk, rj, en: on_generate(
                a, g, _monk_from_choice(m), s, sd, lk, rj, en
            ),
            inputs=[age, gender, monk, strength, seed, lock_seed, reject, enlarge],
            outputs=[image, report, seed, footer],
        )
        download.click(on_download, inputs=[image], outputs=[download_file])
        interpolate_button.click(
            lambda a, g, m, ab, gb, mb, s, sd, n: on_interpolate(
                a, g, _monk_from_choice(m), ab, gb, _monk_from_choice(mb), s, sd, n
            ),
            inputs=[age, gender, monk, age_b, gender_b, monk_b, strength, seed, frames],
            outputs=[strip, strip_caption],
        )


    return demo


# Ordre de recherche des poids. `poids/` d'abord : c'est ce que contient le dépôt cloné,
# donc le seul chemin qui existe sur une machine autre que celles où le modèle a été
# entraîné. `runs/` ensuite, pour que l'interface continue de démarrer sans argument sur
# nos machines, où les checkpoints d'entraînement complets sont encore présents.
GENERATOR_CANDIDATES = ("poids/facedit_v3.pt", "runs/final_v3/ckpt_final.pt")
ORACLE_CANDIDATES = ("poids/oracle.pt", "runs/oracle/oracle_best.pt")


def _first_existing(candidates) -> Optional[str]:
    return next((path for path in candidates if Path(path).exists()), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interface FaceDiT")
    parser.add_argument("--ckpt", default=None,
                        help=f"défaut : le premier trouvé parmi {', '.join(GENERATOR_CANDIDATES)}")
    parser.add_argument("--oracle", default=None,
                        help=f"défaut : le premier trouvé parmi {', '.join(ORACLE_CANDIDATES)}")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Adresse d'écoute. L'adresse littérale plutôt que `localhost` : sur Windows "
             "`localhost` peut se résoudre en ::1 (IPv6) alors que le serveur n'écoute "
             "qu'en IPv4, et Gradio refuse alors de démarrer.",
    )
    args = parser.parse_args()

    checkpoint = args.ckpt or _first_existing(GENERATOR_CANDIDATES)
    if checkpoint is None or not Path(checkpoint).exists():
        # Sans générateur l'interface n'a rien à montrer : mieux vaut s'arrêter avec une
        # consigne qu'ouvrir une page dont chaque bouton renvoie une erreur.
        cherche = checkpoint or " ou ".join(GENERATOR_CANDIDATES)
        raise SystemExit(
            f"[app] Aucun fichier de poids trouvé ({cherche}).\n"
            f"      Les poids sont versionnés dans poids/ : vérifiez que le dépôt a été\n"
            f"      cloné en entier, ou passez un chemin avec --ckpt."
        )

    print(f"[app] chargement du générateur : {checkpoint}")
    STATE["generator"] = load_generator(checkpoint)
    print(f"[app] {STATE['generator'].model.num_parameters / 1e6:.2f} M paramètres "
          f"· pas {STATE['generator'].step} · calcul sur {STATE['generator'].device}")

    oracle_path = args.oracle or _first_existing(ORACLE_CANDIDATES)
    if oracle_path is not None:
        from facedit.oracle.model import load_oracle

        STATE["oracle"] = load_oracle(oracle_path, STATE["generator"].device)
        print(f"[app] oracle de mesure : {oracle_path}")
    else:
        # Dégradation volontaire et annoncée : le teint reste mesuré (calcul analytique,
        # sans réseau), l'âge et le genre ne le sont plus.
        print("[app] AVERTISSEMENT : oracle absent. Le teint restera mesuré, mais pas "
              "l'âge ni le genre.")

    build_interface().launch(server_name=args.host, server_port=args.port,
                             share=args.share, show_api=False)


if __name__ == "__main__":
    main()
