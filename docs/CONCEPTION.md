# FaceDiT — journal de conception (palier 1)

> **AVERTISSEMENT — document d'archive.** Ce texte a été écrit à la fin du **palier 1** et
> décrit le modèle de cette époque : DiT 9,95 M paramètres, 60 000 pas, FID 32,0. Le modèle
> **livré** est trois paliers plus loin : 32,96 M paramètres, 220 000 pas, FID 20,05. Les
> commandes citées ici pointent encore vers `configs/final.yaml` et `runs/final/`.
>
> | Pour... | Voir |
> |---|---|
> | tester le projet | [`README.md`](../README.md) |
> | les chiffres à jour | le rapport de projet, et `results/*/results_*.json` |
> | reproduire le modèle livré | `scripts/run_full_pipeline.sh` |
>
> Il est conservé parce qu'il documente le raisonnement qui a mené aux paliers suivants —
> en particulier le piège de mesure de la section « Un piège de mesure, trouvé et corrigé »,
> le premier des trois que ce projet a rencontrés. Rien n'y a été réécrit après coup.

---

**Génération de visages photo-réalistes avec contrôle d'attributs vérifiable.**

Générer un visage réaliste est un problème résolu. Le problème ouvert est le **contrôle
vérifiable** : demander « une femme de 62 ans » et obtenir un visage que l'on *mesure* à
62 ans, pas à 48. FaceDiT génère un visage à partir de trois attributs — âge, genre,
teint — et **mesure en temps réel l'écart entre l'attribut demandé et l'attribut obtenu**.

La contribution n'est pas le générateur, c'est la **boucle fermée génération → mesure**,
et le fait que le teint soit traité comme une grandeur physique mesurable (ITA) plutôt
que comme une catégorie ethnique.

> Le seul réseau génératif entraîné par l'équipe est le DiT ([facedit/models/dit.py](facedit/models/dit.py)),
> écrit à la main en PyTorch. Le VAE, le planning de bruit et le modèle de restauration
> sont importés **gelés**. Voir [Périmètre honnête](#périmètre-honnête).

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate                     # Windows
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install mediapipe==0.10.21             # optionnel mais recommandé (masque de peau)
```

Testé sur RTX 4060 Laptop 8 Go / Windows 11 / Python 3.12 / CUDA 12.6.

---

## Reproduire la figure principale en une commande

```bash
python -m facedit.eval.run_eval --ckpt runs/final/ckpt_final.pt
```

Produit `results/final/results_facedit-dit.json` et, dans `results/final/figures/` :
`requested_vs_measured.png` (la figure principale), `leakage_matrix.png`,
`subgroups_*.png` et `w_sweep.png`.

---

## Pipeline complet

```bash
# 0. Vérifier l'hypothèse H3 : le VAE reconstruit-il des visages à cette résolution ?
python -m facedit.data.encode --config configs/final.yaml --check-vae

# 1. Données : ITA + pré-encodage VAE  →  latents.npy / labels.npy
python -m facedit.data.encode --config configs/final.yaml --split train

# 2. Oracle d'attributs (la barre d'erreur de toutes les mesures ultérieures)
python -m facedit.oracle.train_oracle --config configs/final.yaml

# 3. Palier 0 — test de fumée OBLIGATOIRE avant tout entraînement long (~15 min)
python -m facedit.data.encode --config configs/smoke.yaml --split train
python -m facedit.train      --config configs/smoke.yaml
python -m facedit.sample     --ckpt runs/smoke/ckpt_final.pt --sweep

# 4. Entraînement final
python -m facedit.train --config configs/final.yaml
python -m facedit.train --config configs/final.yaml --resume auto   # après coupure

# 5. Évaluation + figures
python -m facedit.eval.run_eval --ckpt runs/final/ckpt_final.pt

# 6. Démonstration
python app.py --ckpt runs/final/ckpt_final.pt
```

Toute expérience est rejouable depuis **un fichier de configuration et une seed** (NF-6).
`train.py` recopie la configuration effective dans le dossier de run, et chaque
`results.json` embarque le hash du checkpoint évalué, le hash de la configuration et
l'empreinte d'environnement.

Surcharges ponctuelles sans éditer le YAML :

```bash
python -m facedit.train --config configs/final.yaml --set train.batch_size=64 sample.guidance=6
```

---

## Architecture

```
FairFace ──► data/ ──► latents.npy ──► train/ ──► checkpoint EMA
              (ITA)                                    │
                                    ┌──────────────────┼──────────────────┐
                                 sample/             eval/              app/
                                 CFG+DDIM          métriques           Gradio
                                                      │                  │
                                              oracle/ ResNet-18 à 3 têtes
```

L'oracle est une dépendance transverse : il sert à l'évaluation **et** à l'affichage
temps réel dans l'interface.

### Contrats entre modules (figés)

```python
latents: np.ndarray   # (N, 4, 16, 16) float16
labels:  np.ndarray   # (N, 3) float32 : [age_years, gender_01, ita_deg]

DiT.forward(x, t, age, gender, ita) -> eps_pred     # gender == 2 → « inconnu » (CFG)
Oracle.predict(images_uint8) -> {"gender", "gender_conf", "age", "ita"}
run_eval(checkpoint_path, config) -> results.json
```

Une baseline évaluable n'a besoin d'exposer qu'une seule méthode, `generate(...)` —
voir [baselines/base.py](baselines/base.py). C'est ce qui permet à StyleGAN2 et à un
modèle texte-image de passer par **le même harnais** que notre DiT.

### Le modèle — DiT-XS

| Paramètre | Valeur |
|---|---|
| Latent | 16×16×4 (VAE `sd-vae-ft-mse`, gelé, échelle 0.18215) |
| Patch / tokens | 2 → 64 tokens |
| Profondeur / dim / têtes | 8 / 256 / 4 |
| Paramètres | **9.95 M** |
| Conditionnement | adaLN-Zero |
| Âge, ITA | features de Fourier (64 fréquences, log-espacées, fixes) |
| Genre | `nn.Embedding(3, 256)` — homme / femme / inconnu |

Le vecteur de condition est une **somme** : `c = emb_t + emb_age + emb_gender + emb_ita`.
Une somme et non une concaténation, pour que l'interpolation d'attributs soit linéaire et
que retirer un attribut isolément (guidage par attribut) ait un sens.

---

## Ce que le projet mesure

| Métrique | Ce qu'elle dit | Mesurée par |
|---|---|---|
| FID / FD-DINOv2 | réalisme | `clean-fid`, DINOv2 |
| Précision de genre, MAE d'âge | obéissance | **oracle** (faillible) |
| ΔITA | obéissance sur le teint | **calcul analytique, sans réseau** |
| LPIPS intra-condition | diversité / effondrement de mode | `lpips` |
| Matrice de fuite 3×3 | désenchevêtrement | oracle + ITA |
| Grille 32 sous-groupes | biais | tout |
| Plus proche voisin | mémorisation | latents + LPIPS |

**Deux instruments distincts, délibérément.** L'âge et le genre passent par l'oracle ;
le teint est mesuré analytiquement. Si les trois attributs étaient mesurés par le même
réseau entraîné sur les mêmes données que le générateur, un biais partagé rendrait le
système *d'apparence* obéissante sans l'être. L'ITA est le point d'ancrage indépendant.

**Toute mesure d'attribut se lit en regard de l'erreur de l'oracle** sur des visages
réels (`runs/oracle/oracle_metrics.json`). Un écart plus petit que cette erreur n'est pas
interprétable.

---

## Résultats mesurés

Modèle final : DiT-XS 9.95 M paramètres, 60 000 pas, **0.84 h** sur RTX 4060 Laptop.
Données : 59 705 visages FairFace retenus (119 410 latents avec miroirs, 245 Mo).
Tous les chiffres viennent de `results/final/results_facedit-dit.json`.

### Confrontation aux cibles du cahier des charges

| Métrique | Cible | Mesuré | Verdict |
|---|---|---|---|
| FID (`clean-fid`, 10k vs 10k) | < 30 | **32.0** | entre cible et seuil d'échec (60) |
| Précision de genre | > 90 % | **98.5 %** | cible atteinte |
| MAE d'âge | < 8 ans | **5.05 ans** | cible atteinte |
| ΔITA | < 10° | **6.40°** | cible atteinte |
| LPIPS intra-condition | > 0.35 | **0.422** | cible atteinte |
| Désenchevêtrement (diag/hors-diag) | > 5 | **16.2** | cible atteinte |

Quatre cibles sur cinq sont atteintes ; le FID termine à 32.0 pour une cible de 30, très
au-dessus du seuil d'échec.

Exigences non fonctionnelles (§5), toutes conformes : latence **0.244 s** (< 3 s), VRAM
entraînement **1.05 Go** (< 6), VRAM inférence **0.489 Go** (< 4), durée d'entraînement
**0.71 h** (< 8), cache en RAM **0.245 Go** (< 1).

### Obéissance aux attributs

La figure principale (`results/final/figures/requested_vs_measured.png`) donne une pente
de régression de **0.87 pour l'âge** et **1.13 pour l'ITA**, pour une obéissance parfaite
à 1.00. Corrélations : r = 0.90 (âge), r = 0.97 (ITA).

La MAE d'âge de 5.05 ans se lit en regard des **5.74 ans d'erreur de l'oracle sur des
visages réels** : l'écart imputable au générateur est essentiellement nul. Cela ne veut
pas dire que le modèle est parfait, mais que **l'instrument de mesure est saturé** — pour
aller plus loin il faudrait un oracle plus précis, pas un meilleur générateur.

### Matrice de fuite (F-E4)

Déplacement en écarts-types du set réel, à seed constante :

| piloté \ mesuré | âge | genre | teint |
|---|---|---|---|
| **âge** | **3.07** | 0.20 | 0.03 |
| **genre** | 0.55 | **1.87** | 0.01 |
| **teint** | 0.11 | 0.02 | **2.56** |

Diagonale forte, hors-diagonale quasi nulle : ratio **16.2**. Le seul couplage notable est
genre → âge (0.55) — changer le genre déplace l'âge perçu d'un demi écart-type.

### Balayage de `w` — question ouverte n°4 tranchée

| w | FID | genre | MAE âge | ΔITA |
|---|---|---|---|---|
| 1 | 43.8 | 0.834 | 10.17 | 12.56° |
| 3 | 38.0 | 0.981 | 5.16 | 6.20° |
| **4** | **37.1** | **0.990** | **4.77** | **5.22°** |
| 6 | 38.4 | 0.996 | 4.83 | 4.70° |
| 8 | 41.9 | 0.997 | 5.03 | 4.91° |
| 10 | 46.7 | 0.997 | 5.14 | 5.60° |

**w = 4 retenu comme défaut de l'interface** : le FID y est minimal, et l'obéissance y est
déjà à 99 % de ce qu'on gagne en poussant plus loin. Au-delà, le réalisme se dégrade
franchement pour un gain d'obéissance marginal.

### Biais par sous-groupe (F-E7, E-2)

Sur les 32 cellules (âge × genre × teint) : précision de genre au pire **90 %**, MAE d'âge
au pire **6.79 ans**, ΔITA au pire **7.81°**. Cinq cellules n'ont pas assez d'images
réelles pour un FID de référence — ce vide est lui-même un résultat sur la couverture de
FairFace.

### Mémorisation (F-E9, E-3)

**Aucune quasi-copie.** Le test calibre son seuil sur la distance entre personnes réelles
*distinctes* (LPIPS 1er centile = 0.103) plutôt que sur une constante arbitraire ; aucune
des 500 images générées ne descend en dessous.

### L'ITA est exploitable en conditionnement continu

| Grandeur | Valeur |
|---|---|
| Écart-type inter-images (signal) | 42.6° |
| Sensibilité de la médiane à l'éclairage ±15 % (bruit) | 5.0° |
| **Rapport signal/bruit** | **8.5** |

→ **Question ouverte n°1 tranchée : conditionnement continu**, pas 5 bins discrets.
Le plancher d'erreur de ΔITA est ≈ 5° ; le ΔITA obtenu (6.40°) en est à 1.4°.

Corollaire : la MAE ITA de l'oracle de référence (**5.08°**) coïncide avec ce plancher de
bruit. L'oracle a atteint la limite intrinsèque de sa cible.

### L'ITA mesure bien ce qu'il prétend mesurer

ITA médian par groupe perçu (labels FairFace jamais utilisés comme conditionnement,
uniquement comme contrôle de validité) : Latino_Hispanic +21.2°, White +18.6°,
Middle Eastern +3.9°, East Asian +1.4°, Southeast Asian −15.6°, Indian −22.7°,
Black −26.5°. L'ordonnancement attendu est retrouvé.

### Le VAE n'est pas le facteur limitant (H3 validée)

Reconstruction sur 256 visages réels en 128×128 : **PSNR 32.1 dB**, **LPIPS 0.028**. Le
repli 256×256 prévu au §12 est inutile.

### Oracle de référence — la barre d'erreur du protocole (F-O4)

| Métrique | Valeur |
|---|---|
| Précision de genre | 96.04 % |
| MAE d'âge | 5.74 ans |
| MAE d'ITA | 5.08° |
| r(ITA) tête apprise vs calcul analytique | 0.974 |

`r = 0.974` répond à F-O5 : l'ITA est un signal réellement présent dans l'image, pas un
artefact du calcul.

---

## Un piège de mesure, trouvé et corrigé

Une première évaluation donnait **ΔITA = 18.35°**, très au-dessus de la cible, et aurait
conduit au rapport la conclusion « le modèle n'obéit pas au teint ». La décomposition de
l'erreur montrait autre chose : sur 18.35° d'écart absolu, **−17.54° étaient un biais
constant**, pas de la dispersion.

Un décalage constant n'est pas de la désobéissance, c'est un problème d'étalonnage. La
cause : les labels d'entraînement avaient été calculés sur la région de peau **MediaPipe**,
tandis que l'évaluation mesurait sur le **gabarit géométrique** de repli. Vérification sur
les *mêmes* images réelles : ITA médian +1.8° avec MediaPipe contre −21.1° avec le
gabarit, soit ~20° d'écart systématique — l'ordre de grandeur exact du biais observé.

Avec l'instrument rendu cohérent entre construction des labels et évaluation, ΔITA tombe à
**6.40°**, biais résiduel −2.5°.

La leçon vaut d'être retenue au-delà de ce projet : **une métrique ne vaut que si
l'instrument est le même des deux côtés de la comparaison**. Le garde-fou est en place
dans le code (`generate_and_measure`), et la proportion réellement mesurée est publiée
(`ita_measurable_fraction` = 99.95 %) pour que tout échec de détection reste visible.

---

## Décisions de conception qui s'écartent de la lecture naïve du cahier des charges

Chacune est assumée et documentée dans le code, à l'endroit concerné.

**La tranche d'âge `10-19` est rejetée en entier**, pas rognée à [18, 20). L'âge exact
intra-tranche est inconnu : rogner étiquetterait des enfants de 12 ans comme de jeunes
adultes. Conséquence : la plage réellement couverte est **[20, 70)** et non [18, 70).
Réglable par `data.drop_partial_age_bins`.

**~6 % de FairFace est écarté du cache pour cause de mesure d'ITA achromatique.**
ITA = arctan((L*−50)/b*) sature à ±90° quand b* → 0, ce qui est le cas des photographies
en noir et blanc. Ces images produiraient +90° ou −90° selon leur seule luminosité, et
ces valeurs aberrantes se logeraient exactement aux extrêmes de la plage d'ITA — là où le
contrôle est le plus difficile et où l'évaluation le sonde. Elles sont rejetées plutôt que
bornées, pour ne pas accumuler une masse artificielle aux extrémités.

**L'ITA est toujours mesuré à 128×128**, quelle que soit la résolution du palier. C'est
une statistique de couleur : la mesurer sur du 32×32 ne donnerait qu'une soixantaine de
pixels de peau et, surtout, changerait la définition de la métrique d'un palier à l'autre.

**Le curseur de genre de l'interface est continu, le conditionnement reste ternaire.**
Le slider interpole les *embeddings* (`p·emb(femme) + (1−p)·emb(homme)`) ; à p = 0 et
p = 1 il coïncide exactement avec les jetons appris que l'évaluation mesure. Les valeurs
intermédiaires explorent l'espace de condition, elles ne désignent pas une catégorie.

**Le guidage par attribut utilise une décomposition télescopique** plutôt qu'une variante
« leave-one-out », parce qu'elle se réduit **exactement** au CFG standard quand les trois
échelles sont égales — propriété vérifiée par un test. Le mode F-S6 est donc une
généralisation stricte du mode par défaut, et le balayage de `w` reste comparable.

**L'évaluation est faite sur les images 128×128 brutes, jamais après restauration.**
CodeFormer est entraîné sur des visages réels : l'appliquer avant le FID rapprocherait
mécaniquement nos images de la distribution réelle sans que le générateur y soit pour rien.

**Le test de mémorisation se calibre sur le jeu lui-même, il n'a pas de seuil codé en
dur.** Une première version comparait la distance LPIPS au plus proche voisin à une
constante (0.15) : au palier 0, elle déclarait *100 % des sorties* quasi-copies, parce
que LPIPS se contracte quand la résolution baisse et que deux visages flous quelconques
sont perceptuellement proches. Un seuil absolu mesure donc autant la résolution que la
mémorisation. La référence correcte est la distance au plus proche voisin **entre
personnes réelles distinctes** (en excluant l'image elle-même et son miroir, puisque le
cache contient les deux). Une image générée n'est suspecte que si elle est plus proche
d'une image d'entraînement que ne le sont deux personnes différentes.

---

## Périmètre honnête

**Entraîné par nous :** le DiT (9.95 M paramètres) et l'oracle (ResNet-18 *fine-tuné*).

**Importé gelé :** le VAE `stabilityai/sd-vae-ft-mse`, les plannings `DDPMScheduler` /
`DDIMScheduler` de `diffusers`, MediaPipe, CodeFormer, StyleGAN2-ADA, tout modèle T2I de
baseline.

**Hors périmètre**, explicitement : génération à identité imposée, édition d'une photo
réelle fournie par l'utilisateur, expression, pose, coiffure, arrière-plan, résolution
native supérieure à 128×128 (la montée en résolution est un post-traitement), vidéo,
déploiement en production.

---

## Éthique — des mesures, pas une annexe

- **E-1** Le teint est conditionné par l'ITA, grandeur colorimétrique mesurable, **jamais**
  par une catégorie ethnique. Les labels `race` de FairFace ne servent qu'au contrôle de
  validité et à l'audit de biais.
- **E-2** Performances ventilées sur 32 sous-groupes (âge × genre × teint), publiées en
  heatmap. L'oracle lui-même est audité par sous-groupe : l'instrument de mesure a ses
  propres biais.
- **E-3** Test de mémorisation par plus proche voisin ; toute quasi-copie d'une personne
  réelle est signalée.
- **E-4 / F-I8** Toute image sortant de l'interface porte la mention « IMAGE GÉNÉRÉE PAR
  IA » **incrustée dans les pixels** — un libellé HTML disparaîtrait au premier
  « enregistrer l'image ».
- **E-5** Les labels de FairFace sont des **perceptions d'annotateurs**, pas des vérités
  biologiques. Le genre binaire des labels sources est une limitation du jeu de données,
  pas une position de l'équipe.
- **E-8** Double usage : un système qui contrôle des attributs démographiques est aussi un
  système qui les classifie. L'oracle embarqué *est* ce classifieur, retourné. Il est
  publié avec ses performances par sous-groupe précisément pour que cet usage soit
  auditable.

---

## Tests

```bash
python -m pytest tests/ -q          # 149 tests aujourd'hui, ~15 s, sans GPU ni dataset
```

Le test qui compte le plus est
`test_conditions_differentes_donnent_sorties_differentes` : c'est le garde-fou exigé
contre le risque n°1 du cahier des charges — un bug de conditionnement silencieux, qui
produit une loss parfaitement décroissante et un modèle qui ignore ses attributs.
Il est vérifié attribut par attribut, parce qu'un conditionnement branché pour l'âge mais
mort pour l'ITA passerait un test global.

---

## Structure

```
configs/          smoke.yaml (32², 3k pas) · proto.yaml (64², 20k) · final.yaml (128², 60k)
                  final_v2/v3/v4.yaml — les paliers postérieurs à ce document
facedit/
  data/           fairface.py · ita.py · encode.py · dataset.py
  models/         dit.py ← le modèle, écrit à la main · embeddings.py · ema.py
  oracle/         model.py · train_oracle.py
  eval/           metrics.py · attributes.py · leakage.py · plots.py · run_eval.py
  train.py · sample.py
baselines/        base.py · stylegan_directions.py · flux_reference.py
tests/            149 tests
app.py            interface Gradio
```

---

## Licences et attributions

FairFace (CC BY 4.0) · `sd-vae-ft-mse` (CreativeML Open RAIL-M) · StyleGAN2-ADA
(NVIDIA Source Code License, poids non redistribués) · CodeFormer (S-Lab License 1.0,
usage non commercial) · MediaPipe (Apache 2.0).
