# FaceDiT — génération de visages à attributs contrôlés

Projet MSC AIC — **ACHALI Ikram · BENLOUALI Ouiam**

FaceDiT génère des visages à partir de trois consignes : l'**âge** (de 20 à 70 ans, en
continu), le **genre** et le **teint de peau**. Le modèle est un Diffusion Transformer
latent, écrit en PyTorch et entraîné depuis zéro pour ce projet.

Sa particularité est de ne pas se contenter de produire une image : chaque visage généré
est aussitôt re-mesuré, et l'écart entre ce qui a été demandé et ce qui a été obtenu
s'affiche sous l'image.

Le détail de la méthode, des choix et des résultats se trouve dans le rapport de projet,
transmis séparément.

---

## Installation

Python 3.12 recommandé. **Aucune carte graphique n'est nécessaire.**

```bash
git clone https://github.com/ACHALIIkram/image_GEN.git
cd image_GEN

python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# .venv\Scripts\activate.bat        # Windows cmd
# source .venv/bin/activate         # Linux / macOS
```

PyTorch s'installe d'abord, depuis son propre index. Une seule de ces deux lignes :

```bash
# sans carte graphique
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu

# avec une carte NVIDIA (CUDA 12.6)
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

Puis le reste :

```bash
pip install -r requirements.txt
```

---

## Lancer l'interface

```bash
python app.py
```

L'interface s'ouvre sur **http://127.0.0.1:7860**. Les poids entraînés sont fournis dans
le dépôt : il n'y a rien à télécharger ni à entraîner.

Au démarrage, la console indique ce qui a été chargé :

```
[app] chargement du générateur : poids/facedit_v3.pt
[app] 32.96 M paramètres · pas 220000 · calcul sur cpu
[app] oracle de mesure : poids/oracle.pt
```

Le premier lancement télécharge le décodeur d'images depuis Hugging Face (320 Mo, mis en
cache). C'est la seule dépendance réseau.

Options utiles :

```bash
python app.py --port 7861     # changer le port
python app.py --ckpt <fichier>  # utiliser d'autres poids
```

Comptez environ 5 secondes par image sur processeur, moins d'une seconde sur carte
graphique.

---

## Conclusion

Le projet livre un générateur de visages conditionné par trois attributs, accompagné des
instruments qui permettent de vérifier qu'il obéit réellement à ce qu'on lui demande. Cette
vérification est le cœur du travail : un générateur qui affiche « 60 ans » et produit un
visage de 40 ans reste indétectable sans elle.

Tous les visages produits sont synthétiques et ne représentent aucune personne réelle.

Les résultats chiffrés, les limites du modèle et les considérations éthiques sont
développés dans le rapport de projet.
