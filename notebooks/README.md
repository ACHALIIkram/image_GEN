# Notebooks

Espace d'exploration. **Rien d'ici n'est une dépendance du pipeline** : tout ce qui
doit être reproductible vit dans `facedit/` et se lance en ligne de commande.

Règle de travail : dès qu'une analyse de notebook sert au rapport, elle est réécrite en
fonction dans `facedit/eval/` et rappelée depuis `run_eval`. Un chiffre du rapport qui
n'existerait que dans l'état d'exécution d'un notebook n'est pas reproductible, et le
critère d'acceptation §9 demande que la figure principale se régénère en une commande.

## Analyses déjà industrialisées (ne pas refaire ici)

| Analyse | Où elle vit |
|---|---|
| Dispersion de l'ITA et plancher de bruit | `facedit.data.ita.ita_dispersion_report` |
| Reconstruction VAE (H3) | `python -m facedit.data.encode --check-vae` |
| Statistiques du dataset encodé | `artifacts/*_dataset_meta.json` |
| Performance de l'oracle par sous-groupe | `runs/oracle/oracle_metrics.json` |
| Matrice de fuite, grille de sous-groupes, balayage de w | `facedit.eval.run_eval` |
| Exigences non fonctionnelles §5 | `python -m facedit.benchmark` |

## Utile à explorer ici

- inspection visuelle des cas où l'ITA est déclaré invalide (photos N&B, masques ratés) ;
- comparaison qualitative des checkpoints à seed fixe ;
- lecture des plus proches voisins du test de mémorisation, qui demande un œil humain.
