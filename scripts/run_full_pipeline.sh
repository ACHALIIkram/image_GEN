#!/usr/bin/env bash
# Reproduction complète du modèle livré : données → oracle → v2 → v3 → évaluation.
#
#   bash scripts/run_full_pipeline.sh
#
# Durée totale ≈ 24 h sur RTX 4060 Laptop 8 Go, dont 17 h d'entraînement. Chaque étape
# écrit son journal dans logs/ et le script s'arrête à la première erreur.
#
# CE SCRIPT N'EST PAS NÉCESSAIRE POUR TESTER LE PROJET. Les poids entraînés sont dans
# poids/ ; `python app.py` suffit. Ce fichier existe pour que le résultat soit
# reproductible, pas pour être exécuté par un relecteur.
#
# --- trois contraintes apprises à la dure, à ne pas défaire -----------------------------
#
# STRICTEMENT SÉQUENTIEL. Deux tâches GPU simultanées sur 8 Go se tuent mutuellement :
# l'oracle est mort silencieusement pendant qu'un benchmark tournait.
#
# `num_workers=0` sur les étapes de données. Chaque worker de DataLoader est un processus
# qui charge sa propre copie de PyTorch (~600 Mo). Sur 16 Go dont ~2,4 Go réellement
# libres, les workers ont produit `WinError 1455: The paging file is too small` puis
# `Unable to allocate 12.0 MiB`, avec un encodage qui thrashait (ETA de 20 min à 1 h 10).
# Le coût est faible : l'étape limitante de l'encodage est le calcul d'ITA, mono-thread
# NumPy de toute façon.
#
# MODE HORS LIGNE. Sans `HF_DATASETS_OFFLINE`, `load_dataset` interroge le Hub à chaque
# ouverture du jeu : le filtrage est passé de 64 s à plusieurs heures, sans que rien ne
# l'indique dans les journaux. Le cache local doit être déjà peuplé.
# ---------------------------------------------------------------------------------------

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/Scripts/python.exe
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p logs

step() {
  local name="$1"; shift
  echo ""
  echo "======================================================================"
  echo "[$(date +%H:%M:%S)] $name"
  echo "======================================================================"
  local start=$SECONDS
  if ! "$@" > "logs/$name.log" 2>&1; then
    echo "ÉCHEC — 40 dernières lignes de logs/$name.log :"
    tail -40 "logs/$name.log"
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] $name terminé en $(( (SECONDS - start) / 60 )) min"
  tail -4 "logs/$name.log" | tr '\r' '\n' | grep -viE "^W0000|absl|it/s|img/s" | tail -3 || true
}

# ---------------------------------------------------------------------------------------
# 1. Données
# ---------------------------------------------------------------------------------------

# 1a. Filtre qualité (F-D9) : un seul visage, net, de face. Les seuils sont appliqués
# GLOBALEMENT à la fusion et non par tranche, pour que le résultat ne dépende pas du
# découpage. ~64 s.
step "01a_filter" $PY -m facedit.data.filter_faces --config configs/final_v2.yaml \
     --split train --sharpness-quantile 0.10

# 1b. Même filtre, plus le critère de pose (lacet < 0,35). Masque distinct : le palier v2
# doit rester rejouable à l'identique, ses chiffres sont publiés.
step "01b_filter_pose" $PY -m facedit.data.filter_faces --config configs/final_v2.yaml \
     --split train --sharpness-quantile 0.10 --max-yaw 0.35 --tag pose

# 1c. Pré-encodage VAE + calcul d'ITA analytique (F-D4 à F-D6), un cache par palier.
# `--resume` : l'encodage dure ~40 min et a déjà été perdu une fois à 61 %.
step "01c_encode_v2" $PY -m facedit.data.encode --config configs/final_v2.yaml \
     --split train --resume --keep-mask artifacts/train_128_keep.npy \
     --set data.num_workers=0 data.encode_batch_size=32

step "01d_encode_v3" $PY -m facedit.data.encode --config configs/final_v3.yaml \
     --split train --resume --keep-mask artifacts/train_128_keep_pose.npy \
     --set data.num_workers=0 data.encode_batch_size=32

# 1e. Hypothèse H3 : le VAE reconstruit-il des visages à 128 px sans perdre l'essentiel ?
# Mesuré, pas supposé — 32,14 dB de PSNR, LPIPS 0,028.
step "02_check_vae" $PY -m facedit.data.encode --config configs/final_v2.yaml \
     --split val --check-vae

# ---------------------------------------------------------------------------------------
# 2. Oracle — la barre d'erreur de tout le protocole (F-O2, F-O4)
# ---------------------------------------------------------------------------------------
# À entraîner AVANT le générateur : sans lui, aucune mesure d'attribut n'est possible, et
# un générateur qu'on ne sait pas mesurer ne se débogue pas.
step "03_oracle" $PY -m facedit.oracle.train_oracle --config configs/final_v2.yaml \
     --set oracle.num_workers=0

# ---------------------------------------------------------------------------------------
# 3. Générateur — deux paliers, le second reprenant le premier
# ---------------------------------------------------------------------------------------
# `steps` est un compteur ABSOLU : v3 va jusqu'à 220 000 en repartant des 120 000 de v2,
# soit 100 000 pas supplémentaires sur le jeu filtré en pose.
step "04_train_v2" $PY -m facedit.train --config configs/final_v2.yaml \
     --set data.num_workers=4

step "05_train_v3" $PY -m facedit.train --config configs/final_v3.yaml \
     --resume runs/final_v2/ckpt_final.pt --set data.num_workers=4

# ---------------------------------------------------------------------------------------
# 4. Vérification et export
# ---------------------------------------------------------------------------------------
step "06_benchmark" $PY -m facedit.benchmark --config configs/final_v3.yaml

step "07_export" $PY scripts/export_weights.py runs/final_v3/ckpt_final.pt \
     poids/facedit_v3.pt

echo ""
echo "======================================================================"
echo "Pipeline terminé."
echo "Évaluation :  $PY -m facedit.eval.run_eval --ckpt runs/final_v3/ckpt_final.pt \\"
echo "                   --config configs/eval_fast.yaml --label facedit-dit-v3"
echo "Interface  :  $PY app.py"
echo "======================================================================"
