"""Entraînement de l'oracle — F-O2, F-O4, F-O5.

    python -m facedit.oracle.train_oracle --config configs/final.yaml

Produit `runs/oracle/oracle_best.pt` et `runs/oracle/oracle_metrics.json`. Ce second
fichier est la **barre d'erreur** citée par F-O4 : toute métrique d'attribut mesurée
plus tard sur des images générées doit être lue en regard de ces valeurs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from facedit.data.fairface import RACE_ORDER, open_fairface, resize_uint8
from facedit.data.ita import compute_ita
from facedit.oracle.model import AttributeOracle
from facedit.utils.config import Config, load_config
from facedit.utils.repro import seed_everything


# --------------------------------------------------------------------------------------
# Labels d'attributs (l'ITA analytique est coûteux : on le calcule une fois et on le cache)
# --------------------------------------------------------------------------------------


def build_oracle_labels(
    cfg: Config,
    split: str,
    use_mediapipe: Optional[bool] = None,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Calcule (ou relit) les labels d'attributs alignés sur `source.records`.

    L'ITA analytique passe par MediaPipe et coûte quelques dizaines de minutes sur 75 k
    images. Le résultat est mis en cache : il ne dépend que du jeu source, du filtre
    d'âge et de la résolution, pas des hyperparamètres d'entraînement.

    `limit` restreint le calcul aux `limit` premiers enregistrements. Destiné à la
    semaine 1, pour vérifier que la chaîne complète tourne sans attendre une heure de
    calcul d'ITA. Le cache correspondant porte un nom distinct : un oracle entraîné sur
    un sous-ensemble ne doit jamais être confondu avec l'oracle de référence, puisque
    c'est lui qui fixe la barre d'erreur de tout le protocole (F-O4).
    """
    if use_mediapipe is None:
        use_mediapipe = cfg.data.ita_use_mediapipe

    size = cfg.oracle.image_size
    tag = f"{split}_{size}_{'mp' if use_mediapipe else 'geo'}"
    tag += "" if cfg.data.drop_partial_age_bins else "_all"
    if limit is not None:
        tag += f"_limit{limit}"
    cache_path = Path(cfg.data.cache_dir) / f"oracle_labels_{tag}.npz"

    source = open_fairface(cfg.data, split)
    count = len(source) if limit is None else min(limit, len(source))

    if cache_path.exists() and not overwrite:
        cached = np.load(cache_path, allow_pickle=True)
        if int(cached["n_records"]) == count:
            print(f"[oracle] labels relus depuis {cache_path} ({count} images)")
            return {k: cached[k] for k in ("gender", "race", "age_low", "age_high", "ita", "valid")}
        print(f"[oracle] cache {cache_path} périmé ({cached['n_records']} ≠ {count})")
    gender = np.zeros(count, dtype=np.int64)
    race = np.zeros(count, dtype=np.int64)
    age_low = np.zeros(count, dtype=np.float32)
    age_high = np.zeros(count, dtype=np.float32)
    ita = np.full(count, np.nan, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)

    for index in tqdm(range(count), desc=f"ITA/{split}", unit="img"):
        record = source.records[index]
        gender[index] = record.gender
        race[index] = record.race
        age_low[index] = record.age_low
        age_high[index] = record.age_high
        image = resize_uint8(source.load_image(index), size)
        result = compute_ita(
            image, use_mediapipe=use_mediapipe, min_pixels=cfg.data.ita_min_pixels
        )
        ita[index] = result.ita
        valid[index] = result.valid

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        n_records=count,
        origin=source.origin,
        gender=gender,
        race=race,
        age_low=age_low,
        age_high=age_high,
        ita=ita,
        valid=valid,
    )
    print(f"[oracle] labels écrits → {cache_path} · ITA valide sur {valid.mean() * 100:.1f} %")
    return {"gender": gender, "race": race, "age_low": age_low, "age_high": age_high,
            "ita": ita, "valid": valid}


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------


class OracleDataset(torch.utils.data.Dataset):
    """Images réelles + trois cibles.

    **Aucune augmentation colorimétrique.** Le flip horizontal est la seule transformation
    appliquée. Un jitter de luminosité, de contraste ou de teinte — pourtant le réflexe
    standard pour un classifieur — modifierait la couleur de peau sans modifier la cible
    ITA, et apprendrait donc explicitement à l'oracle à ignorer le signal qu'on lui
    demande de mesurer. C'est le piège central de cet entraînement.
    """

    def __init__(
        self,
        source,
        labels: Dict[str, np.ndarray],
        image_size: int,
        train: bool,
        oracle: AttributeOracle,
    ):
        self.source = source
        self.labels = labels
        self.image_size = image_size
        self.train = train
        self.oracle = oracle
        # Seules les images dont l'ITA est mesurable entrent dans l'entraînement : une
        # cible `nan` propagerait dans la loss.
        self.indices = np.flatnonzero(labels["valid"])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = int(self.indices[position])
        image = resize_uint8(self.source.load_image(index), self.image_size)

        if self.train and torch.rand(()).item() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])

        pixels = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.0
        pixels = (pixels - self.oracle.pixel_mean[0].cpu()) / self.oracle.pixel_std[0].cpu()

        low, high = self.labels["age_low"][index], self.labels["age_high"][index]
        # Âge continu tiré dans la tranche (F-D3), exactement comme pour le générateur :
        # l'oracle doit apprendre la même notion d'âge que celle sur laquelle le DiT est
        # conditionné, sinon la MAE mesurée mélangerait deux définitions.
        age = float(low) + (float(high) - float(low)) * torch.rand(()).item()

        return {
            "pixels": pixels,
            "gender": torch.tensor(int(self.labels["gender"][index]), dtype=torch.long),
            "age": torch.tensor(age, dtype=torch.float32),
            "ita": torch.tensor(float(self.labels["ita"][index]), dtype=torch.float32),
            "race": torch.tensor(int(self.labels["race"][index]), dtype=torch.long),
        }


# --------------------------------------------------------------------------------------
# Évaluation
# --------------------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: AttributeOracle, loader, device: torch.device) -> Dict[str, object]:
    """Métriques de validation, globales puis ventilées (F-O4, E-2)."""
    model.eval()
    chunks = {k: [] for k in ("gender_true", "gender_pred", "gender_conf",
                              "age_true", "age_pred", "ita_true", "ita_pred", "race")}

    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=True)
        out = model(pixels)
        age_pred, ita_pred = model.denormalize(out["age_norm"].float(), out["ita_norm"].float())
        probabilities = torch.softmax(out["gender_logits"].float(), dim=-1)
        confidence, predicted = probabilities.max(dim=-1)

        chunks["gender_true"].append(batch["gender"].numpy())
        chunks["gender_pred"].append(predicted.cpu().numpy())
        chunks["gender_conf"].append(confidence.cpu().numpy())
        chunks["age_true"].append(batch["age"].numpy())
        chunks["age_pred"].append(age_pred.cpu().numpy())
        chunks["ita_true"].append(batch["ita"].numpy())
        chunks["ita_pred"].append(ita_pred.cpu().numpy())
        chunks["race"].append(batch["race"].numpy())

    data = {k: np.concatenate(v) for k, v in chunks.items()}
    metrics: Dict[str, object] = {
        "n": int(len(data["gender_true"])),
        "gender_accuracy": float((data["gender_true"] == data["gender_pred"]).mean()),
        "gender_mean_confidence": float(data["gender_conf"].mean()),
        "age_mae": float(np.abs(data["age_true"] - data["age_pred"]).mean()),
        "age_bias": float((data["age_pred"] - data["age_true"]).mean()),
        "ita_mae": float(np.abs(data["ita_true"] - data["ita_pred"]).mean()),
        "ita_bias": float((data["ita_pred"] - data["ita_true"]).mean()),
        # F-O5 : la tête ITA est une vérification croisée du calcul analytique.
        # Une corrélation < 0.85 signifierait que l'ITA n'est pas apprenable depuis
        # l'image, donc que le calcul analytique est dominé par le bruit d'éclairage.
        "ita_pearson_r": float(np.corrcoef(data["ita_true"], data["ita_pred"])[0, 1]),
        "age_pearson_r": float(np.corrcoef(data["age_true"], data["age_pred"])[0, 1]),
    }

    # Ventilation par groupe perçu : E-2 exige que le biais de l'instrument de mesure
    # soit publié, pas seulement celui du générateur.
    by_race = {}
    for race_index, race_name in enumerate(RACE_ORDER):
        mask = data["race"] == race_index
        if mask.sum() < 20:
            continue
        by_race[race_name] = {
            "n": int(mask.sum()),
            "gender_accuracy": float(
                (data["gender_true"][mask] == data["gender_pred"][mask]).mean()
            ),
            "age_mae": float(np.abs(data["age_true"][mask] - data["age_pred"][mask]).mean()),
            "ita_mae": float(np.abs(data["ita_true"][mask] - data["ita_pred"][mask]).mean()),
        }
    metrics["by_perceived_race"] = by_race

    by_gender = {}
    for gender_index, gender_name in enumerate(("Male", "Female")):
        mask = data["gender_true"] == gender_index
        if mask.sum() < 20:
            continue
        by_gender[gender_name] = {
            "n": int(mask.sum()),
            "gender_accuracy": float(
                (data["gender_true"][mask] == data["gender_pred"][mask]).mean()
            ),
            "age_mae": float(np.abs(data["age_true"][mask] - data["age_pred"][mask]).mean()),
            "ita_mae": float(np.abs(data["ita_true"][mask] - data["ita_pred"][mask]).mean()),
        }
    metrics["by_gender"] = by_gender

    model.train()
    return metrics


# --------------------------------------------------------------------------------------
# Entraînement
# --------------------------------------------------------------------------------------


def train_oracle(
    cfg: Config, overwrite_labels: bool = False, limit: Optional[int] = None
) -> Path:
    from torch.utils.tensorboard import SummaryWriter

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.oracle.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = AttributeOracle(
        pretrained=cfg.oracle.pretrained,
        image_size=cfg.oracle.image_size,
        age_mean=cfg.oracle.age_mean,
        age_std=cfg.oracle.age_std,
        ita_mean=cfg.oracle.ita_mean,
        ita_std=cfg.oracle.ita_std,
    ).to(device)

    loaders = {}
    for split in ("train", "val"):
        source = open_fairface(cfg.data, split)
        split_limit = None if limit is None else max(64, limit // (1 if split == "train" else 4))
        labels = build_oracle_labels(
            cfg, split, overwrite=overwrite_labels, limit=split_limit
        )
        dataset = OracleDataset(
            source, labels, cfg.oracle.image_size, train=(split == "train"), oracle=model
        )
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.oracle.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.oracle.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            persistent_workers=cfg.oracle.num_workers > 0,
        )
        print(f"[oracle] {split}: {len(dataset)} images utilisables")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.oracle.lr, weight_decay=cfg.oracle.weight_decay
    )
    steps_per_epoch = len(loaders["train"])
    lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.oracle.lr, total_steps=cfg.oracle.epochs * steps_per_epoch,
        pct_start=0.15,
    )
    use_amp = device.type == "cuda"
    writer = SummaryWriter(out_dir / "tb")

    best_score = float("inf")
    best_path = out_dir / "oracle_best.pt"
    history = []
    started = time.time()

    for epoch in range(cfg.oracle.epochs):
        model.train()
        running = {"loss": 0.0, "gender": 0.0, "age": 0.0, "ita": 0.0, "n": 0}
        progress = tqdm(loaders["train"], desc=f"oracle {epoch + 1}/{cfg.oracle.epochs}")

        for batch in progress:
            pixels = batch["pixels"].to(device, non_blocking=True)
            gender = batch["gender"].to(device, non_blocking=True)
            age_target, ita_target = model.normalize_targets(
                batch["age"].to(device), batch["ita"].to(device)
            )

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(pixels)
                loss_gender = F.cross_entropy(
                    out["gender_logits"], gender, label_smoothing=cfg.oracle.label_smoothing
                )
                # Huber plutôt que MSE sur les deux régressions : les tranches d'âge de
                # FairFace produisent des cibles bruitées par construction (± 5 ans) et
                # l'ITA porte des valeurs aberrantes d'éclairage. Une MSE laisserait ces
                # queues dominer le gradient.
                delta = cfg.oracle.huber_delta / cfg.oracle.age_std
                loss_age = F.huber_loss(out["age_norm"].float(), age_target, delta=delta)
                loss_ita = F.huber_loss(out["ita_norm"].float(), ita_target, delta=delta)
                loss = (
                    cfg.oracle.gender_loss_weight * loss_gender
                    + cfg.oracle.age_loss_weight * loss_age
                    + cfg.oracle.ita_loss_weight * loss_ita
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_schedule.step()

            size = pixels.shape[0]
            running["loss"] += loss.item() * size
            running["gender"] += loss_gender.item() * size
            running["age"] += loss_age.item() * size
            running["ita"] += loss_ita.item() * size
            running["n"] += size
            progress.set_postfix(loss=f"{running['loss'] / running['n']:.4f}")

        metrics = evaluate(model, loaders["val"], device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running["loss"] / running["n"]
        history.append(metrics)

        for key in ("gender_accuracy", "age_mae", "ita_mae", "ita_pearson_r"):
            writer.add_scalar(f"val/{key}", metrics[key], epoch + 1)
        writer.add_scalar("train/loss", metrics["train_loss"], epoch + 1)

        print(
            f"[oracle] epoch {epoch + 1}: genre {metrics['gender_accuracy'] * 100:.2f} % · "
            f"MAE âge {metrics['age_mae']:.2f} ans · MAE ITA {metrics['ita_mae']:.2f}° · "
            f"r(ITA) {metrics['ita_pearson_r']:.3f}"
        )

        # Critère de sélection composite, en unités comparables : une erreur de genre
        # d'un point de pourcentage, un an d'âge et un degré d'ITA ne pèsent pas
        # naturellement pareil. Les échelles retenues (1 % ≈ 0.1 an ≈ 0.2°) reflètent
        # l'usage : l'interface affiche l'âge en années et l'ITA en degrés.
        score = (
            (1.0 - metrics["gender_accuracy"]) * 100.0
            + metrics["age_mae"] / 0.1 * 0.01
            + metrics["ita_mae"] / 0.2 * 0.01
        )
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "hyper": {
                        "image_size": cfg.oracle.image_size,
                        "age_mean": cfg.oracle.age_mean,
                        "age_std": cfg.oracle.age_std,
                        "ita_mean": cfg.oracle.ita_mean,
                        "ita_std": cfg.oracle.ita_std,
                    },
                    "metrics": metrics,
                    "epoch": epoch + 1,
                    "config_hash": cfg.hash(),
                },
                best_path,
            )
            print(f"[oracle] nouveau meilleur → {best_path}")

    writer.close()
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    report = {
        "role": "F-O4 — barre d'erreur de toutes les mesures d'attributs ultérieures",
        "subset_limit": limit,
        "is_reference_oracle": limit is None,
        "best_epoch": best["epoch"],
        "metrics": best["metrics"],
        "history": history,
        "wall_clock_minutes": round((time.time() - started) / 60, 1),
        "config_hash": cfg.hash(),
        "interpretation": (
            "Une MAE d'âge de X ans mesurée sur des images générées doit être comparée à "
            f"{best['metrics']['age_mae']:.2f} ans, l'erreur de l'oracle sur des visages "
            "réels. L'écart imputable au générateur est ce qui dépasse cette valeur. "
            "Même raisonnement pour l'ITA et la précision de genre."
        ),
    }
    (out_dir / "oracle_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[oracle] rapport → {out_dir / 'oracle_metrics.json'}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraînement de l'oracle 3 têtes (F-O2)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument("--labels-only", action="store_true",
                        help="Calcule et cache les labels ITA, sans entraîner")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Restreint l'entraînement aux N premières images. Sert à valider la chaîne "
             "en semaine 1 ; l'oracle produit N'EST PAS l'oracle de référence.",
    )
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.labels_only:
        for split in ("train", "val"):
            build_oracle_labels(cfg, split, overwrite=args.overwrite_labels, limit=args.limit)
        return
    if args.limit is not None:
        print(
            f"[oracle] MODE SOUS-ENSEMBLE ({args.limit} images) — l'oracle produit sert à "
            f"valider la chaîne, pas à fixer la barre d'erreur du rapport (F-O4)."
        )
    train_oracle(cfg, args.overwrite_labels, args.limit)


if __name__ == "__main__":
    main()
