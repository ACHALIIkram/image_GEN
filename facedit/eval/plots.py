"""Figures du rapport, générées automatiquement depuis `results.json`.

Critère d'acceptation §9 : « La matrice de fuite et la heatmap de sous-groupes sont
générées automatiquement. » Ce module est donc appelé en fin de `run_eval`, et peut
aussi être rejoué seul sur un `results.json` existant :

    python -m facedit.eval.plots --results results/results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # aucun affichage interactif : le harnais tourne sans écran
import matplotlib.pyplot as plt
import numpy as np

FIG_DPI = 150


def _annotate(ax, image, matrix: np.ndarray, fmt: str = "{:.2f}") -> None:
    """Écrit la valeur dans chaque case, en noir ou blanc selon le fond réel.

    La couleur du texte est déduite de la **luminance de la couleur effectivement
    tracée**, interrogée au mappable matplotlib, et non d'une règle sur la valeur.
    Une règle sur la valeur ne peut pas être correcte pour deux colormaps opposées :
    `magma` va du sombre au clair (valeur haute → fond clair → texte noir), tandis que
    `RdYlGn` est claire au centre et saturée aux deux extrémités (valeur moyenne → fond
    clair). Une seule des deux serait lisible.
    """
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isfinite(value):
                ax.text(j, i, "—", ha="center", va="center", color="grey", fontsize=9)
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            # Luminance relative ITU-R BT.709.
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                j, i, fmt.format(value), ha="center", va="center",
                color="black" if luminance > 0.55 else "white", fontsize=9,
            )


# --------------------------------------------------------------------------------------
# F-E4 — matrice de fuite
# --------------------------------------------------------------------------------------


def plot_leakage(leakage: Dict, out_path: Path) -> Path:
    labels = {"age": "âge", "gender": "genre", "ita": "teint (ITA)"}
    names = [labels[a] for a in leakage["attributes"]]
    matrix = np.asarray(leakage["matrix_normalized"], dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    image = ax.imshow(matrix, cmap="magma", vmin=0.0)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("attribut mesuré")
    ax.set_ylabel("attribut piloté")
    ax.set_title(
        f"Matrice de fuite (F-E4)\nratio de désenchevêtrement = "
        f"{leakage['summary']['disentanglement_ratio']:.2f}",
        fontsize=11,
    )
    _annotate(ax, image, matrix)
    fig.colorbar(image, ax=ax, label="déplacement (σ du set réel)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# F-E7 — grille de sous-groupes
# --------------------------------------------------------------------------------------


def plot_subgroups(subgroups: Dict, out_dir: Path) -> List[Path]:
    """Une heatmap par métrique. Lignes = (tranche d'âge × genre), colonnes = bin de teint."""
    cells = subgroups["cells"]
    age_bins = sorted({c["age_bin"] for c in cells}, key=lambda s: float(s.split("-")[0]))
    ita_bins = sorted({c["ita_bin"] for c in cells}, key=lambda s: float(s.split("..")[0]))
    genders = ["Male", "Female"]

    row_labels = [f"{age} · {'F' if g == 'Female' else 'H'}" for age in age_bins for g in genders]
    lookup = {(c["age_bin"], c["gender"], c["ita_bin"]): c for c in cells}

    panels = [
        ("gender_accuracy", "Précision de genre", "RdYlGn", "{:.2f}", None, None),
        ("age_mae", "MAE d'âge (années)", "RdYlGn_r", "{:.1f}", 0.0, None),
        ("ita_delta_mean", "ΔITA moyen (degrés)", "RdYlGn_r", "{:.1f}", 0.0, None),
        ("fid", "FID par cellule", "RdYlGn_r", "{:.0f}", None, None),
    ]

    written: List[Path] = []
    for key, title, cmap, fmt, vmin, vmax in panels:
        matrix = np.full((len(row_labels), len(ita_bins)), np.nan)
        for row, age in enumerate(age_bins):
            for gender_offset, gender in enumerate(genders):
                for col, ita in enumerate(ita_bins):
                    cell = lookup.get((age, gender, ita))
                    if cell and cell.get(key) is not None:
                        matrix[row * 2 + gender_offset, col] = cell[key]
        if not np.isfinite(matrix).any():
            continue

        fig, ax = plt.subplots(figsize=(1.5 * len(ita_bins) + 3.2, 0.5 * len(row_labels) + 2.2))
        image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ita_bins)), [f"ITA\n{b}" for b in ita_bins], fontsize=8)
        ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
        ax.set_title(f"{title} par sous-groupe (F-E7)", fontsize=11)
        _annotate(ax, image, matrix, fmt)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        path = out_dir / f"subgroups_{key}.png"
        fig.savefig(path, dpi=FIG_DPI)
        plt.close(fig)
        written.append(path)
    return written


# --------------------------------------------------------------------------------------
# F-E6 — balayage de w
# --------------------------------------------------------------------------------------


def plot_w_sweep(sweep: Dict, out_path: Path) -> Path:
    """FID et fidélité sur le **même** graphique (F-E6), deux axes verticaux.

    L'intérêt de superposer les deux courbes est de rendre visible l'arbitrage : le CFG
    améliore l'obéissance aux attributs et dégrade le réalisme et la diversité. La valeur
    de `w` à retenir (question ouverte n°4) est le coude, pas l'optimum d'une des deux.
    """
    points = sweep["points"]
    w_values = [p["w"] for p in points]

    fig, ax_left = plt.subplots(figsize=(7.2, 4.6))
    ax_right = ax_left.twinx()

    fids = [p.get("fid") for p in points]
    if any(f is not None for f in fids):
        ax_left.plot(
            w_values, [np.nan if f is None else f for f in fids],
            "o-", color="#1f77b4", label="FID (↓ meilleur)",
        )
    ax_left.set_xlabel("échelle de guidage w")
    ax_left.set_ylabel("FID", color="#1f77b4")
    ax_left.tick_params(axis="y", labelcolor="#1f77b4")

    ax_right.plot(
        w_values, [p["age_mae"] for p in points],
        "s--", color="#d62728", label="MAE âge (↓ meilleur)",
    )
    ax_right.plot(
        w_values, [p["ita_delta_mean"] for p in points],
        "^--", color="#ff7f0e", label="ΔITA (↓ meilleur)",
    )
    ax_right.plot(
        w_values, [100.0 * (1.0 - p["gender_accuracy"]) for p in points],
        "v--", color="#9467bd", label="erreur de genre % (↓ meilleur)",
    )
    ax_right.set_ylabel("erreur d'attribut", color="#333333")

    if sweep.get("recommended_w") is not None:
        ax_left.axvline(
            sweep["recommended_w"], color="grey", linestyle=":",
            label=f"w retenu = {sweep['recommended_w']:g}",
        )

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(handles_left + handles_right, labels_left + labels_right, fontsize=8, loc="best")
    ax_left.set_title("Réalisme contre obéissance : balayage de w (F-E6)", fontsize=11)
    ax_left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Figure principale du rapport : demandé contre mesuré
# --------------------------------------------------------------------------------------


def plot_requested_vs_measured(scatter: Dict, out_path: Path) -> Path:
    """La figure qui matérialise la proposition de valeur du §1.2 : l'écart demandé/mesuré.

    La droite y = x est l'obéissance parfaite. Une régression de pente < 1 est la
    signature exacte du problème décrit au §1.1 — le modèle régresse vers la moyenne de
    sa distribution d'entraînement.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    for ax, key, label, unit in (
        (axes[0], "age", "âge", "ans"),
        (axes[1], "ita", "teint (ITA)", "°"),
    ):
        requested = np.asarray(scatter[f"{key}_requested"], dtype=float)
        measured = np.asarray(scatter[f"{key}_measured"], dtype=float)
        mask = np.isfinite(requested) & np.isfinite(measured)
        requested, measured = requested[mask], measured[mask]
        if requested.size < 2:
            continue

        ax.scatter(requested, measured, s=6, alpha=0.3, color="#1f77b4", edgecolors="none")
        limits = [min(requested.min(), measured.min()), max(requested.max(), measured.max())]
        ax.plot(limits, limits, "k--", linewidth=1, label="obéissance parfaite")

        slope, intercept = np.polyfit(requested, measured, 1)
        grid = np.linspace(requested.min(), requested.max(), 50)
        ax.plot(
            grid, slope * grid + intercept, "-", color="#d62728", linewidth=1.6,
            label=f"régression (pente {slope:.2f})",
        )
        ax.set_xlabel(f"{label} demandé ({unit})")
        ax.set_ylabel(f"{label} mesuré ({unit})")
        ax.set_title(f"{label.capitalize()} : demandé contre mesuré", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def render_all(results: Dict, out_dir: str | Path) -> List[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if results.get("leakage"):
        written.append(plot_leakage(results["leakage"], out_dir / "leakage_matrix.png"))
    if results.get("subgroups"):
        written.extend(plot_subgroups(results["subgroups"], out_dir))
    if results.get("w_sweep"):
        written.append(plot_w_sweep(results["w_sweep"], out_dir / "w_sweep.png"))
    if results.get("scatter"):
        written.append(
            plot_requested_vs_measured(results["scatter"], out_dir / "requested_vs_measured.png")
        )

    return [str(p) for p in written]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Figures depuis un results.json")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    path = Path(args.results)
    results = json.loads(path.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else path.parent / "figures"
    for figure in render_all(results, out_dir):
        print(f"→ {figure}")


if __name__ == "__main__":
    main()
