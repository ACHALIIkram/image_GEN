"""FaceDiT — génération de visages photo-réalistes avec contrôle d'attributs vérifiable.

Le seul réseau génératif entraîné par l'équipe est le DiT (`facedit.models.dit`).
Le VAE, l'échantillonneur de bruit et les modèles de restauration sont importés gelés.
"""

import sys

__version__ = "1.0.0"


def _force_utf8_console() -> None:
    """Bascule stdout/stderr en UTF-8.

    La machine cible est sous Windows (§1.3) et la console y utilise cp1252 par défaut.
    Tous les messages du projet sont en français et contiennent des accents, des degrés
    et des flèches ; sans cette bascule, un simple `print` de fin d'encodage lève une
    `UnicodeEncodeError` et fait échouer un script qui avait pourtant fini son travail.

    `errors="replace"` plutôt que `"strict"` : un caractère non représentable doit
    dégrader l'affichage, jamais interrompre un entraînement de plusieurs heures.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue  # flux redirigé vers un objet sans encodage propre
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


_force_utf8_console()
