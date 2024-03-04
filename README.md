PIE MSXS-08-2023/2024
==============================

ISAE-SUPAERO Group project for Prof en Poche

Project Organization

==============================

Les commandes de base facile à lancé sont visible avec `make help`

==============================

# Configuration minimum :

-   Make
-   python3 et python3-venv
-   GPU qui support cuda (testé la command `nvidia-smi`)

L'instalation a seulement été testé sur des Linux

# Installation

cette outil utilise Makefile, pip et l'outils d'environement virtuelle de python3.

pour installé les dépendance sur linux, run `Make setup`

une fois les dépendance installé, l'interface peut etre lancé avec `Make startWebUI`

==============================

# Modèle et base de donnée

Les chemin de base et les list de modèle peuvent etre modifier dans le fichier `src/app/utils/constants.py`

------------

    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── Makefile           <- Makefile with default command to run
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        └── testing        <- Code use to test different models
            ├── test.py
            ├── auto-tester.py
            ├── exemple.txt
            └── README.md <- information on how to use theses scripts

--------
