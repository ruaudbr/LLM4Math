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

Les chemin de base et les listes de modèle peuvent etre modifier dans le fichier `src/app/utils/constants.py`

## Modèle hugging face

Les modèles utilisant la librairie Hugging-face/transform sont automatiquement téléchargé quand vous voulé les chargés. 
Selectionné le modèle dans la liste de l'interface graphic et cliqué sur `load`

Pour ajouté un modèle dans la liste, modifier le dictionère `MODEL_ID` dans `src/utils/constants`.
le format est ` "name_of_model" : "path_to_model" `.

Les modèle seront téléchargé dans le dossier `../models/hf_models/` (peut etre modifier en modifiant la constante `DEFAULT_HF_CACHE`)

## Modèle GGUF

Les modèle gguf ne peuvent pas être téléchargé automatiquement par l'outil. vous devez les téléchargé manuellement et les placé dans le dossier `../models/gguf_models/` (peut etre modifier en modifiant la constante `DEFAULT_GGUF_CACHE`)

Une fois téléchargé, relancé l'outil pour voir le modèle apparaitre dans la liste des modèle disponible en gguf.

## modèle RAG

Pour utilisé RAG, l'outils utilise la bibliothèque Ollama. pour rajouté un modèle, rajouté le nom du modèle dans la liste `OLLAMA_MODEL`

Les base de donnée RAG doivent se trouvé dans le dossier `../vdb/` (modifiable en modifiant la constant `RAG_FOLDER_PATH`) dans un dossier specifique a cette base de donnée.

Rajouté ensuite un et le nom du dossier dans le dictionaire `RAG_DATABASE` (le nom seras utilsé pour l'affichage dans l'interface graphique)

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
