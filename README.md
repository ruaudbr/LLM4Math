# PIE MSXS-08 2023/2024

## Arborescence de la solution

    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── Makefile           <- Makefile with default command to run.
    │
    ├── requirements.txt   <- The requirements file for reproducing the 
    |                         analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── app.py         <- Code de l'interface Gradio.
        ├── utils          <- Dossier contenant le code derrière l'interface.
        |   ├── utils.py   <- Fonction de traitement les inputs.
        |   ├── constants.py <- Fichier contenant les chemins et les constantes.
        |   └── RAG_utils.py <- Fonction suplémentaire pour RAG.
        ├── RAG            <- Dossier contenant des scripts pour gérer des bases de 
        |                     donnée.
        └── legacy_scripts <- Dossier contenant d'anciens scripts de génération
                              ! leur fonctionnement n'est pas garanti !

## Configuration minimale

-   make;
-   python3 et python3-venv;
-   GPU qui supporte cuda (par exemple, tester la command `nvidia-smi`).

L'installation a uniquement été testée sur Linux.

## Installation

Cet outil utilise Makefile, pip et l'outil d'environnement virtuel de python3.

Pour installer les dépendances sur linux, lancer `make setup`.

Une fois les dépendances installées, l'interface peut être lancée avec `make startWebUI`.


## Modèle et base de donnée

Les chemins de base et les listes de modèles peuvent être modifiés dans le fichier `src/app/utils/constants.py`.

### Modèle hugging face

Les modèles utilisant la librairie Hugging-face/transform sont automatiquement téléchargés quand vous voulez les charger. 
Selectionner le modèle dans la liste de l'interface graphique et cliquer sur `load`.

Pour ajouter un modèle dans la liste, modifier le dictionnaire `MODEL_ID` dans `src/utils/constants`.
Le format est ` "name_of_model" : "path_to_model" `.

Les modèles seront téléchargés dans le dossier `../models/hf_models/` (peut être modifiés en modifiant la constante `DEFAULT_HF_CACHE`).

### Modèle GGUF

Les modèles gguf ne peuvent pas être téléchargés automatiquement par l'outil. Vous devez les télécharger manuellement et les placer dans le dossier `../models/gguf_models/` (peut être modifié en modifiant la constante `DEFAULT_GGUF_CACHE`).

Une fois téléchargé, relancer l'outil pour voir le modèle apparaître dans la liste des modèles disponibles en gguf.

### Modèle RAG

Pour utiliser RAG, l'outil utilise la bibliothèque Ollama. pour rajouter un modèle, rajouter le nom du modèle dans la liste `OLLAMA_MODEL`.

Les bases de donnée RAG doivent se trouver dans le dossier `../vdb/` (modifiable en modifiant la constant `RAG_FOLDER_PATH`) dans un dossier specifique à cette base de donnée.

Rajouter ensuite un et le nom du dossier dans le dictionnaire `RAG_DATABASE` (le nom sera utilisé pour l'affichage dans l'interface graphique).

Pour créer des bases de donnée, se référer au [README.md](./src/RAG/README.md) présent dans le dossier `src/RAG`.

