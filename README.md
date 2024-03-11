PIE MSXS-08-2023/2024
==============================

Organisation du projet

==============================

Les commandes de base faciles à lancer sont visibles avec `make help`.

==============================

# Configuration minimale :

-   Make
-   python3 et python3-venv
-   GPU qui supporte cuda (par exemple, tester la command `nvidia-smi`)

L'installation a uniquement été testée sur Linux.

# Installation

Cet outil utilise Makefile, pip et l'outil d'environnement virtuel de python3.

Pour installer les dépendances sur linux, lancer `make setup`.

Une fois les dépendances installées, l'interface peut être lancée avec `make startWebUI`.

==============================

# Modèle et base de donnée

Les chemins de base et les listes de modèles peuvent être modifiés dans le fichier `src/app/utils/constants.py`.

## Modèle hugging face

Les modèles utilisant la librairie Hugging-face/transform sont automatiquement téléchargés quand vous voulez les charger. 
Selectionner le modèle dans la liste de l'interface graphique et cliquer sur `load`.

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

Pour creer des base de donnée, ce référé au [README.md](./src/RAG/README.md) présent dans le dossier `src/RAG`.

------------ 
Arboréssence de la solution

    ├── README.md          <- The top-level README for developers using this project.
    |
    ├── Makefile           <- Makefile with default command to run
    │
    ├── requirements.txt   <- The requirements file for reproducing the 
    |                         analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── app.py         <- Code de l'interface Gradio
        ├── utils          <- dossier contenant le code derrière l'interface
        |   ├── utils.py   <- fonction de traitement les inputs
        |   ├── constants.py <- fichier contenant les chemin et les constants
        |   └── RAG_utils.py <- fonction suplémentaire pour RAG
        ├── RAG            <- Dossier contenant des script pour géré des base de 
        |                     donnée
        └── legacy_scripts <- Dossier contenant d'ancien script de génération
                              ! leur fonctionnement n'est pas guaranti !

--------
