a faire tourné avec le venv `source ../venv/bin/activate`

!! Ces scripts ne sont plus a jour. aucune guaranty sur leur fonctionnement n'est fournis!!
# test_your_prompts.py

Script pour faire tourner un LLM sur une série de prompts. 

Le LLM sera quantifié à la précision indiquée lors de l'appel du script. Les prompts seront lues depuis .csv dans le dossier `./prompts`.

La commande pour appeler ce script depuis ce dossier :

`python ./generate.py model_name precision input_prompts_file_name`

Il faut écrire le nom d'un modèle valide puis selectionner la précision souhaitée pour la quantification du modèle :
- '4' = 4 bits;
- '8' = 8 bits;
- '16' = 16 bits (half-precision);
- '32' = 32 bits (full-precision).

Les prompts sont à écrire dans un fichier csv placé dans le dossier `./prompts`.

Les prompts seront traitées indépendemment les unes des autres.

Les réponses générées seront écrites dans un fichier csv qui reprendra le nom du fichier lu en entrée et sera placé dans le dossier `./generated_answers`.

# playground.py

Script permettant de tester simplement un modèle

`python ./playground.py`

Le scipt vous guidera pour choisir un modèle, une précision et vous proposera d'écrire votre prompt.
