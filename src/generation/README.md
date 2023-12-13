## generate.py

Script pour faire tourner un LLM quantifié sur une série de prompts et récupérer ses réponses.

`python ./generate.py model_name precision`

Il faut écrire le nom d'un modèle valide
puis selectionner la précision souhaitée:
- '4' = 4 bits;
- '8' = 8 bits;
- '16' = 16 bits (half-precision);
- '32' = 32 bits (full-precision).

Les prompts utilisées sont écrites dans le fichier `constants.py`.

Les prompts sont traitées indépendemment les unes des autres.

\! Ce n'est pas un chat \!

Les réponses seront stockées dans un .json avec les prompts associées.

## playground.py

Traduction du notebook intial en script python.

Il permet de tester un modèle.

`python ./playground.py`

Le scipt vous guidera pour choisir un modèle, une précision et vous proposera d'écrire votre prompt.
