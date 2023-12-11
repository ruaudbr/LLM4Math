## generate.py

Outil pour générer les réponses de modèles à notre base de prompts.

`python ./generate.py`

Il faut écrire le nom d'un modèle valide
puis selectionner la précision souhaitée:
- '1' = 4 bits;
- '2' = 8 bits;
- '3' = 16 bits (half-precision);
- '4' = 32 bits (full-precision).

Il est ensuite possible d'envoyer des prompts au modèle.

Chaque prompt est indépendant et le modèle n'as pas de mémoire.

\! Ce n'est pas un chat \!

## playground.py

Traduction du notebook intial en script python.

Il permet de tester la génération de prompt sur tel ou tel modèle.

`python ./auto_tester.py [model_name] [precision] [input_file] [output_file]`

La `precision` est définie comme précédemment.

`input_file` est un fichier text dans lequel une ligne représente un prompt. Un example de fichier peut être trouvé [ici](./prompt.txt)