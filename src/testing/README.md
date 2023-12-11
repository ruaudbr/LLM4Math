## test.py

ce script permet de lancé un model et de lui envoye des prompts

`python ./test.py`

il faut ecrire le nom d'un model valide
puis selectionné la précision souhaité:
- '1' = 4 bits
- '2' = 8 bits
- '3' = 16 bits (half-precision)
- '4' = 32 bits (full-precision)

il est ensuite possible d'envoyer des prompt au model.

chaque prompt est indémendant et le model n'as pas de memoire

\! ce n'est pas un chat \!

## auto_tester.py

Ce script permet de tester automatiquement une liste de prompt sur un model donnée

usage : `python ./auto_tester.py [model_name] [precision] [input_file] [output_file]`

Le `model_name` est les nom de model comme dans `test.py`

La `precision` est la meme que dans `test.py`

`input_file` est un fichier text dans le quelle, une ligne represente un prompt. un example de fichier peut etre trouver [ici](./prompt.txt)