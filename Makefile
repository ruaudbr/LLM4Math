VENV_PATH = ~/dataSSD/venv/bin/activate
PYTHON_PATH = ~/dataSSD/venv/bin/python3

.PHONY: help
help: ## Liste toutes les commandes
	@echo "Available targets:"
	@awk '/^^([a-zA-Z0-9_-]+):[ \t]*##[ \t]+(.+)/' $(MAKEFILE_LIST) | column -t -s ':'

.PHONY: startWebAI

startWebAI: ## Démarre une interface web 
	@$(PYTHON_PATH) ./src/app/app.py

.PHONY: GPUutilisation

GPUutilisation: ## Affiche l'utilisation de la carte graphique 
	@nvidia-smi

