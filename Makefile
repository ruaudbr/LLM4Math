VENV_PATH = ~/dataSSD/venv/bin/activate
PYTHON_PATH = ~/dataSSD/venv/bin/python3

.PHONY: help
help: ## list toutes les commandes
	@echo "Available targets:"
	@awk '/^^([a-zA-Z0-9_-]+):[ \t]*##[ \t]+(.+)/' $(MAKEFILE_LIST) | column -t -s ':'

.PHONY: startWebAI

startWebAI: ## demare une interface web avec les ia
	@$(PYTHON_PATH) ./src/app/app_GGUF.py

.PHONY: GPUutilisation

GPUutilisation: ## affiche l'utilisation de la carte graphique !!ne pas lancé de scripte si qqc tourne deja!!
	@nvidia-smi

