VENV_PATH = ./venv/bin/activate
PYTHON_PATH = ./venv/bin/python3

.PHONY: help startWebUI setup
help: ## Liste toutes les commandes 
	@echo "Available targets:"
	@awk '/^^([a-zA-Z0-9_-]+):[ \t]*##[ \t]+(.+)/' $(MAKEFILE_LIST) | column -t -s ':'

startWebUI: ## Démarre l'interface web
	@$(PYTHON_PATH) ./src/app.py

setup: ## Crée l'environnement virtuel et installe toutes les dépendances
	@python3 -m venv venv
	@$(PYTHON_PATH) -m pip install -r ./requirements.txt
