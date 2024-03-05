VENV_PATH = ./venv/bin/activate
PYTHON_PATH = ./venv/bin/python3

.PHONY: help startWebUI setup
help: ## list toutes les commandes
	@echo "Available targets:"
	@awk '/^^([a-zA-Z0-9_-]+):[ \t]*##[ \t]+(.+)/' $(MAKEFILE_LIST) | column -t -s ':'

startWebUI: ## demare une interface web avec les ia
	@$(PYTHON_PATH) ./src/app.py

setup: ## create the python venv and install all the dependancy
	@python3 -m venv venv
	@$(PYTHON_PATH) -m pip install -r ./requirements.txt
