VENV_PATH = ~/dataSSD/venv/bin/activate

.PHONY: help
help: ## list toutes les commandes
	@echo "Available targets:"
	@awk '/^^([a-zA-Z0-9_-]+):[ \t]*##[ \t]+(.+)/' $(MAKEFILE_LIST) | column -t -s ':'

.PHONY: startWebAI

startWebAI: ## demare une interface web avec les ia
	@python ./src/app/app_GGUF.py