cant ?= 100
cant_sol ?= 10

.PHONY: dev
dev:
	python3 src/main.py

.PHONY: scrapper
scrapper:
	python3 src/scrapper.py $(cant) $(cant_sol)
