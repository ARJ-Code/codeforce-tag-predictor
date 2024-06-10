cant ?= 100

.PHONY: dev
dev:
	python3 src/main.py

.PHONY: scrapper
scrapper:
	python3 src/scrapper.py $(cant)
